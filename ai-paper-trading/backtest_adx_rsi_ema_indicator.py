"""
Backtest: ADX RSI EMA Setting (mirrors the TradingView Pine Script logic 1:1)

Entry rules (from the indicator):
  SELL: close < EMA  AND  ADX(5) > 30  AND  red candle  AND  RSI(3) crosses back
        below 80 this bar (i.e. RSI[1] > 80 and RSI[0] <= 80)
  BUY:  close > EMA  AND  ADX(5) > 30  AND  green candle AND  RSI(3) crosses back
        above 20 this bar (i.e. RSI[1] < 20 and RSI[0] >= 20)

Exit rule (configurable): opposite signal, or fixed pip SL/TP using bar high/low
for realistic fills (matches your existing run_backtest convention).

Data source priority: MetaTrader5 (live/demo terminal) -> yfinance fallback.
Pip convention (Finex broker, XAUUSD): point = 0.01, 1 pip = 0.1 price units.

Run this locally where MT5 terminal / yfinance is available - it will NOT run
in a sandboxed environment with no market-data access.
"""

import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ==================== CONFIG ====================
SYMBOL          = "XAUUSD"      # MT5 symbol name (check your broker's exact suffix, e.g. "XAUUSD" or "XAUUSDm")
TIMEFRAME_MIN   = 5             # M5, matches ADX(5)/RSI(3) short-period design

EMA_LEN         = 20
RSI_LEN         = 3
RSI_OB          = 80
RSI_OS          = 20
ADX_LEN         = 5
ADX_SMOOTH      = 5
ADX_LEVEL       = 30

CMF_LEN         = 20            # Chaikin Money Flow lookback
RVOL_LEN        = 20            # relative-volume lookback
RVOL_MIN        = 1.0           # used only if REQUIRE_VOLUME_CONFIRM is enabled below
REQUIRE_VOLUME_CONFIRM = False  # opt-in filter: if True, signals must also have rvol >= RVOL_MIN

POINT           = 0.01          # Finex XAUUSD: 2 decimal digits
PIP_SIZE        = POINT * 10    # 1 pip = 0.1 price units

EXIT_MODE       = "opposite_signal"   # "opposite_signal" or "fixed_sltp"
SL_PIPS         = 50
TP_PIPS         = 100
SPREAD_PIPS     = 3.0           # modelled cost per round trip - adjust to your live spread

CONTRACT_SIZE   = 100           # oz per 1.0 standard lot (typical XAUUSD contract - verify with Finex)
# Pip value per 1.0 lot = CONTRACT_SIZE * PIP_SIZE = 100 * 0.1 = $10/pip/lot (standard convention)

# ---- Break & Retest strategy config (alternative to the ADX/RSI/EMA logic above) ----
# Three setups, price-action only, no indicators:
#   1. Previous day high/low break + retest
#   2. Opening range break + retest (first bar of each session at this data's timeframe)
#   3. Order block retest (last opposite-colored candle before an impulse move)
BR_RETEST_TOLERANCE_PIPS   = 15.0   # how close price must come back to a level to count as "retest"
BR_OB_TOLERANCE_PIPS       = 8.0    # tighter tolerance for order block zone touches (scalp setup)
BR_OB_LOOKBACK_BARS        = 20     # how far back to search for the last opposite-colored candle
BR_ENABLE_PREV_DAY         = True   # toggle each setup independently for isolated testing
BR_ENABLE_OPENING_RANGE    = True
BR_ENABLE_ORDER_BLOCK      = True


# ==================== USER INPUT: CAPITAL, LOT SIZE, DATE RANGE ====================
def resolve_date_range(start_date_str, duration_days, now=None):
    """Turn an optional YYYY-MM-DD start date + duration into a concrete (date_from, date_to)
    window. Blank start date -> last `duration_days` days ending now. Shared by the CLI
    prompts and the MCP server so both apply identical defaulting logic."""
    now = now or datetime.now()
    if start_date_str:
        try:
            date_from = datetime.strptime(start_date_str, "%Y-%m-%d")
        except ValueError:
            date_from = now - timedelta(days=duration_days)
        date_to = min(date_from + timedelta(days=duration_days), now)
    else:
        date_to = now
        date_from = now - timedelta(days=duration_days)
    return date_from, date_to


def get_account_inputs():
    parser = argparse.ArgumentParser(description="ADX RSI EMA backtest")
    parser.add_argument("--capital", type=float, default=None, help="Starting capital in USD")
    parser.add_argument("--lot", type=float, default=None, help="Fixed lot size per trade (e.g. 0.01)")
    parser.add_argument("--start-date", type=str, default=None, help="Backtest start date, format YYYY-MM-DD")
    parser.add_argument("--days", type=int, default=None, help="Backtest duration in days")
    parser.add_argument("--strategy", type=str, default="adx_rsi_ema",
                         choices=["adx_rsi_ema", "break_retest"],
                         help="Which strategy to backtest: 'adx_rsi_ema' (default) or 'break_retest'")
    args, _ = parser.parse_known_args()

    capital = args.capital
    if capital is None:
        try:
            capital = float(input("Enter starting capital (USD): ").strip())
        except (ValueError, EOFError):
            capital = 1000.0
            print(f"No/invalid input - defaulting capital to ${capital:.2f}")

    lot = args.lot
    if lot is None:
        try:
            raw = input("Enter lot size per trade (default 0.01): ").strip()
            lot = float(raw) if raw else 0.01
        except (ValueError, EOFError):
            lot = 0.01
            print(f"No/invalid input - defaulting lot size to {lot}")

    start_date_str = args.start_date
    if start_date_str is None:
        try:
            start_date_str = input("Enter start date YYYY-MM-DD (blank = last week from today): ").strip()
        except EOFError:
            start_date_str = ""

    duration_days = args.days
    if duration_days is None:
        try:
            raw = input("Enter backtest duration in days (default 7): ").strip()
            duration_days = int(raw) if raw else 7
        except (ValueError, EOFError):
            duration_days = 7
            print(f"No/invalid input - defaulting duration to {duration_days} days")

    now = datetime.now()
    date_from, date_to = resolve_date_range(start_date_str, duration_days, now)

    print(f"Backtest window  : {date_from:%Y-%m-%d} -> {date_to:%Y-%m-%d}  ({duration_days} days)")

    # ---- SL/TP sweep mode ----
    parser.add_argument("--sweep", action="store_true", help="Run an SL/TP sweep instead of a single backtest")
    parser.add_argument("--sl-values", type=str, default=None, help="Comma-separated SL pip values to test")
    parser.add_argument("--tp-values", type=str, default=None, help="Comma-separated TP pip values to test")
    args2, _ = parser.parse_known_args()

    run_sweep = args2.sweep
    if not run_sweep and args.start_date is None and args.capital is None:
        # only prompt interactively if the user isn't already scripting this via CLI flags
        try:
            ans = input("Run an SL/TP sweep to find the best stop loss? (y/N): ").strip().lower()
            run_sweep = ans == "y"
        except EOFError:
            run_sweep = False

    default_sl_list = [10, 20, 30, 50, 75, 100, 150, 200]
    default_tp_list = [50, 100, 150, 200]
    sl_values, tp_values = default_sl_list, default_tp_list

    if run_sweep:
        sl_str = args2.sl_values
        if sl_str is None:
            try:
                raw = input(f"SL pip values to test, comma-separated (default {default_sl_list}): ").strip()
                sl_str = raw if raw else None
            except EOFError:
                sl_str = None
        if sl_str:
            sl_values = [int(x.strip()) for x in sl_str.split(",") if x.strip()]

        tp_str = args2.tp_values
        if tp_str is None:
            try:
                raw = input(f"TP pip values to test, comma-separated (default {default_tp_list}): ").strip()
                tp_str = raw if raw else None
            except EOFError:
                tp_str = None
        if tp_str:
            tp_values = [int(x.strip()) for x in tp_str.split(",") if x.strip()]

        print(f"Sweep enabled    : SL={sl_values}  TP={tp_values}")

    return capital, lot, date_from, date_to, run_sweep, sl_values, tp_values, args.strategy


# ==================== DATA FETCH ====================
def fetch_mt5(symbol, minutes, date_from, date_to):
    import importlib
    try:
        mt5 = importlib.import_module("MetaTrader5")
    except ImportError as e:
        raise RuntimeError("MetaTrader5 package is not installed. Install it with `pip install MetaTrader5`.") from e

    if not mt5.initialize():
        raise RuntimeError(f"MT5 init failed: {mt5.last_error()}")
    tf_map = {1: mt5.TIMEFRAME_M1, 5: mt5.TIMEFRAME_M5, 15: mt5.TIMEFRAME_M15, 60: mt5.TIMEFRAME_H1}
    tf = tf_map[minutes]
    rates = mt5.copy_rates_range(symbol, tf, date_from, date_to)
    mt5.shutdown()
    if rates is None or len(rates) == 0:
        raise RuntimeError("No MT5 data returned - check symbol name / market watch visibility / date range")
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close",
                             "tick_volume": "Volume"})
    # MT5 doesn't report true traded volume for OTC symbols like XAUUSD - tick_volume
    # (number of price changes per bar) is the standard proxy used here.
    if "Volume" not in df.columns:
        df["Volume"] = np.nan
    return df.set_index("time")[["Open", "High", "Low", "Close", "Volume"]]


def fetch_yfinance(date_from, date_to):
    import yfinance as yf
    # Yahoo intraday (5m) history is only available for the last ~60 days.
    df = yf.download("GC=F", start=date_from, end=date_to, interval="5m", progress=False)
    if df.empty:
        raise RuntimeError("No yfinance data returned - Yahoo only keeps ~60 days of 5m history")
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    if "Volume" not in df.columns:
        df["Volume"] = np.nan
    return df[["Open", "High", "Low", "Close", "Volume"]]


def get_data(date_from, date_to):
    try:
        print("Fetching from MT5...")
        return fetch_mt5(SYMBOL, TIMEFRAME_MIN, date_from, date_to)
    except Exception as e:
        print(f"MT5 fetch failed ({e}), falling back to yfinance (GC=F, delayed)...")
        return fetch_yfinance(date_from, date_to)


# ---- LIVE / LATEST-BARS FETCH (used by the live signal monitor & MCP "current signal" tool) ----
# Free data source options, tried in this order:
#   1. MT5 terminal      - your existing demo/live connection, true real-time, zero extra signup.
#   2. OANDA practice API - free practice account you already use; near-real-time (poll every
#                            few seconds), genuinely free, no card required. Set OANDA_API_KEY
#                            in your environment.
#   3. yfinance (GC=F)    - no signup needed at all, but delayed ~15-20 min and rate-limited by
#                            Yahoo informally. Last-resort fallback only.
def fetch_latest_mt5(symbol, minutes, n_bars):
    import importlib
    try:
        mt5 = importlib.import_module("MetaTrader5")
    except ImportError as e:
        raise RuntimeError("MetaTrader5 package is not installed. Install it with `pip install MetaTrader5`.") from e
    if not mt5.initialize():
        raise RuntimeError(f"MT5 init failed: {mt5.last_error()}")
    tf_map = {1: mt5.TIMEFRAME_M1, 5: mt5.TIMEFRAME_M5, 15: mt5.TIMEFRAME_M15, 60: mt5.TIMEFRAME_H1}
    tf = tf_map[minutes]
    rates = mt5.copy_rates_from_pos(symbol, tf, 0, n_bars)
    mt5.shutdown()
    if rates is None or len(rates) == 0:
        raise RuntimeError("No MT5 data returned - check symbol name / market watch visibility")
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close",
                             "tick_volume": "Volume"})
    if "Volume" not in df.columns:
        df["Volume"] = np.nan
    return df.set_index("time")[["Open", "High", "Low", "Close", "Volume"]]


def fetch_latest_oanda(n_bars=200, instrument="XAU_USD", granularity="M5"):
    import os
    import requests
    api_key = os.environ.get("OANDA_API_KEY")
    if not api_key:
        raise RuntimeError("OANDA_API_KEY not set in environment")
    url = f"https://api-fxpractice.oanda.com/v3/instruments/{instrument}/candles"
    params = {"count": n_bars, "granularity": granularity, "price": "M"}
    headers = {"Authorization": f"Bearer {api_key}"}
    resp = requests.get(url, headers=headers, params=params, timeout=10)
    resp.raise_for_status()
    candles = resp.json().get("candles", [])
    if not candles:
        raise RuntimeError("No OANDA candles returned")
    rows = [{
        "time": c["time"], "Open": float(c["mid"]["o"]), "High": float(c["mid"]["h"]),
        "Low": float(c["mid"]["l"]), "Close": float(c["mid"]["c"]),
        "Volume": float(c.get("volume", np.nan)),
    } for c in candles if c.get("complete", True)]
    # OANDA "volume" is the count of price ticks in the candle (their own proxy,
    # same idea as MT5 tick_volume) - not a traded-contract count either.
    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["time"])
    return df.set_index("time")[["Open", "High", "Low", "Close", "Volume"]]


def fetch_latest_yfinance(n_bars=200):
    import yfinance as yf
    df = yf.download("GC=F", period="5d", interval="5m", progress=False)
    if df.empty:
        raise RuntimeError("No yfinance data returned")
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    if "Volume" not in df.columns:
        df["Volume"] = np.nan
    return df[["Open", "High", "Low", "Close", "Volume"]].tail(n_bars)


def get_latest_bars(n_bars=200, verbose=True):
    """Fetch the most recent n_bars for live signal checking, trying MT5 -> OANDA -> yfinance."""
    sources = [
        ("MT5", lambda: fetch_latest_mt5(SYMBOL, TIMEFRAME_MIN, n_bars)),
        ("OANDA", lambda: fetch_latest_oanda(n_bars)),
        ("yfinance", lambda: fetch_latest_yfinance(n_bars)),
    ]
    last_err = None
    for name, fn in sources:
        try:
            df = fn()
            if verbose:
                print(f"[data] using {name} ({len(df)} bars)")
            return df, name
        except Exception as e:
            last_err = e
            if verbose:
                print(f"[data] {name} failed: {e}")
    raise RuntimeError(f"All data sources failed. Last error: {last_err}")


# ==================== INDICATORS (no external 'ta' dependency) ====================
def ema(series, length):
    return series.ewm(span=length, adjust=False).mean()


def rsi(series, length):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def obv(close, volume):
    """On-Balance Volume: running total of volume, signed by the direction of
    each bar's close-to-close move. Confirms whether a price move has real
    participation behind it vs. drifting on thin activity."""
    direction = np.sign(close.diff().fillna(0))
    return (direction * volume.fillna(0)).cumsum()


def cmf(df, length=20):
    """Chaikin Money Flow: volume-weighted measure of buying vs selling
    pressure over `length` bars, using each bar's close position within its
    own high-low range. Range [-1, 1]; > 0 = net buying pressure."""
    high, low, close, volume = df["High"], df["Low"], df["Close"], df["Volume"]
    mf_mult = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    mf_vol = mf_mult * volume
    return mf_vol.rolling(length).sum() / volume.rolling(length).sum()


def relative_volume(volume, length=20):
    """Current bar's volume vs. its own rolling average - flags whether a
    signal fires on unusually high or low participation."""
    return volume / volume.rolling(length).mean()


def adx_di(df, length, smoothing):
    high, low, close = df["High"], df["Low"], df["Close"]
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = pd.concat([
        (high - low),
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1 / length, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / length, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / length, adjust=False).mean() / atr

    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
    adx_val = dx.ewm(alpha=1 / smoothing, adjust=False).mean()
    return plus_di, minus_di, adx_val


def build_signals(df, ema_len=None, rsi_len=None, rsi_ob=None, rsi_os=None,
                   adx_len=None, adx_smooth=None, adx_level=None):
    """Compute indicators + signals. Every parameter defaults to the module-level
    CONFIG constant if not passed, so existing callers (CLI, live monitor) are
    unaffected, while the MCP server can override any of them per-call to let
    you experiment with different settings without editing this file."""
    ema_len = EMA_LEN if ema_len is None else ema_len
    rsi_len = RSI_LEN if rsi_len is None else rsi_len
    rsi_ob = RSI_OB if rsi_ob is None else rsi_ob
    rsi_os = RSI_OS if rsi_os is None else rsi_os
    adx_len = ADX_LEN if adx_len is None else adx_len
    adx_smooth = ADX_SMOOTH if adx_smooth is None else adx_smooth
    adx_level = ADX_LEVEL if adx_level is None else adx_level

    df = df.copy()
    df["EMA"] = ema(df["Close"], ema_len)
    df["RSI"] = rsi(df["Close"], rsi_len)
    df["DIplus"], df["DIminus"], df["ADX"] = adx_di(df, adx_len, adx_smooth)

    df["is_red"] = df["Close"] < df["Open"]
    df["is_green"] = df["Close"] > df["Open"]
    df["trend_down"] = df["Close"] < df["EMA"]
    df["trend_up"] = df["Close"] > df["EMA"]
    df["strong_vol"] = df["ADX"] > adx_level

    df["rsi_prev"] = df["RSI"].shift(1)
    df["rsi_exit_ob"] = (df["rsi_prev"] > rsi_ob) & (df["RSI"] <= rsi_ob)
    df["rsi_exit_os"] = (df["rsi_prev"] < rsi_os) & (df["RSI"] >= rsi_os)

    # --- volume-based indicators (reference / optional filter, not part of the
    # original TradingView logic - see module docstring) ---
    if "Volume" in df.columns and df["Volume"].notna().any():
        df["OBV"] = obv(df["Close"], df["Volume"])
        df["CMF"] = cmf(df, CMF_LEN)
        df["RVOL"] = relative_volume(df["Volume"], RVOL_LEN)
        vol_confirm = df["RVOL"] >= RVOL_MIN
    else:
        df["OBV"] = np.nan
        df["CMF"] = np.nan
        df["RVOL"] = np.nan
        vol_confirm = pd.Series(True, index=df.index)  # no volume data -> filter is a no-op

    df["sell_signal"] = df["trend_down"] & df["strong_vol"] & df["is_red"] & df["rsi_exit_ob"]
    df["buy_signal"] = df["trend_up"] & df["strong_vol"] & df["is_green"] & df["rsi_exit_os"]

    if REQUIRE_VOLUME_CONFIRM:
        df["sell_signal"] &= vol_confirm
        df["buy_signal"] &= vol_confirm

    return df


# ==================== ALTERNATIVE STRATEGY: BREAK & RETEST ====================
# Price-action only, no indicators. Three setups from the "break and retest"
# methodology: previous day high/low retest, opening range retest, and order
# block retest. Produces the same buy_signal / sell_signal columns as
# build_signals() above so it's a drop-in alternative for run_backtest().
def compute_prev_day_levels(df):
    """Attach previous calendar day's High/Low to every intraday row."""
    daily = df.resample("1D").agg({"High": "max", "Low": "min"}).dropna()
    daily["prev_day_high"] = daily["High"].shift(1)
    daily["prev_day_low"] = daily["Low"].shift(1)

    session_date = df.index.normalize()
    out = df.copy()
    out["prev_day_high"] = session_date.map(daily["prev_day_high"])
    out["prev_day_low"] = session_date.map(daily["prev_day_low"])
    return out


def compute_opening_range_levels(df):
    """Opening range = the first bar of each calendar day at this dataframe's
    own timeframe (e.g. the first 5m candle of the day on M5 data). This
    sidesteps timezone mismatches between MT5/OANDA/yfinance feeds - if you
    want a strict NY-09:30 opening range instead, filter df to session-open
    bars before calling this."""
    session_date = df.index.normalize()
    out = df.copy()
    out["_date"] = session_date
    out["opening_range_high"] = out.groupby("_date")["High"].transform("first")
    out["opening_range_low"] = out.groupby("_date")["Low"].transform("first")
    out["_is_first_bar_of_day"] = ~out["_date"].duplicated(keep="first")
    return out.drop(columns="_date")


def _break_retest_reaction_signals(df, level_high_col, level_low_col, tolerance_pips):
    """Vectorized break -> retest -> reaction pattern against a given pair of
    level columns (e.g. prev_day_high/low or opening_range_high/low):
      bar t-2: price breaks through the level
      bar t-1: price comes back to retest it
      bar t  : reaction candle confirms (closes back on the breakout side)
    Returns (long_signal, short_signal) boolean Series aligned to df.index.
    """
    tol = tolerance_pips * PIP_SIZE

    broke_above = df["High"].shift(2) > df[level_high_col]
    retested_above = df["Low"].shift(1) <= df[level_high_col] + tol
    reaction_long = (df["Close"] > df[level_high_col]) & (df["Close"] > df["Open"])
    long_signal = broke_above & retested_above & reaction_long

    broke_below = df["Low"].shift(2) < df[level_low_col]
    retested_below = df["High"].shift(1) >= df[level_low_col] - tol
    reaction_short = (df["Close"] < df[level_low_col]) & (df["Close"] < df["Open"])
    short_signal = broke_below & retested_below & reaction_short

    return long_signal.fillna(False), short_signal.fillna(False)


def _order_block_signals(df, lookback_bars, tolerance_pips):
    """Loop-based (not vectorized) since each bar needs to look back for the
    most recent opposite-colored candle - same O(n) cost as run_backtest's
    own trade-simulation loop, so this doesn't change the script's overall
    complexity class. Returns (long_signal, short_signal) as plain lists."""
    tol = tolerance_pips * PIP_SIZE
    is_up = (df["Close"] > df["Open"]).to_numpy()
    is_down = (df["Close"] < df["Open"]).to_numpy()
    open_ = df["Open"].to_numpy()
    high = df["High"].to_numpy()
    low = df["Low"].to_numpy()
    close = df["Close"].to_numpy()
    n = len(df)

    long_signal = [False] * n
    short_signal = [False] * n

    for i in range(1, n):
        window_start = max(0, i - lookback_bars)

        # --- bearish order block: last up-close candle before this bar, for shorts ---
        ob_idx = None
        for j in range(i - 1, window_start - 1, -1):
            if is_up[j]:
                ob_idx = j
                break
        if ob_idx is not None:
            zone_top = max(open_[ob_idx], close[ob_idx])   # body top
            zone_bottom = low[ob_idx]                       # candle low (wick)
            touched = (zone_bottom - tol) <= high[i] <= (zone_top + tol)
            weak_reaction = is_down[i] and close[i] < close[i - 1]
            if touched and weak_reaction:
                short_signal[i] = True

        # --- bullish order block: last down-close candle before this bar, for longs ---
        ob_idx = None
        for j in range(i - 1, window_start - 1, -1):
            if is_down[j]:
                ob_idx = j
                break
        if ob_idx is not None:
            zone_bottom = min(open_[ob_idx], close[ob_idx])  # body bottom
            zone_top = high[ob_idx]                            # candle high (wick)
            touched = (zone_bottom - tol) <= low[i] <= (zone_top + tol)
            strong_reaction = is_up[i] and close[i] > close[i - 1]
            if touched and strong_reaction:
                long_signal[i] = True

    return long_signal, short_signal


def build_break_retest_signals(df, retest_tolerance_pips=None, ob_tolerance_pips=None,
                                ob_lookback_bars=None, enable_prev_day=None,
                                enable_opening_range=None, enable_order_block=None):
    """Alternative to build_signals(): price-action break-and-retest logic
    instead of ADX/RSI/EMA. Produces the same buy_signal/sell_signal columns
    so it plugs straight into the existing run_backtest() / generate chart /
    summarize pipeline. Also adds a 'setup' column naming which of the three
    setups fired, so trade logs and summaries can break results down by setup.
    """
    retest_tolerance_pips = BR_RETEST_TOLERANCE_PIPS if retest_tolerance_pips is None else retest_tolerance_pips
    ob_tolerance_pips = BR_OB_TOLERANCE_PIPS if ob_tolerance_pips is None else ob_tolerance_pips
    ob_lookback_bars = BR_OB_LOOKBACK_BARS if ob_lookback_bars is None else ob_lookback_bars
    enable_prev_day = BR_ENABLE_PREV_DAY if enable_prev_day is None else enable_prev_day
    enable_opening_range = BR_ENABLE_OPENING_RANGE if enable_opening_range is None else enable_opening_range
    enable_order_block = BR_ENABLE_ORDER_BLOCK if enable_order_block is None else enable_order_block

    df = compute_prev_day_levels(df)
    df = compute_opening_range_levels(df)

    n = len(df)
    pdh_long = pdh_short = pd.Series(False, index=df.index)
    or_long = or_short = pd.Series(False, index=df.index)
    ob_long = ob_short = pd.Series(False, index=df.index)

    if enable_prev_day:
        pdh_long, pdh_short = _break_retest_reaction_signals(
            df, "prev_day_high", "prev_day_low", retest_tolerance_pips)

    if enable_opening_range:
        or_long, or_short = _break_retest_reaction_signals(
            df, "opening_range_high", "opening_range_low", retest_tolerance_pips)
        # don't fire on the opening-range bar itself or before it's established
        or_long = or_long & ~df["_is_first_bar_of_day"]
        or_short = or_short & ~df["_is_first_bar_of_day"]

    if enable_order_block:
        ob_long_list, ob_short_list = _order_block_signals(df, ob_lookback_bars, ob_tolerance_pips)
        ob_long = pd.Series(ob_long_list, index=df.index)
        ob_short = pd.Series(ob_short_list, index=df.index)

    df["buy_signal"] = pdh_long | or_long | ob_long
    df["sell_signal"] = pdh_short | or_short | ob_short

    setup = pd.Series("", index=df.index, dtype=object)
    setup = setup.mask(pdh_long | pdh_short, "previous_day_high_low_retest")
    setup = setup.mask(or_long | or_short, "opening_range_retest")
    setup = setup.mask(ob_long | ob_short, "order_block_retest")
    df["setup"] = setup

    return df.drop(columns="_is_first_bar_of_day")


def generate_break_retest_chart(signals, trades=None, title="XAUUSD - Break & Retest", lookback_bars=500):
    """Chart variant for the break-retest strategy: overlays previous-day
    high/low and opening-range levels instead of EMA/RSI/ADX panels, since
    this strategy has no indicators to plot."""
    import io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = signals if lookback_bars is None else signals.tail(lookback_bars)

    fig, ax_price = plt.subplots(1, 1, figsize=(13, 6))

    ax_price.plot(df.index, df["Close"], color="#1f77b4", linewidth=1.0, label="Close", zorder=2)
    ax_price.plot(df.index, df["prev_day_high"], color="#d62728", linewidth=0.8,
                  linestyle="--", alpha=0.7, label="Prev day high")
    ax_price.plot(df.index, df["prev_day_low"], color="#2ca02c", linewidth=0.8,
                  linestyle="--", alpha=0.7, label="Prev day low")
    ax_price.plot(df.index, df["opening_range_high"], color="#9467bd", linewidth=0.8,
                  linestyle=":", alpha=0.7, label="Opening range high")
    ax_price.plot(df.index, df["opening_range_low"], color="#8c564b", linewidth=0.8,
                  linestyle=":", alpha=0.7, label="Opening range low")

    buys = df[df["buy_signal"]]
    sells = df[df["sell_signal"]]
    if len(buys):
        ax_price.scatter(buys.index, buys["Close"], marker="^", color="lime", s=80,
                          zorder=5, edgecolors="black", linewidths=0.5, label="BUY signal")
    if len(sells):
        ax_price.scatter(sells.index, sells["Close"], marker="v", color="red", s=80,
                          zorder=5, edgecolors="black", linewidths=0.5, label="SELL signal")

    if trades is not None and len(trades):
        t = trades[(trades["entry_time"] >= df.index.min()) & (trades["entry_time"] <= df.index.max())]
        for _, tr in t.iterrows():
            color = "green" if tr["pnl_usd"] > 0 else "crimson"
            ax_price.plot([tr["entry_time"], tr["exit_time"]], [tr["entry_price"], tr["exit_price"]],
                          color=color, linestyle="--", linewidth=1.3, alpha=0.85, zorder=4)
            entry_marker = "^" if tr["side"] == "BUY" else "v"
            ax_price.scatter([tr["entry_time"]], [tr["entry_price"]], marker=entry_marker, s=70,
                             facecolors="none", edgecolors="black", linewidths=1.3, zorder=6)
            ax_price.scatter([tr["exit_time"]], [tr["exit_price"]], marker="x", s=60,
                             color=color, linewidths=1.6, zorder=6)

    ax_price.set_title(title)
    ax_price.legend(loc="upper left", fontsize=8, ncol=2)
    ax_price.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    return buf.getvalue()


def generate_signal_chart(signals, trades=None, title="XAUUSD - ADX/RSI/EMA", lookback_bars=500,
                           rsi_ob=None, rsi_os=None, adx_level=None):
    """
    Render a 3-panel PNG chart (price+EMA with BUY/SELL markers, RSI, ADX) and
    return it as raw PNG bytes, ready to wrap in an MCP Image() content block.

    If `trades` (the DataFrame returned by run_backtest) is supplied, each trade's
    entry/exit is drawn as a dashed line - green if it closed in profit, red if it
    closed at a loss - so you can see exactly why a trade worked or didn't.

    `lookback_bars` caps how many of the most recent bars are plotted, since a
    multi-week M5 backtest can be tens of thousands of bars and unreadable on one
    chart; pass None to plot the full window.
    """
    import io
    import matplotlib
    matplotlib.use("Agg")  # headless - this runs inside an MCP server, no display available
    import matplotlib.pyplot as plt

    rsi_ob = RSI_OB if rsi_ob is None else rsi_ob
    rsi_os = RSI_OS if rsi_os is None else rsi_os
    adx_level = ADX_LEVEL if adx_level is None else adx_level

    df = signals if lookback_bars is None else signals.tail(lookback_bars)

    fig, (ax_price, ax_rsi, ax_adx) = plt.subplots(
        3, 1, figsize=(13, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1, 1]},
    )

    # ---- price + EMA ----
    ax_price.plot(df.index, df["Close"], color="#1f77b4", linewidth=1.0, label="Close")
    ax_price.plot(df.index, df["EMA"], color="#ff7f0e", linewidth=1.0, label="EMA")

    buys = df[df["buy_signal"]]
    sells = df[df["sell_signal"]]
    if len(buys):
        ax_price.scatter(buys.index, buys["Close"], marker="^", color="lime", s=80,
                          zorder=5, edgecolors="black", linewidths=0.5, label="BUY signal")
    if len(sells):
        ax_price.scatter(sells.index, sells["Close"], marker="v", color="red", s=80,
                          zorder=5, edgecolors="black", linewidths=0.5, label="SELL signal")

    # ---- trade entries/exits, colored by outcome ----
    if trades is not None and len(trades):
        t = trades[(trades["entry_time"] >= df.index.min()) & (trades["entry_time"] <= df.index.max())]
        for _, tr in t.iterrows():
            color = "green" if tr["pnl_usd"] > 0 else "crimson"
            ax_price.plot([tr["entry_time"], tr["exit_time"]], [tr["entry_price"], tr["exit_price"]],
                          color=color, linestyle="--", linewidth=1.3, alpha=0.85, zorder=4)
            entry_marker = "^" if tr["side"] == "BUY" else "v"
            ax_price.scatter([tr["entry_time"]], [tr["entry_price"]], marker=entry_marker, s=70,
                             facecolors="none", edgecolors="black", linewidths=1.3, zorder=6)
            ax_price.scatter([tr["exit_time"]], [tr["exit_price"]], marker="x", s=60,
                             color=color, linewidths=1.6, zorder=6)

    ax_price.set_title(title)
    ax_price.legend(loc="upper left", fontsize=8, ncol=2)
    ax_price.grid(alpha=0.3)

    # ---- RSI ----
    ax_rsi.plot(df.index, df["RSI"], color="#9467bd", linewidth=1)
    ax_rsi.axhline(rsi_ob, color="gray", linestyle="--", linewidth=0.7)
    ax_rsi.axhline(rsi_os, color="gray", linestyle="--", linewidth=0.7)
    ax_rsi.set_ylabel("RSI")
    ax_rsi.grid(alpha=0.3)

    # ---- ADX ----
    ax_adx.plot(df.index, df["ADX"], color="#2ca02c", linewidth=1)
    ax_adx.axhline(adx_level, color="gray", linestyle="--", linewidth=0.7)
    ax_adx.set_ylabel("ADX")
    ax_adx.grid(alpha=0.3)

    fig.autofmt_xdate()
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    return buf.getvalue()


# ==================== BACKTEST ====================
def run_backtest(df, lot_size, exit_mode=None, sl_pips=None, tp_pips=None, spread_pips=None):
    exit_mode = exit_mode or EXIT_MODE
    sl_pips = SL_PIPS if sl_pips is None else sl_pips
    tp_pips = TP_PIPS if tp_pips is None else tp_pips
    spread_pips = SPREAD_PIPS if spread_pips is None else spread_pips

    pip_value = CONTRACT_SIZE * PIP_SIZE * lot_size  # $ per pip for this lot size
    trades = []
    position = None  # dict: side, entry_price, entry_time, sl, tp

    for i in range(1, len(df)):
        row = df.iloc[i]
        time = df.index[i]

        # --- manage open position ---
        if position is not None:
            exit_now, exit_price, reason = False, None, None

            if exit_mode == "fixed_sltp":
                if position["side"] == "BUY":
                    if row["Low"] <= position["sl"]:
                        exit_now, exit_price, reason = True, position["sl"], "SL"
                    elif row["High"] >= position["tp"]:
                        exit_now, exit_price, reason = True, position["tp"], "TP"
                else:
                    if row["High"] >= position["sl"]:
                        exit_now, exit_price, reason = True, position["sl"], "SL"
                    elif row["Low"] <= position["tp"]:
                        exit_now, exit_price, reason = True, position["tp"], "TP"

            if not exit_now and exit_mode == "opposite_signal":
                if position["side"] == "BUY" and row["sell_signal"]:
                    exit_now, exit_price, reason = True, row["Close"], "OPPOSITE_SIGNAL"
                elif position["side"] == "SELL" and row["buy_signal"]:
                    exit_now, exit_price, reason = True, row["Close"], "OPPOSITE_SIGNAL"

            if exit_now:
                raw_diff = (exit_price - position["entry_price"]) if position["side"] == "BUY" \
                    else (position["entry_price"] - exit_price)
                pips = (raw_diff / PIP_SIZE) - spread_pips
                pnl_usd = pips * pip_value
                trades.append({
                    "side": position["side"], "entry_time": position["entry_time"],
                    "exit_time": time, "entry_price": position["entry_price"],
                    "exit_price": exit_price, "pips": pips, "pnl_usd": pnl_usd,
                    "reason": reason,
                    "entry_volume": position.get("entry_volume"),
                    "entry_rvol": position.get("entry_rvol"),
                    "entry_cmf": position.get("entry_cmf"),
                    "setup": position.get("setup"),
                })
                position = None

        # --- open new position ---
        if position is None:
            if row["buy_signal"]:
                position = {
                    "side": "BUY", "entry_price": row["Close"], "entry_time": time,
                    "sl": row["Close"] - sl_pips * PIP_SIZE,
                    "tp": row["Close"] + tp_pips * PIP_SIZE,
                    "entry_volume": row.get("Volume"), "entry_rvol": row.get("RVOL"),
                    "entry_cmf": row.get("CMF"), "setup": row.get("setup"),
                }
            elif row["sell_signal"]:
                position = {
                    "side": "SELL", "entry_price": row["Close"], "entry_time": time,
                    "sl": row["Close"] + sl_pips * PIP_SIZE,
                    "tp": row["Close"] - tp_pips * PIP_SIZE,
                    "entry_volume": row.get("Volume"), "entry_rvol": row.get("RVOL"),
                    "entry_cmf": row.get("CMF"), "setup": row.get("setup"),
                }

    return pd.DataFrame(trades)


def summarize(trades, capital, lot_size):
    if trades.empty:
        print("No trades were generated in this window.")
        return

    # running equity curve
    trades = trades.copy()
    trades["equity"] = capital + trades["pnl_usd"].cumsum()
    trades["equity_peak"] = trades["equity"].cummax()
    trades["drawdown_pct"] = (trades["equity"] - trades["equity_peak"]) / trades["equity_peak"] * 100

    wins = trades[trades["pips"] > 0]
    losses = trades[trades["pips"] <= 0]

    total_pnl_usd = trades["pnl_usd"].sum()
    final_balance = capital + total_pnl_usd
    total_return_pct = (total_pnl_usd / capital) * 100
    gross_profit = wins["pnl_usd"].sum() if len(wins) else 0.0
    gross_loss = losses["pnl_usd"].sum() if len(losses) else 0.0  # negative number
    max_dd_pct = trades["drawdown_pct"].min()

    print("=" * 55)
    print(f"Starting capital : ${capital:,.2f}   |   Lot size: {lot_size}")
    print("-" * 55)
    print(f"Total trades     : {len(trades)}")
    print(f"Buy / Sell       : {len(trades[trades.side=='BUY'])} / {len(trades[trades.side=='SELL'])}")
    print(f"Win rate         : {len(wins) / len(trades) * 100:.1f}%  ({len(wins)}W / {len(losses)}L)")
    print(f"Total pips       : {trades['pips'].sum():.1f}")
    print("-" * 55)
    print(f"Gross profit     : ${gross_profit:,.2f}")
    print(f"Gross loss       : ${gross_loss:,.2f}")
    print(f"Net P&L          : ${total_pnl_usd:,.2f}  ({total_return_pct:+.2f}% of capital)")
    print(f"Final balance    : ${final_balance:,.2f}")
    print(f"Max drawdown     : {max_dd_pct:.2f}%")
    print("-" * 55)
    print(f"Avg win          : ${wins['pnl_usd'].mean() if len(wins) else 0:,.2f}  "
          f"({wins['pnl_usd'].mean()/capital*100 if len(wins) else 0:+.2f}%)")
    print(f"Avg loss         : ${losses['pnl_usd'].mean() if len(losses) else 0:,.2f}  "
          f"({losses['pnl_usd'].mean()/capital*100 if len(losses) else 0:+.2f}%)")
    print(f"Best trade       : ${trades['pnl_usd'].max():,.2f}  "
          f"({trades['pnl_usd'].max()/capital*100:+.2f}%)")
    print(f"Worst trade      : ${trades['pnl_usd'].min():,.2f}  "
          f"({trades['pnl_usd'].min()/capital*100:+.2f}%)")

    if "setup" in trades.columns and trades["setup"].notna().any():
        print("-" * 55)
        print("Breakdown by setup:")
        for setup_name, grp in trades.groupby("setup"):
            grp_wins = grp[grp["pips"] > 0]
            print(f"  {setup_name:<32} {len(grp):>3} trades  "
                  f"win rate {len(grp_wins)/len(grp)*100:5.1f}%  "
                  f"net ${grp['pnl_usd'].sum():>9,.2f}")

    print("=" * 55)

    trades.to_csv("backtest_adx_rsi_ema_results.csv", index=False)
    print("Saved trade log -> backtest_adx_rsi_ema_results.csv")


def run_sl_tp_sweep(df, lot_size, capital, sl_values, tp_values):
    results = []
    for sl in sl_values:
        for tp in tp_values:
            trades = run_backtest(df, lot_size, exit_mode="fixed_sltp", sl_pips=sl, tp_pips=tp)
            if trades.empty:
                results.append({
                    "sl_pips": sl, "tp_pips": tp, "trades": 0, "win_rate_pct": 0.0,
                    "total_pips": 0.0, "net_pnl_usd": 0.0, "return_pct": 0.0, "max_drawdown_pct": 0.0
                })
                continue
            wins = trades[trades["pips"] > 0]
            equity = capital + trades["pnl_usd"].cumsum()
            peak = equity.cummax()
            max_dd_pct = ((equity - peak) / peak * 100).min()
            total_pnl = trades["pnl_usd"].sum()
            results.append({
                "sl_pips": sl, "tp_pips": tp, "trades": len(trades),
                "win_rate_pct": len(wins) / len(trades) * 100,
                "total_pips": trades["pips"].sum(),
                "net_pnl_usd": total_pnl,
                "return_pct": total_pnl / capital * 100,
                "max_drawdown_pct": max_dd_pct
            })
    return pd.DataFrame(results)


def summarize_sweep(sweep_df, capital):
    if sweep_df.empty or sweep_df["trades"].sum() == 0:
        print("No trades were generated for any SL/TP combination in this window.")
        return

    ranked = sweep_df.sort_values("net_pnl_usd", ascending=False).reset_index(drop=True)
    best = ranked.iloc[0]

    print("=" * 78)
    print(f"SL/TP SWEEP RESULTS  (starting capital: ${capital:,.2f}, lot: fixed per run)")
    print("-" * 78)
    print(f"{'SL':>6} {'TP':>6} {'Trades':>7} {'WinRate':>9} {'Pips':>9} "
          f"{'NetP&L':>10} {'Return%':>9} {'MaxDD%':>8}")
    for _, r in ranked.iterrows():
        print(f"{r.sl_pips:>6.0f} {r.tp_pips:>6.0f} {r.trades:>7.0f} {r.win_rate_pct:>8.1f}% "
              f"{r.total_pips:>9.1f} ${r.net_pnl_usd:>9.2f} {r.return_pct:>8.2f}% {r.max_drawdown_pct:>7.2f}%")
    print("-" * 78)
    print(f"BEST BY NET P&L  -> SL={best.sl_pips:.0f} pips, TP={best.tp_pips:.0f} pips  "
          f"| {best.trades:.0f} trades | win rate {best.win_rate_pct:.1f}% "
          f"| net ${best.net_pnl_usd:,.2f} ({best.return_pct:+.2f}%) | max DD {best.max_drawdown_pct:.2f}%")
    print("=" * 78)
    print("Note: results are on this single backtest window only - a handful of trades")
    print("can make one SL/TP combo look best by chance. Re-run over a longer window")
    print("or multiple periods before trusting this as a real optimum.")

    sweep_df.to_csv("sl_tp_sweep_results.csv", index=False)
    print("Saved full sweep -> sl_tp_sweep_results.csv")


def open_file_cross_platform(path):
    """Best-effort: open a file with the OS default viewer, so the chart PNG
    pops up automatically right after a CLI backtest run instead of you having
    to go find it in the folder."""
    import subprocess
    import sys
    try:
        if sys.platform.startswith("win"):
            import os
            os.startfile(path)  # Windows only
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])
    except Exception as e:
        print(f"(Couldn't auto-open the chart: {e}. Open it manually -> {path})")


if __name__ == "__main__":
    chart_parser = argparse.ArgumentParser(add_help=False)
    chart_parser.add_argument("--no-chart", action="store_true",
                               help="Skip saving/opening a chart PNG after the backtest")
    chart_parser.add_argument("--chart-file", type=str, default="backtest_chart.png",
                               help="Output PNG path for the chart (default: backtest_chart.png)")
    chart_args, _ = chart_parser.parse_known_args()

    capital, lot_size, date_from, date_to, run_sweep, sl_values, tp_values, strategy = get_account_inputs()
    data = get_data(date_from, date_to)

    if strategy == "break_retest":
        data = build_break_retest_signals(data)
        chart_fn = generate_break_retest_chart
        strategy_label = "Break & Retest"
    else:
        data = build_signals(data)
        chart_fn = generate_signal_chart
        strategy_label = "ADX/RSI/EMA"

    print(f"Strategy         : {strategy_label}")

    if run_sweep:
        sweep_results = run_sl_tp_sweep(data, lot_size, capital, sl_values, tp_values)
        summarize_sweep(sweep_results, capital)
    else:
        trades = run_backtest(data, lot_size)
        summarize(trades, capital, lot_size)

        if not chart_args.no_chart:
            title = f"XAUUSD ({strategy_label})  {date_from:%Y-%m-%d} to {date_to:%Y-%m-%d}  |  {len(trades)} trades"
            png_bytes = chart_fn(data, trades=trades if not trades.empty else None, title=title)
            with open(chart_args.chart_file, "wb") as f:
                f.write(png_bytes)
            print(f"Saved chart -> {chart_args.chart_file}")
            open_file_cross_platform(chart_args.chart_file)
