"""
Live XAUUSD chart with ADX / +DI / -DI, RSI, and crossover markers.
Runs two panels: M5 and H1, both refreshing continuously from your
local MetaTrader 5 terminal.

REQUIREMENTS (run once in Command Prompt on the Windows machine
running MT5):
    pip install MetaTrader5 pandas numpy matplotlib

HOW TO RUN:
    python live_adx_chart.py

NOTES:
- MT5 terminal must already be open and logged in to your account.
- This connects directly via the MetaTrader5 Python package (the same
  library the MCP server uses) -- it does not go through Claude or
  the MCP server, so it can update continuously on its own.
- Markers show +DI / -DI crossovers filtered by ADX strength and
  confirmed by RSI being on the same side of 50. This is a simple,
  commonly-used trend signal combo -- not a complete strategy.
  Treat markers as one input among several, not automated trade calls.
- Signals within the last few bars (FLASH_BARS_WINDOW) pulse in size/
  brightness and show a BUY/SELL label so new signals are hard to miss.
  The flash alternates each refresh cycle, so a shorter
  REFRESH_SECONDS makes the flash faster.
- A beep plays once per new signal (higher pitch for BUY, lower for
  SELL) via Windows' built-in winsound module -- no extra install
  needed. Set SOUND_ENABLED = False near the top to mute it.
- Every signal (and its ADX/DI/RSI values at that moment) is appended
  to signals_log.csv in the same folder as this script, so you have a
  running history even after closing the chart.
- Press 'b' or 's' with the chart window focused to trigger a test
  BUY/SELL alert (sound + log entry) immediately, without waiting for
  a real crossover -- useful for confirming everything works.
- This script only reads market data. It does not place, modify, or
  close any orders.
"""

import sys
import os
import csv
import time
import threading
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

try:
    import MetaTrader5 as mt5
except ImportError:
    sys.exit(
        "MetaTrader5 package not found. Install it with:\n"
        "    pip install MetaTrader5\n"
        "(Windows only -- this package will not install on Mac/Linux.)"
    )

try:
    import winsound
    SOUND_AVAILABLE = True
except ImportError:
    SOUND_AVAILABLE = False  # non-Windows fallback, shouldn't happen since MT5 is Windows-only

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SYMBOL = "XAUUSD"
TIMEFRAMES = {
    "M5": mt5.TIMEFRAME_M5,
    "H1": mt5.TIMEFRAME_H1,
}
BARS_TO_FETCH = 200        # how many candles to pull each refresh
ADX_PERIOD = 14
ADX_TREND_THRESHOLD = 20   # only flag crossovers when ADX is above this
RSI_PERIOD = 14
RSI_MIDLINE = 50           # RSI must agree with crossover direction vs this line
REFRESH_SECONDS = 5        # how often to poll MT5 for new data
FLASH_BARS_WINDOW = 3      # a signal within this many bars of the latest bar "flashes"
SOUND_ENABLED = True       # set False to disable beeps entirely
BUY_BEEP_FREQ = 1200       # Hz
SELL_BEEP_FREQ = 500       # Hz
BEEP_DURATION_MS = 300
LOG_FILE = "signals_log.csv"


def play_beep(frequency: int, duration_ms: int):
    """Play a beep on a background thread so it never blocks the chart UI."""
    if not (SOUND_ENABLED and SOUND_AVAILABLE):
        return
    threading.Thread(
        target=winsound.Beep, args=(frequency, duration_ms), daemon=True
    ).start()


# Tracks which signal bars (by timestamp) have already triggered a beep,
# per timeframe -- so each new signal only beeps once, not on every refresh
# while it's still in the flashing window.
alerted_signals = {tf_name: set() for tf_name in TIMEFRAMES}


def log_signal_to_csv(row_data: dict):
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row_data.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_data)


def check_and_alert(tf_name: str, df: pd.DataFrame):
    recent = df.iloc[-FLASH_BARS_WINDOW:]
    for ts, row in recent.iterrows():
        if row["bullish_cross"] and ts not in alerted_signals[tf_name]:
            alerted_signals[tf_name].add(ts)
            print(f"[{datetime.now():%H:%M:%S}] {tf_name} BUY signal at {ts}")
            play_beep(BUY_BEEP_FREQ, BEEP_DURATION_MS)
            log_signal_to_csv({
                "logged_at": datetime.now().isoformat(timespec="seconds"),
                "symbol": SYMBOL,
                "timeframe": tf_name,
                "signal": "BUY",
                "bar_time": ts.isoformat(),
                "price": round(row["close"], 3),
                "adx": round(row["adx"], 2),
                "plus_di": round(row["plus_di"], 2),
                "minus_di": round(row["minus_di"], 2),
                "rsi": round(row["rsi"], 2),
            })
        elif row["bearish_cross"] and ts not in alerted_signals[tf_name]:
            alerted_signals[tf_name].add(ts)
            print(f"[{datetime.now():%H:%M:%S}] {tf_name} SELL signal at {ts}")
            play_beep(SELL_BEEP_FREQ, BEEP_DURATION_MS)
            log_signal_to_csv({
                "logged_at": datetime.now().isoformat(timespec="seconds"),
                "symbol": SYMBOL,
                "timeframe": tf_name,
                "signal": "SELL",
                "bar_time": ts.isoformat(),
                "price": round(row["close"], 3),
                "adx": round(row["adx"], 2),
                "plus_di": round(row["plus_di"], 2),
                "minus_di": round(row["minus_di"], 2),
                "rsi": round(row["rsi"], 2),
            })


# ---------------------------------------------------------------------------
# Indicator calculation (Wilder's ADX / +DI / -DI)
# ---------------------------------------------------------------------------
def calculate_adx(df: pd.DataFrame, period: int = ADX_PERIOD) -> pd.DataFrame:
    high = df["high"]
    low = df["low"]
    close = df["close"]

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(
        alpha=1 / period, adjust=False
    ).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(
        alpha=1 / period, adjust=False
    ).mean() / atr

    dx = (plus_di - minus_di).abs() / (plus_di + minus_di) * 100
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()

    df["plus_di"] = plus_di
    df["minus_di"] = minus_di
    df["adx"] = adx
    return df


def calculate_rsi(df: pd.DataFrame, period: int = RSI_PERIOD) -> pd.DataFrame:
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

    rs = avg_gain / avg_loss
    df["rsi"] = 100 - (100 / (1 + rs))
    return df


def find_crossovers(df: pd.DataFrame) -> pd.DataFrame:
    """Flag bullish/bearish +DI/-DI crossovers filtered by ADX strength
    and confirmed by RSI being on the same side of the midline."""
    df["bullish_cross"] = (
        (df["plus_di"] > df["minus_di"])
        & (df["plus_di"].shift(1) <= df["minus_di"].shift(1))
        & (df["adx"] > ADX_TREND_THRESHOLD)
        & (df["rsi"] > RSI_MIDLINE)
    )
    df["bearish_cross"] = (
        (df["minus_di"] > df["plus_di"])
        & (df["minus_di"].shift(1) <= df["plus_di"].shift(1))
        & (df["adx"] > ADX_TREND_THRESHOLD)
        & (df["rsi"] < RSI_MIDLINE)
    )
    return df


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------
def fetch_data(symbol: str, timeframe) -> pd.DataFrame:
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, BARS_TO_FETCH)
    if rates is None or len(rates) == 0:
        raise RuntimeError(
            f"No data returned for {symbol}. Check that the symbol name "
            f"matches exactly what your broker uses (e.g. 'XAUUSD', "
            f"'XAUUSD.m', 'GOLD') and that it's visible in Market Watch."
        )
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.set_index("time", inplace=True)
    return df


# ---------------------------------------------------------------------------
# Setup MT5 connection
# ---------------------------------------------------------------------------
if not mt5.initialize():
    sys.exit(f"MT5 initialize() failed, error code = {mt5.last_error()}")

symbol_info = mt5.symbol_info(SYMBOL)
if symbol_info is None:
    mt5.shutdown()
    sys.exit(
        f"Symbol '{SYMBOL}' not found. Check the exact symbol name your "
        f"broker uses (open Market Watch in MT5 to see it)."
    )
if not symbol_info.visible:
    mt5.symbol_select(SYMBOL, True)

print(f"Connected to MT5. Streaming {SYMBOL} on {', '.join(TIMEFRAMES)}...")

# ---------------------------------------------------------------------------
# Plotting setup
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(
    nrows=3, ncols=2, figsize=(14, 10), sharex="col",
    gridspec_kw={"height_ratios": [3, 1, 1]},
)
fig.suptitle(
    f"{SYMBOL} — Live ADX / DI Crossover Signals   "
    f"(press 'b' = test BUY alert, 's' = test SELL alert)",
    fontsize=12,
)

tf_names = list(TIMEFRAMES.keys())  # ["M5", "H1"]

# Lets you confirm sound/logging work immediately, without waiting for a
# real crossover. Press 'b' or 's' anywhere in the chart window.
def on_key_press(event):
    if event.key == "b":
        print(f"[{datetime.now():%H:%M:%S}] TEST BUY alert triggered manually")
        play_beep(BUY_BEEP_FREQ, BEEP_DURATION_MS)
        log_signal_to_csv({
            "logged_at": datetime.now().isoformat(timespec="seconds"),
            "symbol": SYMBOL, "timeframe": "TEST", "signal": "BUY",
            "bar_time": "", "price": "", "adx": "", "plus_di": "",
            "minus_di": "", "rsi": "",
        })
    elif event.key == "s":
        print(f"[{datetime.now():%H:%M:%S}] TEST SELL alert triggered manually")
        play_beep(SELL_BEEP_FREQ, BEEP_DURATION_MS)
        log_signal_to_csv({
            "logged_at": datetime.now().isoformat(timespec="seconds"),
            "symbol": SYMBOL, "timeframe": "TEST", "signal": "SELL",
            "bar_time": "", "price": "", "adx": "", "plus_di": "",
            "minus_di": "", "rsi": "",
        })


def draw_panel(col: int, tf_name: str, df: pd.DataFrame, flash_on: bool):
    price_ax = axes[0, col]
    adx_ax = axes[1, col]
    rsi_ax = axes[2, col]
    price_ax.clear()
    adx_ax.clear()
    rsi_ax.clear()

    price_ax.plot(df.index, df["close"], color="#333333", linewidth=1, label="Close")
    price_ax.set_title(f"{tf_name}")

    bulls = df[df["bullish_cross"]]
    bears = df[df["bearish_cross"]]

    # Split into "recent" (flashes) vs older (static) signals
    latest_idx = len(df) - 1
    is_recent = df.index.to_series().apply(
        lambda t: (latest_idx - df.index.get_loc(t)) < FLASH_BARS_WINDOW
    )
    bulls_recent = bulls[is_recent.reindex(bulls.index, fill_value=False)]
    bulls_old = bulls.drop(bulls_recent.index)
    bears_recent = bears[is_recent.reindex(bears.index, fill_value=False)]
    bears_old = bears.drop(bears_recent.index)

    # Static (older) markers -- normal size
    price_ax.scatter(
        bulls_old.index, bulls_old["close"], marker="^", color="green",
        s=90, zorder=5, label="Buy signal"
    )
    price_ax.scatter(
        bears_old.index, bears_old["close"], marker="v", color="red",
        s=90, zorder=5, label="Sell signal"
    )

    # Flashing (recent) markers -- pulse size/alpha each refresh, plus label
    flash_size = 320 if flash_on else 140
    flash_alpha = 1.0 if flash_on else 0.45
    if not bulls_recent.empty:
        price_ax.scatter(
            bulls_recent.index, bulls_recent["close"], marker="^", color="lime",
            edgecolors="darkgreen", linewidths=1.5, s=flash_size, alpha=flash_alpha,
            zorder=6,
        )
        for t, row in bulls_recent.iterrows():
            price_ax.annotate(
                "BUY", (t, row["close"]),
                xytext=(0, 14), textcoords="offset points",
                ha="center", fontsize=9, fontweight="bold",
                color="white" if flash_on else "darkgreen",
                bbox=dict(
                    boxstyle="round,pad=0.25",
                    fc="green" if flash_on else "white",
                    ec="darkgreen", alpha=0.9,
                ),
            )
    if not bears_recent.empty:
        price_ax.scatter(
            bears_recent.index, bears_recent["close"], marker="v", color="orangered",
            edgecolors="darkred", linewidths=1.5, s=flash_size, alpha=flash_alpha,
            zorder=6,
        )
        for t, row in bears_recent.iterrows():
            price_ax.annotate(
                "SELL", (t, row["close"]),
                xytext=(0, -18), textcoords="offset points",
                ha="center", fontsize=9, fontweight="bold",
                color="white" if flash_on else "darkred",
                bbox=dict(
                    boxstyle="round,pad=0.25",
                    fc="red" if flash_on else "white",
                    ec="darkred", alpha=0.9,
                ),
            )

    price_ax.legend(loc="upper left", fontsize=8)
    price_ax.grid(alpha=0.3)

    adx_ax.plot(df.index, df["adx"], color="black", linewidth=1.3, label="ADX")
    adx_ax.plot(df.index, df["plus_di"], color="green", linewidth=1, label="+DI")
    adx_ax.plot(df.index, df["minus_di"], color="red", linewidth=1, label="-DI")
    adx_ax.axhline(ADX_TREND_THRESHOLD, color="gray", linestyle="--", linewidth=0.8)
    adx_ax.legend(loc="upper left", fontsize=7)
    adx_ax.grid(alpha=0.3)

    rsi_ax.plot(df.index, df["rsi"], color="#7b2fbe", linewidth=1.2, label="RSI")
    rsi_ax.axhline(70, color="red", linestyle="--", linewidth=0.7)
    rsi_ax.axhline(RSI_MIDLINE, color="gray", linestyle="--", linewidth=0.7)
    rsi_ax.axhline(30, color="green", linestyle="--", linewidth=0.7)
    rsi_ax.set_ylim(0, 100)
    rsi_ax.legend(loc="upper left", fontsize=7)
    rsi_ax.grid(alpha=0.3)
    rsi_ax.tick_params(axis="x", rotation=30)


def update(frame):
    flash_on = (frame % 2 == 0)  # alternates every refresh -> pulsing effect
    for col, tf_name in enumerate(tf_names):
        try:
            df = fetch_data(SYMBOL, TIMEFRAMES[tf_name])
            df = calculate_adx(df)
            df = calculate_rsi(df)
            df = find_crossovers(df)
            check_and_alert(tf_name, df)
            draw_panel(col, tf_name, df, flash_on)
        except RuntimeError as e:
            print(f"[{datetime.now():%H:%M:%S}] {e}")
    fig.tight_layout(rect=[0, 0, 1, 0.96])


fig.canvas.mpl_connect("key_press_event", on_key_press)

ani = animation.FuncAnimation(
    fig, update, interval=REFRESH_SECONDS * 1000, cache_frame_data=False
)

print(f"Signal log will be written to: {os.path.abspath(LOG_FILE)}")
print("In the chart window: press 'b' to test a BUY alert, 's' to test a SELL alert.")

try:
    plt.show()
finally:
    mt5.shutdown()
    print("MT5 connection closed.")
