"""
MCP server exposing the ADX RSI EMA backtest and SL/TP sweep as tools for
Claude Desktop. This lets you say things like "backtest last week with
$1000 capital" or "find the best stop loss for the last 14 days" directly
in chat, and Claude will call this script, get structured results back,
and reason over them. It can also render a chart (price+EMA with BUY/SELL
markers and trade entries/exits, plus RSI/ADX panels) as an image.

Setup:
  1. pip install mcp pandas numpy matplotlib MetaTrader5 yfinance
  2. Make sure backtest_adx_rsi_ema_indicator.py lives in the SAME folder
     as this file (it's imported directly, not installed as a package).
  3. Add this server to claude_desktop_config.json (see bottom of this file
     for the exact JSON block) using the FULL ABSOLUTE PATH to your python
     executable - same fix you already applied for mt5-mcp's ENOENT/PATH
     ambiguity issue between Windows Store Python and Anaconda.
  4. Restart Claude Desktop. You should see "backtest-adx-rsi-ema" show up
     as an available tool source.

This server is READ-ONLY: it only fetches price data and simulates trades
on paper. It never places orders. That keeps it safe to run even while
your mt5-mcp connection has live-trading tools enabled.
"""

from datetime import datetime
from mcp.server import FastMCP
from mcp.server.fastmcp import Image

from backtest_adx_rsi_ema_indicator import (
    get_data,
    get_latest_bars,
    build_signals,
    run_backtest,
    run_sl_tp_sweep,
    resolve_date_range,
    generate_signal_chart,
)

mcp = FastMCP("backtest-adx-rsi-ema")


def _prepare_data(start_date: str, days: int, ema_len=None, rsi_len=None, rsi_ob=None,
                   rsi_os=None, adx_len=None, adx_smooth=None, adx_level=None):
    date_from, date_to = resolve_date_range(start_date, days)
    raw = get_data(date_from, date_to)
    signals = build_signals(raw, ema_len=ema_len, rsi_len=rsi_len, rsi_ob=rsi_ob,
                             rsi_os=rsi_os, adx_len=adx_len, adx_smooth=adx_smooth,
                             adx_level=adx_level)
    return signals, date_from, date_to


@mcp.tool()
def run_single_backtest(
    capital: float = 1000.0,
    lot_size: float = 0.01,
    start_date: str = "",
    days: int = 7,
    exit_mode: str = "opposite_signal",
    sl_pips: int = 50,
    tp_pips: int = 100,
    spread_pips: float = 3.0,
    ema_len: int = 20,
    rsi_len: int = 3,
    rsi_overbought: int = 80,
    rsi_oversold: int = 20,
    adx_len: int = 5,
    adx_smooth: int = 5,
    adx_level: int = 30,
) -> dict:
    """
    Backtest the ADX/RSI/EMA strategy on XAUUSD and return summary stats plus
    the full trade list. All indicator settings are adjustable per-call so you
    can experiment (e.g. "try ADX(14) instead of ADX(5)") without editing code.

    Args:
        capital: starting account balance in USD.
        lot_size: fixed lot size per trade (e.g. 0.01 = micro lot).
        start_date: "YYYY-MM-DD", or "" for "days ago from today".
        days: backtest window length in days.
        exit_mode: "opposite_signal" (hold until the opposite signal fires)
                   or "fixed_sltp" (exit at a fixed SL/TP in pips).
        sl_pips: stop loss distance in pips, only used when exit_mode="fixed_sltp".
        tp_pips: take profit distance in pips, only used when exit_mode="fixed_sltp".
        spread_pips: modelled round-trip spread cost in pips, deducted from every trade.
        ema_len: EMA period used for the trend filter (default 20).
        rsi_len: RSI period (default 3, the fast/reactive setting from the original setup).
        rsi_overbought: RSI level the SELL signal must cross back down through (default 80).
        rsi_oversold: RSI level the BUY signal must cross back up through (default 20).
        adx_len: ADX/DI smoothing period (default 5).
        adx_smooth: additional ADX smoothing period (default 5).
        adx_level: minimum ADX value required to confirm trend strength (default 30).
    """
    signals, date_from, date_to = _prepare_data(
        start_date, days, ema_len=ema_len, rsi_len=rsi_len, rsi_ob=rsi_overbought,
        rsi_os=rsi_oversold, adx_len=adx_len, adx_smooth=adx_smooth, adx_level=adx_level,
    )
    trades = run_backtest(signals, lot_size, exit_mode=exit_mode, sl_pips=sl_pips,
                           tp_pips=tp_pips, spread_pips=spread_pips)

    if trades.empty:
        return {
            "window": f"{date_from:%Y-%m-%d} to {date_to:%Y-%m-%d}",
            "trades": 0,
            "message": "No signals fired in this window.",
        }

    wins = trades[trades["pips"] > 0]
    total_pnl = trades["pnl_usd"].sum()
    equity = capital + trades["pnl_usd"].cumsum()
    max_dd_pct = ((equity - equity.cummax()) / equity.cummax() * 100).min()

    return {
        "window": f"{date_from:%Y-%m-%d} to {date_to:%Y-%m-%d}",
        "capital": capital,
        "lot_size": lot_size,
        "exit_mode": exit_mode,
        "total_trades": int(len(trades)),
        "buy_trades": int((trades["side"] == "BUY").sum()),
        "sell_trades": int((trades["side"] == "SELL").sum()),
        "win_rate_pct": round(len(wins) / len(trades) * 100, 1),
        "total_pips": round(float(trades["pips"].sum()), 1),
        "net_pnl_usd": round(float(total_pnl), 2),
        "return_pct": round(float(total_pnl / capital * 100), 2),
        "final_balance": round(float(capital + total_pnl), 2),
        "max_drawdown_pct": round(float(max_dd_pct), 2),
        "trade_log": trades.assign(
            entry_time=trades["entry_time"].astype(str),
            exit_time=trades["exit_time"].astype(str),
        ).round(3).where(lambda d: d.notna(), None).to_dict(orient="records"),
    }


@mcp.tool()
def get_backtest_chart(
    capital: float = 1000.0,
    lot_size: float = 0.01,
    start_date: str = "",
    days: int = 7,
    exit_mode: str = "opposite_signal",
    sl_pips: int = 50,
    tp_pips: int = 100,
    spread_pips: float = 3.0,
    ema_len: int = 20,
    rsi_len: int = 3,
    rsi_overbought: int = 80,
    rsi_oversold: int = 20,
    adx_len: int = 5,
    adx_smooth: int = 5,
    adx_level: int = 30,
    lookback_bars: int = 500,
) -> Image:
    """
    Run the same backtest as run_single_backtest but return a chart image
    instead of stats: price with EMA overlay, green/red BUY/SELL signal
    markers, and each trade's entry-to-exit path drawn as a dashed line
    (green = closed in profit, red = closed at a loss), plus RSI and ADX
    panels underneath. Use this when you want to SEE the signals and trades,
    not just the numbers - all backtest and indicator parameters are the
    same adjustable set as run_single_backtest.

    Args:
        capital, lot_size, start_date, days, exit_mode, sl_pips, tp_pips,
        spread_pips, ema_len, rsi_len, rsi_overbought, rsi_oversold, adx_len,
        adx_smooth, adx_level: same meaning as in run_single_backtest.
        lookback_bars: caps the chart to the most recent N bars so it stays
            readable on longer backtest windows (default 500). Pass a larger
            number, or 0 for "no cap", if you want the full window plotted.
    """
    signals, date_from, date_to = _prepare_data(
        start_date, days, ema_len=ema_len, rsi_len=rsi_len, rsi_ob=rsi_overbought,
        rsi_os=rsi_oversold, adx_len=adx_len, adx_smooth=adx_smooth, adx_level=adx_level,
    )
    trades = run_backtest(signals, lot_size, exit_mode=exit_mode, sl_pips=sl_pips,
                           tp_pips=tp_pips, spread_pips=spread_pips)

    title = (f"XAUUSD  {date_from:%Y-%m-%d} to {date_to:%Y-%m-%d}  |  "
             f"EMA({ema_len}) RSI({rsi_len}) ADX({adx_len}/{adx_smooth})  |  "
             f"{len(trades)} trades")
    png_bytes = generate_signal_chart(
        signals, trades=trades if not trades.empty else None, title=title,
        lookback_bars=None if lookback_bars == 0 else lookback_bars,
        rsi_ob=rsi_overbought, rsi_os=rsi_oversold, adx_level=adx_level,
    )
    return Image(data=png_bytes, format="png")


@mcp.tool()
def run_sl_tp_optimization(
    capital: float = 1000.0,
    lot_size: float = 0.01,
    start_date: str = "",
    days: int = 7,
    sl_values: str = "10,20,30,50,75,100,150,200",
    tp_values: str = "50,100,150,200",
) -> dict:
    """
    Grid-search stop-loss and take-profit levels (in pips) for the ADX/RSI/EMA
    strategy on XAUUSD, using actual bar highs/lows for realistic fills, and
    return every combination ranked by net P&L plus the single best result.

    Args:
        capital: starting account balance in USD.
        lot_size: fixed lot size per trade.
        start_date: "YYYY-MM-DD", or "" for "days ago from today".
        days: backtest window length in days.
        sl_values: comma-separated stop-loss pip values to test.
        tp_values: comma-separated take-profit pip values to test.
    """
    signals, date_from, date_to = _prepare_data(start_date, days)
    sl_list = [int(x.strip()) for x in sl_values.split(",") if x.strip()]
    tp_list = [int(x.strip()) for x in tp_values.split(",") if x.strip()]

    sweep = run_sl_tp_sweep(signals, lot_size, capital, sl_list, tp_list)

    if sweep.empty or sweep["trades"].sum() == 0:
        return {
            "window": f"{date_from:%Y-%m-%d} to {date_to:%Y-%m-%d}",
            "message": "No trades were generated for any SL/TP combination in this window.",
        }

    ranked = sweep.sort_values("net_pnl_usd", ascending=False).reset_index(drop=True)
    best = ranked.iloc[0]

    return {
        "window": f"{date_from:%Y-%m-%d} to {date_to:%Y-%m-%d}",
        "capital": capital,
        "lot_size": lot_size,
        "best_sl_pips": int(best.sl_pips),
        "best_tp_pips": int(best.tp_pips),
        "best_net_pnl_usd": round(float(best.net_pnl_usd), 2),
        "best_return_pct": round(float(best.return_pct), 2),
        "best_win_rate_pct": round(float(best.win_rate_pct), 1),
        "best_max_drawdown_pct": round(float(best.max_drawdown_pct), 2),
        "caveat": "Single-window result on a small sample - re-test across multiple "
                  "periods before treating this SL/TP as a real optimum.",
        "full_grid": ranked.round(3).to_dict(orient="records"),
    }


@mcp.tool()
def get_current_signal() -> dict:
    """
    Check the LIVE ADX(5)/RSI(3)/EMA(20) signal right now, using the most
    recent bars from MT5 (falls back to OANDA practice API, then delayed
    yfinance if MT5 isn't reachable). Returns whether the last confirmed
    closed bar produced a BUY, SELL, or no signal, plus current indicator
    values. Read-only - does not place any order.
    """
    raw, source = get_latest_bars(n_bars=200, verbose=False)
    signals = build_signals(raw)

    if len(signals) < 2:
        return {"message": "Not enough bars returned to evaluate a signal."}

    row = signals.iloc[-2]        # last CONFIRMED closed bar (last row may still be forming)
    row_time = signals.index[-2]
    forming = signals.iloc[-1]
    forming_time = signals.index[-1]

    side = "BUY" if row["buy_signal"] else ("SELL" if row["sell_signal"] else None)

    def _safe_round(val, n=3):
        try:
            return round(float(val), n)
        except (TypeError, ValueError):
            return None

    return {
        "data_source": source,
        "last_closed_bar_time": str(row_time),
        "signal": side or "NONE",
        "price": round(float(row["Close"]), 3),
        "ema": round(float(row["EMA"]), 3),
        "rsi": round(float(row["RSI"]), 2),
        "adx": round(float(row["ADX"]), 2),
        "trend": "UP" if row["trend_up"] else ("DOWN" if row["trend_down"] else "FLAT"),
        "volume": _safe_round(row.get("Volume"), 0),
        "obv": _safe_round(row.get("OBV"), 0),
        "cmf": _safe_round(row.get("CMF"), 3),
        "rvol": _safe_round(row.get("RVOL"), 2),
        "currently_forming_bar_time": str(forming_time),
        "currently_forming_price": round(float(forming["Close"]), 3),
        "note": "Signal is evaluated on the last CLOSED bar, matching how the Pine "
                "Script confirms signals on bar close, not on the still-forming bar. "
                "Volume/OBV/CMF/RVOL use MT5 tick_volume or OANDA tick-count proxies "
                "(true traded volume isn't available for OTC XAUUSD); null if the "
                "active data source doesn't report it.",
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")


# ============================================================================
# claude_desktop_config.json entry to add (merge into the existing "mcpServers"
# object alongside your mt5-mcp entry). Use the SAME full absolute python.exe
# path pattern you already had to use for mt5-mcp to avoid the ENOENT/PATH
# ambiguity between Windows Store Python and Anaconda:
#
# {
#   "mcpServers": {
#     "backtest-adx-rsi-ema": {
#       "command": "C:\\Users\\<you>\\Anaconda3\\python.exe",
#       "args": ["C:\\path\\to\\backtest_mcp_server.py"]
#     }
#   }
# }
# ============================================================================
