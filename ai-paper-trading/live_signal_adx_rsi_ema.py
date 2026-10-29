"""
Live signal monitor + PAPER (demo) trading for the ADX(5)/RSI(3)/EMA(20) setup.

Polls the latest bars on a loop, recomputes the indicator, and:
  - Alerts (console + optional sound beep) the moment a NEW confirmed
    BUY/SELL signal appears on a closed bar.
  - Optionally opens a SIMULATED position with configurable SL/TP (in pips),
    tracks it against real incoming price action, and closes it automatically
    when SL/TP is hit - exactly like the backtest's fixed_sltp mode, but live.
  - Shows running floating (unrealized) and realized gain/loss in $ and %,
    plus a running equity curve.

Data source (tried in this order, all free):
  1. MT5 terminal        - true real-time, your existing demo/live connection.
  2. OANDA practice API  - near-real-time REST polling, free practice account
                            you already use. Requires OANDA_API_KEY env var.
  3. yfinance (GC=F)     - no signup, but ~15-20 min delayed. Last resort only.

IMPORTANT: This is PAPER TRADING ONLY. It never places a real order - it just
simulates a position in memory/CSV and tracks what would have happened. Wire
your own MT5 order_send() call in `open_paper_position` / `close_paper_position`
later if/when you're ready to go from demo to live automation.

Also renders a LIVE MATPLOTLIB CHART (price + EMA overlay with BUY/SELL arrows,
plus RSI and ADX subplots) that refreshes every poll cycle. Disable with --no-chart
if you only want the console/CSV output.

Run: python live_signal_adx_rsi_ema.py
"""

import argparse
import sys
import time
from datetime import datetime

import pandas as pd

import matplotlib
matplotlib.use("TkAgg")  # interactive, works out of the box on Windows
import matplotlib.pyplot as plt

from backtest_adx_rsi_ema_indicator import (
    get_latest_bars,
    build_signals,
    PIP_SIZE,
    CONTRACT_SIZE,
)

# ==================== CONFIG ====================
POLL_INTERVAL_SECONDS = 15     # how often to re-check for a new closed bar / SL-TP hit
N_BARS = 200                   # warmup buffer for EMA(20)/RSI(3)/ADX(5) smoothing
SIGNAL_LOG_FILE = "live_signals_log.csv"
TRADE_LOG_FILE = "paper_trades_log.csv"
SOUND_ALERTS = True            # Windows only (winsound); auto-disabled elsewhere
SPREAD_PIPS = 3.0              # modelled round-trip cost, same convention as the backtest
SHOW_CHART = True              # live matplotlib chart on/off (can override with --no-chart)
CHART_LOOKBACK_BARS = 150      # how many recent bars to display on the chart


# ==================== USER INPUT ====================
def get_paper_trading_inputs():
    parser = argparse.ArgumentParser(description="Live ADX RSI EMA signal + paper trading")
    parser.add_argument("--capital", type=float, default=None, help="Starting demo capital in USD")
    parser.add_argument("--lot", type=float, default=None, help="Lot size per paper trade")
    parser.add_argument("--sl", type=int, default=None, help="Stop loss in pips")
    parser.add_argument("--tp", type=int, default=None, help="Take profit in pips")
    parser.add_argument("--no-trade", action="store_true", help="Signal-only mode, no paper trading")
    parser.add_argument("--no-chart", action="store_true", help="Disable the live matplotlib chart")
    args, _ = parser.parse_known_args()

    global SHOW_CHART
    if args.no_chart:
        SHOW_CHART = False

    if args.no_trade:
        return None, None, None, None, False

    capital = args.capital
    if capital is None:
        try:
            raw = input("Enter demo capital in USD (default 1000, blank = signal-only, no trading): ").strip()
            if raw == "":
                return None, None, None, None, False
            capital = float(raw)
        except (ValueError, EOFError):
            capital = 1000.0
            print(f"Invalid input - defaulting capital to ${capital:.2f}")

    lot = args.lot
    if lot is None:
        try:
            raw = input("Enter lot size per trade (default 0.01): ").strip()
            lot = float(raw) if raw else 0.01
        except (ValueError, EOFError):
            lot = 0.01

    sl_pips = args.sl
    if sl_pips is None:
        try:
            raw = input("Enter stop loss in pips (default 50): ").strip()
            sl_pips = int(raw) if raw else 50
        except (ValueError, EOFError):
            sl_pips = 50

    tp_pips = args.tp
    if tp_pips is None:
        try:
            raw = input("Enter take profit in pips (default 100): ").strip()
            tp_pips = int(raw) if raw else 100
        except (ValueError, EOFError):
            tp_pips = 100

    return capital, lot, sl_pips, tp_pips, True


# ==================== SOUND / LOGGING ====================
def beep(is_buy):
    if not SOUND_ALERTS:
        return
    try:
        import winsound
        if is_buy:
            winsound.Beep(1200, 150)
            winsound.Beep(1500, 150)
        else:
            winsound.Beep(600, 400)
    except Exception:
        pass  # non-Windows or no audio device - silently skip


def _append_csv(path, row_dict):
    import os
    df = pd.DataFrame([row_dict])
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def log_signal(row_time, side, price, ema_val, rsi_val, adx_val, source):
    _append_csv(SIGNAL_LOG_FILE, {
        "time": row_time, "side": side, "price": price,
        "EMA": round(ema_val, 3), "RSI": round(rsi_val, 2), "ADX": round(adx_val, 2),
        "source": source, "logged_at": datetime.now().isoformat(timespec="seconds"),
    })


def log_trade(trade):
    _append_csv(TRADE_LOG_FILE, trade)


# ==================== LIVE CHART ====================
class LiveChart:
    """Non-blocking matplotlib window: price+EMA with BUY/SELL arrows, RSI, ADX."""

    def __init__(self, lookback_bars=CHART_LOOKBACK_BARS):
        self.lookback_bars = lookback_bars
        plt.ion()
        self.fig, (self.ax_price, self.ax_rsi, self.ax_adx) = plt.subplots(
            3, 1, figsize=(12, 8), sharex=True,
            gridspec_kw={"height_ratios": [3, 1, 1]},
        )
        try:
            self.fig.canvas.manager.set_window_title(
                "XAUUSD Live Signal Chart - ADX(5)/RSI(3)/EMA(20)"
            )
        except Exception:
            pass
        plt.show(block=False)

    def update(self, signals, position=None, source=""):
        df = signals.tail(self.lookback_bars)

        for ax in (self.ax_price, self.ax_rsi, self.ax_adx):
            ax.clear()

        # ---- price + EMA ----
        self.ax_price.plot(df.index, df["Close"], color="#1f77b4", linewidth=1.1, label="Close")
        self.ax_price.plot(df.index, df["EMA"], color="#ff7f0e", linewidth=1.0, label="EMA(20)")

        buys = df[df["buy_signal"]]
        sells = df[df["sell_signal"]]
        if len(buys):
            self.ax_price.scatter(buys.index, buys["Close"], marker="^", color="lime",
                                   s=100, zorder=5, edgecolors="black", linewidths=0.6,
                                   label="BUY signal")
        if len(sells):
            self.ax_price.scatter(sells.index, sells["Close"], marker="v", color="red",
                                   s=100, zorder=5, edgecolors="black", linewidths=0.6,
                                   label="SELL signal")

        if position is not None:
            entry_color = "green" if position["side"] == "BUY" else "crimson"
            self.ax_price.axhline(position["entry_price"], color=entry_color,
                                   linestyle="--", linewidth=1, label=f"{position['side']} entry")
            self.ax_price.axhline(position["sl"], color="red", linestyle=":", linewidth=1, label="SL")
            self.ax_price.axhline(position["tp"], color="green", linestyle=":", linewidth=1, label="TP")

        self.ax_price.set_title(
            f"XAUUSD  |  source: {source}  |  last update {datetime.now():%H:%M:%S}"
        )
        self.ax_price.legend(loc="upper left", fontsize=8, ncol=2)
        self.ax_price.grid(alpha=0.3)

        # ---- RSI ----
        self.ax_rsi.plot(df.index, df["RSI"], color="#9467bd", linewidth=1)
        self.ax_rsi.axhline(70, color="gray", linestyle="--", linewidth=0.7)
        self.ax_rsi.axhline(30, color="gray", linestyle="--", linewidth=0.7)
        self.ax_rsi.set_ylabel("RSI(3)")
        self.ax_rsi.grid(alpha=0.3)

        # ---- ADX ----
        self.ax_adx.plot(df.index, df["ADX"], color="#2ca02c", linewidth=1)
        self.ax_adx.axhline(25, color="gray", linestyle="--", linewidth=0.7)
        self.ax_adx.set_ylabel("ADX(5)")
        self.ax_adx.grid(alpha=0.3)

        self.fig.autofmt_xdate()
        self.fig.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def pause(self, seconds):
        """Non-blocking wait that keeps the chart window responsive/redrawn."""
        plt.pause(seconds)

    def close(self):
        try:
            plt.close(self.fig)
        except Exception:
            pass


# ==================== PAPER TRADING ====================
def open_paper_position(side, price, time_, lot_size, sl_pips, tp_pips):
    if side == "BUY":
        sl = price - sl_pips * PIP_SIZE
        tp = price + tp_pips * PIP_SIZE
    else:
        sl = price + sl_pips * PIP_SIZE
        tp = price - tp_pips * PIP_SIZE
    return {
        "side": side, "entry_price": price, "entry_time": time_,
        "sl": sl, "tp": tp, "sl_pips": sl_pips, "tp_pips": tp_pips, "lot_size": lot_size,
    }


def check_paper_position(position, bar):
    """Check the forming/latest bar's High/Low against SL/TP. Returns (exit_now, exit_price, reason)."""
    if position["side"] == "BUY":
        if bar["Low"] <= position["sl"]:
            return True, position["sl"], "SL"
        if bar["High"] >= position["tp"]:
            return True, position["tp"], "TP"
    else:
        if bar["High"] >= position["sl"]:
            return True, position["sl"], "SL"
        if bar["Low"] <= position["tp"]:
            return True, position["tp"], "TP"
    return False, None, None


def calc_pips_pnl(position, exit_price, pip_value):
    raw_diff = (exit_price - position["entry_price"]) if position["side"] == "BUY" \
        else (position["entry_price"] - exit_price)
    pips = (raw_diff / PIP_SIZE) - SPREAD_PIPS
    return pips, pips * pip_value


def print_status(prefix, position, current_price, capital, realized_pnl):
    if position is None:
        equity = capital + realized_pnl
        sys.stdout.write(f"\r{prefix} no open position | equity=${equity:,.2f} "
                          f"(realized ${realized_pnl:+,.2f})    ")
    else:
        pip_value = CONTRACT_SIZE * PIP_SIZE * position["lot_size"]
        floating_pips, floating_pnl = calc_pips_pnl(position, current_price, pip_value)
        equity = capital + realized_pnl + floating_pnl
        sys.stdout.write(
            f"\r{prefix} {position['side']} open @ {position['entry_price']:.2f} | "
            f"price={current_price:.2f} | floating {floating_pips:+.1f}p (${floating_pnl:+,.2f}) | "
            f"SL={position['sl']:.2f} TP={position['tp']:.2f} | equity=${equity:,.2f}    "
        )
    sys.stdout.flush()


# ==================== MAIN LOOP ====================
def run_monitor():
    capital, lot_size, sl_pips, tp_pips, trading_enabled = get_paper_trading_inputs()

    print("=" * 70)
    print("LIVE SIGNAL MONITOR: ADX(5) / RSI(3) / EMA(20)")
    if trading_enabled:
        print(f"PAPER TRADING ON | capital=${capital:,.2f} lot={lot_size} "
              f"SL={sl_pips}p TP={tp_pips}p")
        print(f"Trade log -> {TRADE_LOG_FILE}")
    else:
        print("Signal-only mode (no paper trading)")
    print(f"Polling every {POLL_INTERVAL_SECONDS}s | signal log -> {SIGNAL_LOG_FILE}")
    if SHOW_CHART:
        print("Live chart -> ON (matplotlib window)")
    print("Ctrl+C to stop. READ-ONLY - no real orders are ever placed.")
    print("=" * 70)

    chart = LiveChart() if SHOW_CHART else None

    position = None
    realized_pnl = 0.0
    closed_trades = []
    last_alerted_bar_time = None

    def wait(seconds):
        """Sleep, but keep the chart window alive/responsive if it's open."""
        if chart is not None:
            chart.pause(seconds)
        else:
            time.sleep(seconds)

    while True:
        try:
            raw, source = get_latest_bars(N_BARS, verbose=False)
            signals = build_signals(raw)

            if len(signals) < 2:
                wait(POLL_INTERVAL_SECONDS)
                continue

            closed_row = signals.iloc[-2]
            closed_time = signals.index[-2]
            forming_row = signals.iloc[-1]

            # ---- 1. manage open paper position against the latest (forming) bar ----
            if trading_enabled and position is not None:
                exit_now, exit_price, reason = check_paper_position(position, forming_row)
                if exit_now:
                    pip_value = CONTRACT_SIZE * PIP_SIZE * position["lot_size"]
                    pips, pnl_usd = calc_pips_pnl(position, exit_price, pip_value)
                    realized_pnl += pnl_usd
                    trade_record = {
                        "side": position["side"], "entry_time": position["entry_time"],
                        "exit_time": signals.index[-1], "entry_price": position["entry_price"],
                        "exit_price": exit_price, "pips": round(pips, 1),
                        "pnl_usd": round(pnl_usd, 2), "reason": reason,
                        "equity_after": round(capital + realized_pnl, 2),
                    }
                    closed_trades.append(trade_record)
                    log_trade(trade_record)
                    print(f"\n[{datetime.now():%H:%M:%S}] CLOSED {position['side']} "
                          f"({reason})  {pips:+.1f}p  ${pnl_usd:+,.2f}  "
                          f"equity=${capital + realized_pnl:,.2f}")
                    position = None

            # ---- 2. check for a new confirmed signal on the closed bar ----
            new_bar = closed_time != last_alerted_bar_time
            side = "BUY" if closed_row["buy_signal"] else ("SELL" if closed_row["sell_signal"] else None)

            if side and new_bar:
                last_alerted_bar_time = closed_time
                ts = datetime.now().strftime("%H:%M:%S")
                arrow = "\033[92m▲ BUY \033[0m" if side == "BUY" else "\033[91m▼ SELL\033[0m"
                print(f"\n[{ts}] {arrow}  bar={closed_time}  price={closed_row['Close']:.2f}  "
                      f"EMA={closed_row['EMA']:.2f}  RSI={closed_row['RSI']:.1f}  "
                      f"ADX={closed_row['ADX']:.1f}  (source: {source})")
                beep(side == "BUY")
                log_signal(closed_time, side, closed_row["Close"], closed_row["EMA"],
                           closed_row["RSI"], closed_row["ADX"], source)

                if trading_enabled and position is None:
                    position = open_paper_position(side, closed_row["Close"], closed_time,
                                                     lot_size, sl_pips, tp_pips)
                    print(f"           -> opened PAPER {side} @ {position['entry_price']:.2f}  "
                          f"SL={position['sl']:.2f}  TP={position['tp']:.2f}")

            # ---- 3. update chart ----
            if chart is not None:
                chart.update(signals, position=position, source=source)

            # ---- 4. status line ----
            ts = datetime.now().strftime("%H:%M:%S")
            print_status(f"[{ts}]", position, forming_row["Close"], capital or 0.0, realized_pnl)

        except KeyboardInterrupt:
            print("\nStopped by user.")
            break
        except Exception as e:
            print(f"\n[warn] poll failed: {e} - retrying in {POLL_INTERVAL_SECONDS}s")

        wait(POLL_INTERVAL_SECONDS)

    if chart is not None:
        chart.close()

    # ---- final summary ----
    if trading_enabled:
        print("\n" + "=" * 50)
        print("SESSION SUMMARY")
        print(f"Starting capital : ${capital:,.2f}")
        print(f"Closed trades    : {len(closed_trades)}")
        if closed_trades:
            wins = [t for t in closed_trades if t["pnl_usd"] > 0]
            print(f"Win rate         : {len(wins)/len(closed_trades)*100:.1f}%")
            print(f"Realized P&L     : ${realized_pnl:,.2f} "
                  f"({realized_pnl/capital*100:+.2f}%)")
            print(f"Final equity     : ${capital + realized_pnl:,.2f}")
        if position is not None:
            print(f"Open position    : {position['side']} @ {position['entry_price']:.2f} "
                  f"(still open when stopped - not included in realized P&L)")
        print(f"Full trade log   -> {TRADE_LOG_FILE}")
        print("=" * 50)


if __name__ == "__main__":
    run_monitor()
