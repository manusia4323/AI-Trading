"""
break_retest_service.py

Standalone FastAPI microservice implementing a price-action "Break & Retest"
signal generator for XAUUSD, independent from the existing EMA/RSI/ADX
strategy layer. It exposes its own router and can either be mounted into the
main app (`app.include_router(break_retest_router)`) or run on its own port.

Three setups, based on the break-and-retest methodology:
  1. Previous Day High / Low break & retest
  2. Opening Range retest (NY session open, 09:30 America/New_York)
  3. Order Block retest (last opposite-colored candle before an impulse move)

Design choices:
  - STATELESS core: POST candle data in, get signals out via /analyze. This
    keeps the detection logic decoupled from any specific provider.
  - LIVE MODE (new): a background task polls a live data source on an
    interval, and whenever a new bar closes it recomputes signals and
    (a) caches them for polling via GET /signals/live, and
    (b) pushes them to any connected clients via WebSocket /ws/signals.
    Two sources are supported, selectable via POST /live/start?source=...:
      - "yfinance" (default): works anywhere, no extra setup. Pulls XAUUSD=X.
      - "mt5": connects directly to a running MT5 terminal via the
        MetaTrader5 pip package. WINDOWS ONLY, and only where this process
        can reach your logged-in MT5 terminal (same machine, matching your
        existing mt5-mcp / mt5_signal_server.py setup). Falls back with a
        clear error if selected somewhere the package isn't importable.
  - The original OPTIONAL in-memory push-based store (/candles/daily,
    /candles/intraday, /signals) is kept as-is for anyone feeding candles in
    manually from another provider instead of using either live poller.

Run standalone (yfinance, cross-platform):
    pip install fastapi uvicorn yfinance --break-system-packages
    uvicorn break_retest_service:app --port 8100 --reload

Run with MT5 as the live source (on your Windows machine, MT5 terminal open and logged in):
    pip install MetaTrader5
    # then either flip DEFAULT_LIVE_SOURCE = LiveSource.MT5 below, or leave the
    # default and call POST /break-retest/live/start?source=mt5 once it's running

Mount into an existing FastAPI app instead:
    from break_retest_service import router as break_retest_router, start_live_poller, stop_live_poller
    app.include_router(break_retest_router)
    # then call start_live_poller()/stop_live_poller() from your own lifespan handler,
    # or just import this module's `app` directly if you don't need to combine apps.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime, date, time as dtime, timezone
from enum import Enum
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf
from fastapi import APIRouter, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field, field_validator

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    mt5 = None
    MT5_AVAILABLE = False  # expected on Linux/Mac -- MetaTrader5 is a Windows-only package
    # that talks to a locally running MT5 terminal over IPC. This is normal outside
    # your Windows machine; the /live/start endpoint will return a clear error if
    # someone tries to select source="mt5" in an environment where it's unavailable.

logger = logging.getLogger("break_retest_service")
logging.basicConfig(level=logging.INFO)

# --------------------------------------------------------------------------
# Config -- mirrors constants already in use elsewhere in the project so the
# numbers are directly comparable across services.
# --------------------------------------------------------------------------
PIP_SIZE = 0.1              # XAUUSD: 1 pip = $0.10
SPREAD_PIPS = 3.0
STOP_LOSS_PIPS = 50         # default fallback stop distance if a setup has
                             # no natural structural stop
RISK_REWARD_RATIO = 2.0
INITIAL_BALANCE = 10000

NY_TZ = ZoneInfo("America/New_York")
NY_OPEN = dtime(9, 30)
OPENING_RANGE_MINUTES = 5   # first N minutes after NY open define the range

# How close price needs to come back to a level to count as a "retest",
# expressed in pips. Loosened/tightened per setup below.
RETEST_TOLERANCE_PIPS = 15.0
ORDER_BLOCK_TOLERANCE_PIPS = 8.0

# --------------------------------------------------------------------------
# Live feed config (Yahoo Finance polling)
# --------------------------------------------------------------------------
YF_SYMBOL = "GC=F"        # Yahoo's spot gold ticker. "GC=F" (COMEX futures)
                               # is a reasonable alternate if this one gives
                               # sparse/delayed data for your account region.
YF_INTRADAY_INTERVAL = "5m"   # must match your canonical "5m" timeframe
YF_INTRADAY_PERIOD = "5d"     # yfinance caps 5m history at ~60 days; 5d is
                               # plenty for the 3-bar break/retest/reaction window
YF_DAILY_PERIOD = "5d"        # only need the last 2 completed daily bars
POLL_INTERVAL_SECONDS = 30    # how often to check Yahoo for a new bar
MAX_EMITTED_SIGNAL_KEYS = 500 # cap on the dedupe set so it can't grow forever


class LiveSource(str, Enum):
    YFINANCE = "yfinance"
    MT5 = "mt5"


# Which source the standalone app's lifespan auto-starts with. yfinance works
# anywhere (cross-platform, no extra setup); mt5 requires this process to run
# on the same Windows machine as a logged-in MT5 terminal, with the
# MetaTrader5 pip package installed. Change to LiveSource.MT5 once you've
# verified it works on your machine (see /live/start to switch at runtime
# without restarting the service).
DEFAULT_LIVE_SOURCE = LiveSource.YFINANCE

MT5_SYMBOL = "XAUUSD"           # verify this matches Finex's exact symbol name in
                                  # MT5 Market Watch -- some brokers suffix it
                                  # (e.g. "XAUUSD.m", "XAUUSDm", "XAUUSD.raw")
MT5_INTRADAY_TIMEFRAME_ATTR = "TIMEFRAME_M5"  # resolved against the mt5 module at call time
MT5_INTRADAY_BARS = 200          # enough for the 3-bar pattern + order-block lookback
MT5_DAILY_BARS = 5
MT5_TERMINAL_PATH: Optional[str] = None  # e.g. r"C:\Program Files\Finex MT5 Terminal\terminal64.exe"
                                          # only needed if mt5.initialize() can't auto-locate your terminal


class Side(str, Enum):
    LONG = "long"
    SHORT = "short"


class SetupType(str, Enum):
    PREV_DAY_HL_RETEST = "previous_day_high_low_retest"
    OPENING_RANGE_RETEST = "opening_range_retest"
    ORDER_BLOCK_RETEST = "order_block_retest"


# --------------------------------------------------------------------------
# Data models
# --------------------------------------------------------------------------
class Candle(BaseModel):
    timestamp: datetime = Field(..., description="UTC or tz-aware timestamp of candle open")
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float] = None

    @field_validator("timestamp")
    @classmethod
    def ensure_tz_aware(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v

    @property
    def is_up_close(self) -> bool:
        return self.close > self.open

    @property
    def is_down_close(self) -> bool:
        return self.close < self.open

    @property
    def body_low(self) -> float:
        return min(self.open, self.close)

    @property
    def body_high(self) -> float:
        return max(self.open, self.close)


class AnalyzeRequest(BaseModel):
    symbol: str = "XAUUSD"
    timeframe: str = "5m"  # canonical internal format, per project convention
    daily_candles: list[Candle] = Field(
        default_factory=list,
        description="Recent daily candles (needs at least 2: previous day + current day in progress)",
    )
    intraday_candles: list[Candle] = Field(
        default_factory=list,
        description="Recent intraday candles at `timeframe` resolution, chronological order",
    )


class Signal(BaseModel):
    setup: SetupType
    side: Side
    symbol: str
    timeframe: str
    level: float
    entry: float
    stop_loss: float
    take_profit: float
    stop_pips: float
    reward_pips: float
    reason: str
    triggered_at: datetime


class AnalyzeResponse(BaseModel):
    symbol: str
    timeframe: str
    signals: list[Signal]
    levels: dict[str, Optional[float]]


# --------------------------------------------------------------------------
# In-memory store (optional incremental-feed mode)
# --------------------------------------------------------------------------
class _CandleStore:
    """Very simple per-symbol candle buffer. Not persisted -- restart clears it.
    Swap for Redis/SQLite if you need durability across restarts."""

    def __init__(self, max_daily: int = 10, max_intraday: int = 500):
        self.max_daily = max_daily
        self.max_intraday = max_intraday
        self.daily: dict[str, list[Candle]] = {}
        self.intraday: dict[str, list[Candle]] = {}

    def push_daily(self, symbol: str, candles: list[Candle]) -> None:
        buf = self.daily.setdefault(symbol, [])
        buf.extend(candles)
        buf.sort(key=lambda c: c.timestamp)
        self.daily[symbol] = buf[-self.max_daily :]

    def push_intraday(self, symbol: str, candles: list[Candle]) -> None:
        buf = self.intraday.setdefault(symbol, [])
        buf.extend(candles)
        buf.sort(key=lambda c: c.timestamp)
        self.intraday[symbol] = buf[-self.max_intraday :]

    def get(self, symbol: str) -> tuple[list[Candle], list[Candle]]:
        return self.daily.get(symbol, []), self.intraday.get(symbol, [])


store = _CandleStore()


# --------------------------------------------------------------------------
# Core detection logic
# --------------------------------------------------------------------------
def _pips(a: float, b: float) -> float:
    return abs(a - b) / PIP_SIZE


def _rr_take_profit(entry: float, stop: float, side: Side) -> float:
    risk = abs(entry - stop)
    reward = risk * RISK_REWARD_RATIO
    return entry + reward if side == Side.LONG else entry - reward


def detect_previous_day_hl_retest(
    symbol: str, timeframe: str, daily: list[Candle], intraday: list[Candle]
) -> list[Signal]:
    """Setup 1: break of prior day's high/low, then retest for continuation."""
    if len(daily) < 2 or len(intraday) < 3:
        return []

    prev_day = daily[-2]
    prev_high, prev_low = prev_day.high, prev_day.low
    signals: list[Signal] = []

    # Look at the most recent 3 candles: break candle -> retest candle -> reaction candle
    c_break, c_retest, c_reaction = intraday[-3], intraday[-2], intraday[-1]

    # --- Long: broke above prev_high, retested it, held ---
    if c_break.high > prev_high and c_retest.low <= prev_high + RETEST_TOLERANCE_PIPS * PIP_SIZE:
        if c_reaction.close > prev_high and c_reaction.is_up_close:
            stop = min(c_retest.low, prev_high) - STOP_LOSS_PIPS * PIP_SIZE * 0.2
            entry = c_reaction.close
            tp = _rr_take_profit(entry, stop, Side.LONG)
            signals.append(
                Signal(
                    setup=SetupType.PREV_DAY_HL_RETEST,
                    side=Side.LONG,
                    symbol=symbol,
                    timeframe=timeframe,
                    level=prev_high,
                    entry=entry,
                    stop_loss=stop,
                    take_profit=tp,
                    stop_pips=_pips(entry, stop),
                    reward_pips=_pips(entry, tp),
                    reason="Break above previous day high, retested as support, bullish reaction candle",
                    triggered_at=c_reaction.timestamp,
                )
            )

    # --- Short: broke below prev_low, retested it, held ---
    if c_break.low < prev_low and c_retest.high >= prev_low - RETEST_TOLERANCE_PIPS * PIP_SIZE:
        if c_reaction.close < prev_low and c_reaction.is_down_close:
            stop = max(c_retest.high, prev_low) + STOP_LOSS_PIPS * PIP_SIZE * 0.2
            entry = c_reaction.close
            tp = _rr_take_profit(entry, stop, Side.SHORT)
            signals.append(
                Signal(
                    setup=SetupType.PREV_DAY_HL_RETEST,
                    side=Side.SHORT,
                    symbol=symbol,
                    timeframe=timeframe,
                    level=prev_low,
                    entry=entry,
                    stop_loss=stop,
                    take_profit=tp,
                    stop_pips=_pips(entry, stop),
                    reward_pips=_pips(entry, tp),
                    reason="Break below previous day low, retested as resistance, bearish reaction candle",
                    triggered_at=c_reaction.timestamp,
                )
            )

    return signals


def _ny_session_date(ts: datetime) -> date:
    return ts.astimezone(NY_TZ).date()


def compute_opening_range(intraday: list[Candle]) -> Optional[tuple[float, float, date]]:
    """Finds today's (most recent NY session's) opening range high/low from
    the first OPENING_RANGE_MINUTES of candles after 09:30 America/New_York.
    Works with 1m candles feeding a 5m range, or a single 5m candle directly.
    """
    if not intraday:
        return None

    latest_session_date = _ny_session_date(intraday[-1].timestamp)
    session_open_dt = datetime.combine(latest_session_date, NY_OPEN, tzinfo=NY_TZ)
    session_close_of_range = session_open_dt.replace(
        minute=session_open_dt.minute + OPENING_RANGE_MINUTES
    ) if OPENING_RANGE_MINUTES < 60 - session_open_dt.minute else session_open_dt

    range_candles = [
        c
        for c in intraday
        if _ny_session_date(c.timestamp) == latest_session_date
        and session_open_dt <= c.timestamp.astimezone(NY_TZ) < session_open_dt.replace(
            minute=session_open_dt.minute
        ) + (session_close_of_range - session_open_dt if session_close_of_range != session_open_dt else __import__("datetime").timedelta(minutes=OPENING_RANGE_MINUTES))
    ]

    if not range_candles:
        return None

    high = max(c.high for c in range_candles)
    low = min(c.low for c in range_candles)
    return high, low, latest_session_date


def detect_opening_range_retest(
    symbol: str, timeframe: str, intraday: list[Candle]
) -> list[Signal]:
    """Setup 2: break of NY-open 5-minute range, then retest for continuation."""
    rng = compute_opening_range(intraday)
    if rng is None or len(intraday) < 3:
        return []
    range_high, range_low, session_date = rng

    # Only consider candles that occur after the opening range itself
    post_range = [
        c
        for c in intraday
        if _ny_session_date(c.timestamp) == session_date
        and c.timestamp.astimezone(NY_TZ).time() >= dtime(9, 30 + OPENING_RANGE_MINUTES)
    ]
    if len(post_range) < 3:
        return []

    c_break, c_retest, c_reaction = post_range[-3], post_range[-2], post_range[-1]
    signals: list[Signal] = []

    if c_break.high > range_high and c_retest.low <= range_high + RETEST_TOLERANCE_PIPS * PIP_SIZE:
        if c_reaction.close > range_high and c_reaction.is_up_close:
            stop = range_high - STOP_LOSS_PIPS * PIP_SIZE * 0.2
            entry = c_reaction.close
            tp = _rr_take_profit(entry, stop, Side.LONG)
            signals.append(
                Signal(
                    setup=SetupType.OPENING_RANGE_RETEST,
                    side=Side.LONG,
                    symbol=symbol,
                    timeframe=timeframe,
                    level=range_high,
                    entry=entry,
                    stop_loss=stop,
                    take_profit=tp,
                    stop_pips=_pips(entry, stop),
                    reward_pips=_pips(entry, tp),
                    reason="Break above NY opening range high, retested, bullish reaction",
                    triggered_at=c_reaction.timestamp,
                )
            )

    if c_break.low < range_low and c_retest.high >= range_low - RETEST_TOLERANCE_PIPS * PIP_SIZE:
        if c_reaction.close < range_low and c_reaction.is_down_close:
            stop = range_low + STOP_LOSS_PIPS * PIP_SIZE * 0.2
            entry = c_reaction.close
            tp = _rr_take_profit(entry, stop, Side.SHORT)
            signals.append(
                Signal(
                    setup=SetupType.OPENING_RANGE_RETEST,
                    side=Side.SHORT,
                    symbol=symbol,
                    timeframe=timeframe,
                    level=range_low,
                    entry=entry,
                    stop_loss=stop,
                    take_profit=tp,
                    stop_pips=_pips(entry, stop),
                    reward_pips=_pips(entry, tp),
                    reason="Break below NY opening range low, retested, bearish reaction",
                    triggered_at=c_reaction.timestamp,
                )
            )

    return signals


def _find_order_block(intraday: list[Candle], direction: Side, lookback: int = 20) -> Optional[Candle]:
    """Order block = last opposite-colored candle before an impulse move.
    For a SHORT setup (downtrend), we want the last up-close candle before
    price broke down hard. For a LONG setup, the last down-close candle
    before price broke up hard."""
    if len(intraday) < 4:
        return None

    window = intraday[-lookback:]
    target_pred = (lambda c: c.is_up_close) if direction == Side.SHORT else (lambda c: c.is_down_close)

    for c in reversed(window[:-1]):  # exclude the very latest candle
        if target_pred(c):
            return c
    return None


def detect_order_block_retest(
    symbol: str, timeframe: str, intraday: list[Candle]
) -> list[Signal]:
    """Setup 3: retest of the last opposite-colored candle before an impulse
    move, with a weak reaction candle confirming rejection."""
    if len(intraday) < 5:
        return []

    signals: list[Signal] = []
    latest = intraday[-1]
    prior = intraday[-2]

    # --- Bearish order block (downtrend continuation) ---
    ob_short = _find_order_block(intraday, Side.SHORT)
    if ob_short is not None:
        ob_zone_top = ob_short.body_low  # "bottom of the wick to bottom of the body" per the transcript
        ob_zone_bottom = ob_short.low
        touched = ob_zone_bottom - ORDER_BLOCK_TOLERANCE_PIPS * PIP_SIZE <= latest.high <= ob_zone_top + ORDER_BLOCK_TOLERANCE_PIPS * PIP_SIZE
        weak_reaction = latest.is_down_close and latest.close < prior.close
        if touched and weak_reaction:
            stop = ob_zone_top + STOP_LOSS_PIPS * PIP_SIZE * 0.2
            entry = latest.close
            tp = _rr_take_profit(entry, stop, Side.SHORT)
            signals.append(
                Signal(
                    setup=SetupType.ORDER_BLOCK_RETEST,
                    side=Side.SHORT,
                    symbol=symbol,
                    timeframe=timeframe,
                    level=ob_zone_top,
                    entry=entry,
                    stop_loss=stop,
                    take_profit=tp,
                    stop_pips=_pips(entry, stop),
                    reward_pips=_pips(entry, tp),
                    reason="Retest of bearish order block (last up-close candle before impulse down), weak reaction",
                    triggered_at=latest.timestamp,
                )
            )

    # --- Bullish order block (uptrend continuation) ---
    ob_long = _find_order_block(intraday, Side.LONG)
    if ob_long is not None:
        ob_zone_bottom = ob_long.body_high
        ob_zone_top = ob_long.high
        touched = ob_zone_bottom - ORDER_BLOCK_TOLERANCE_PIPS * PIP_SIZE <= latest.low <= ob_zone_top + ORDER_BLOCK_TOLERANCE_PIPS * PIP_SIZE
        strong_reaction = latest.is_up_close and latest.close > prior.close
        if touched and strong_reaction:
            stop = ob_zone_bottom - STOP_LOSS_PIPS * PIP_SIZE * 0.2
            entry = latest.close
            tp = _rr_take_profit(entry, stop, Side.LONG)
            signals.append(
                Signal(
                    setup=SetupType.ORDER_BLOCK_RETEST,
                    side=Side.LONG,
                    symbol=symbol,
                    timeframe=timeframe,
                    level=ob_zone_bottom,
                    entry=entry,
                    stop_loss=stop,
                    take_profit=tp,
                    stop_pips=_pips(entry, stop),
                    reward_pips=_pips(entry, tp),
                    reason="Retest of bullish order block (last down-close candle before impulse up), strong reaction",
                    triggered_at=latest.timestamp,
                )
            )

    return signals


def run_all_setups(
    symbol: str, timeframe: str, daily: list[Candle], intraday: list[Candle]
) -> AnalyzeResponse:
    signals: list[Signal] = []
    signals += detect_previous_day_hl_retest(symbol, timeframe, daily, intraday)
    signals += detect_opening_range_retest(symbol, timeframe, intraday)
    signals += detect_order_block_retest(symbol, timeframe, intraday)

    levels: dict[str, Optional[float]] = {
        "previous_day_high": daily[-2].high if len(daily) >= 2 else None,
        "previous_day_low": daily[-2].low if len(daily) >= 2 else None,
    }
    rng = compute_opening_range(intraday)
    levels["opening_range_high"] = rng[0] if rng else None
    levels["opening_range_low"] = rng[1] if rng else None

    return AnalyzeResponse(symbol=symbol, timeframe=timeframe, signals=signals, levels=levels)


# --------------------------------------------------------------------------
# Live feed: Yahoo Finance polling
# --------------------------------------------------------------------------
def _yf_history_to_candles(hist: "pd.DataFrame") -> list[Candle]:
    """Convert a yfinance history DataFrame (DatetimeIndex + OHLCV columns)
    into our Candle model list, in chronological order."""
    candles: list[Candle] = []
    if hist is None or hist.empty:
        return candles
    for ts, row in hist.iterrows():
        # yfinance sometimes returns tz-naive intraday indices depending on
        # environment; treat naive timestamps as UTC to stay consistent with
        # the Candle validator's own naive->UTC assumption.
        ts_py = ts.to_pydatetime()
        candles.append(
            Candle(
                timestamp=ts_py,
                open=float(row["Open"]),
                high=float(row["High"]),
                low=float(row["Low"]),
                close=float(row["Close"]),
                volume=float(row["Volume"]) if "Volume" in row and pd.notna(row["Volume"]) else None,
            )
        )
    return candles


def _fetch_yf_candles_sync(symbol: str) -> tuple[list[Candle], list[Candle]]:
    """Blocking call -- run via asyncio.to_thread from async code. yfinance
    uses `requests` under the hood, so it isn't awaitable natively."""
    ticker = yf.Ticker(symbol)

    intraday_hist = ticker.history(period=YF_INTRADAY_PERIOD, interval=YF_INTRADAY_INTERVAL)
    daily_hist = ticker.history(period=YF_DAILY_PERIOD, interval="1d")

    return _yf_history_to_candles(daily_hist), _yf_history_to_candles(intraday_hist)


async def fetch_yf_candles(symbol: str = YF_SYMBOL) -> tuple[list[Candle], list[Candle]]:
    return await asyncio.to_thread(_fetch_yf_candles_sync, symbol)


def _mt5_rates_to_candles(rates) -> list[Candle]:
    """Convert an MT5 copy_rates_from_pos() numpy structured array into our
    Candle model list. MT5 already returns bars in ascending chronological
    order. NOTE: `time` is the broker SERVER time (Finex's server clock),
    not necessarily UTC -- same caveat as the opening-range logic in
    BreakRetestEA.mq5. If your broker's server time has a fixed UTC offset,
    account for it before relying on the opening-range setup specifically;
    the previous-day and order-block setups are unaffected since they're
    relative, not tied to a specific wall-clock session time."""
    candles: list[Candle] = []
    if rates is None:
        return candles
    field_names = rates.dtype.names or ()
    for r in rates:
        candles.append(
            Candle(
                timestamp=datetime.fromtimestamp(int(r["time"]), tz=timezone.utc),
                open=float(r["open"]),
                high=float(r["high"]),
                low=float(r["low"]),
                close=float(r["close"]),
                volume=float(r["tick_volume"]) if "tick_volume" in field_names else None,
            )
        )
    return candles


def _mt5_ensure_connected() -> None:
    if not MT5_AVAILABLE:
        raise RuntimeError(
            "MetaTrader5 package is not installed/importable in this environment. "
            "It only works on Windows, in the same process tree as (or able to reach) "
            "a running, logged-in MT5 terminal. Install with `pip install MetaTrader5` "
            "on that machine."
        )
    if mt5.terminal_info() is None:
        ok = mt5.initialize(path=MT5_TERMINAL_PATH) if MT5_TERMINAL_PATH else mt5.initialize()
        if not ok:
            raise RuntimeError(f"MT5 initialize() failed: {mt5.last_error()}")


def _fetch_mt5_candles_sync(symbol: str) -> tuple[list[Candle], list[Candle]]:
    """Blocking call -- run via asyncio.to_thread from async code. The
    MetaTrader5 package is a synchronous IPC client, not awaitable natively."""
    _mt5_ensure_connected()

    if not mt5.symbol_select(symbol, True):
        raise RuntimeError(
            f"MT5 symbol_select('{symbol}') failed: {mt5.last_error()}. "
            f"Check the exact symbol name in your MT5 Market Watch -- Finex may suffix it "
            f"(e.g. 'XAUUSD.m'), in which case update MT5_SYMBOL."
        )

    timeframe = getattr(mt5, MT5_INTRADAY_TIMEFRAME_ATTR)
    daily_rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_D1, 0, MT5_DAILY_BARS)
    intraday_rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, MT5_INTRADAY_BARS)

    if daily_rates is None or intraday_rates is None:
        raise RuntimeError(f"MT5 copy_rates_from_pos returned None: {mt5.last_error()}")

    return _mt5_rates_to_candles(daily_rates), _mt5_rates_to_candles(intraday_rates)


async def fetch_mt5_candles(symbol: str = MT5_SYMBOL) -> tuple[list[Candle], list[Candle]]:
    return await asyncio.to_thread(_fetch_mt5_candles_sync, symbol)


async def fetch_live_candles(source: LiveSource, symbol: str) -> tuple[list[Candle], list[Candle]]:
    """Single dispatch point so _poll_once() doesn't need to know which
    provider is active."""
    if source == LiveSource.MT5:
        return await fetch_mt5_candles(symbol)
    return await fetch_yf_candles(symbol)


class _ConnectionManager:
    """Tracks connected WebSocket clients and broadcasts new signals to all
    of them. Deliberately simple -- no auth, no rooms; add your own if this
    service is ever exposed beyond localhost/your own network."""

    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)
        logger.info("WebSocket client connected (%d active)", len(self.active))

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)
        logger.info("WebSocket client disconnected (%d active)", len(self.active))

    async def broadcast(self, payload: dict):
        dead: list[WebSocket] = []
        for ws in self.active:
            try:
                await ws.send_json(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


ws_manager = _ConnectionManager()

# Cache of the most recent /analyze-equivalent result from the live poller,
# plus a rolling list of newly-emitted signals and a dedupe set so the same
# signal (same setup+side+bar) doesn't get rebroadcast every poll interval.
_live_lock = asyncio.Lock()
_live_state: dict = {
    "latest": None,          # AnalyzeResponse | None
    "recent_signals": [],    # list[Signal], most recent first, capped
    "last_bar_time": None,   # datetime | None -- last intraday bar we evaluated
    "last_poll_at": None,    # datetime | None
    "last_error": None,      # str | None
    "symbol": YF_SYMBOL,
    "source": DEFAULT_LIVE_SOURCE,
}
_emitted_keys: set[str] = set()
_poller_task: Optional[asyncio.Task] = None


def _signal_key(sig: Signal) -> str:
    return f"{sig.setup}:{sig.side}:{sig.triggered_at.isoformat()}"


async def _poll_once() -> list[Signal]:
    """Fetch latest candles, recompute signals, update cache, return any
    NEWLY triggered signals (i.e. not already emitted)."""
    source = _live_state["source"]
    symbol = _live_state["symbol"]
    daily, intraday = await fetch_live_candles(source, symbol)
    if not intraday:
        raise RuntimeError(
            f"{source.value} returned no intraday candles (market closed, bad symbol, "
            f"MT5 terminal not connected, or rate-limited)"
        )

    report_symbol = symbol.replace("=X", "").replace("=F", "") if source == LiveSource.YFINANCE else symbol
    result = run_all_setups(report_symbol, YF_INTRADAY_INTERVAL, daily, intraday)

    new_signals: list[Signal] = []
    for sig in result.signals:
        key = _signal_key(sig)
        if key not in _emitted_keys:
            _emitted_keys.add(key)
            new_signals.append(sig)

    if len(_emitted_keys) > MAX_EMITTED_SIGNAL_KEYS:
        # drop the oldest half -- cheap way to bound memory without needing
        # an ordered structure for what's a purely advisory dedupe set
        _emitted_keys.clear()
        _emitted_keys.update(_signal_key(s) for s in result.signals)

    async with _live_lock:
        _live_state["latest"] = result
        _live_state["last_bar_time"] = intraday[-1].timestamp
        _live_state["last_poll_at"] = datetime.now(timezone.utc)
        _live_state["last_error"] = None
        if new_signals:
            _live_state["recent_signals"] = (new_signals + _live_state["recent_signals"])[:50]

    return new_signals


async def _poll_loop():
    logger.info("Live poller starting: source=%s symbol=%s interval=%s poll_every=%ss",
                _live_state["source"].value, _live_state["symbol"], YF_INTRADAY_INTERVAL, POLL_INTERVAL_SECONDS)
    while True:
        try:
            new_signals = await _poll_once()
            for sig in new_signals:
                logger.info("New signal: %s %s @ %.2f (%s)", sig.side, sig.setup, sig.entry, sig.reason)
                await ws_manager.broadcast({"type": "signal", "data": sig.model_dump(mode="json")})
        except Exception as exc:  # noqa: BLE001 -- keep the poller alive across transient failures
            logger.warning("Live poller error: %s", exc)
            async with _live_lock:
                _live_state["last_error"] = str(exc)
            await ws_manager.broadcast({"type": "error", "message": str(exc)})
        await asyncio.sleep(POLL_INTERVAL_SECONDS)


def start_live_poller(source: LiveSource = DEFAULT_LIVE_SOURCE, symbol: Optional[str] = None):
    """Call this from your own app's startup/lifespan handler if you mounted
    `router` into an existing FastAPI app instead of running this file's
    standalone `app`. If switching sources, this restarts the poll loop
    (any in-flight poll from the old source just finishes and is discarded)."""
    global _poller_task
    if symbol is None:
        symbol = MT5_SYMBOL if source == LiveSource.MT5 else YF_SYMBOL
    if source == LiveSource.MT5 and not MT5_AVAILABLE:
        raise RuntimeError(
            "Cannot start live poller with source='mt5': the MetaTrader5 package isn't "
            "available in this environment. Run this service on the Windows machine with "
            "your MT5 terminal, with `pip install MetaTrader5` done there."
        )
    _live_state["source"] = source
    _live_state["symbol"] = symbol
    _live_state["last_error"] = None
    if _poller_task is not None and not _poller_task.done():
        _poller_task.cancel()
    _poller_task = asyncio.create_task(_poll_loop())


def stop_live_poller():
    global _poller_task
    if _poller_task is not None:
        _poller_task.cancel()
        _poller_task = None
    if MT5_AVAILABLE and mt5.terminal_info() is not None:
        mt5.shutdown()


# --------------------------------------------------------------------------
# FastAPI router
# --------------------------------------------------------------------------
router = APIRouter(prefix="/break-retest", tags=["break-retest"])


@router.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "break_retest_service"}


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    """Stateless: pass daily + intraday candles directly, get signals back.
    Use this if you're already pulling bars from MT5/OANDA/Yahoo in your
    main app and just want this service to score them."""
    if not req.intraday_candles:
        raise HTTPException(status_code=400, detail="intraday_candles is required")
    return run_all_setups(req.symbol, req.timeframe, req.daily_candles, req.intraday_candles)


@router.post("/candles/daily")
def push_daily_candles(symbol: str, candles: list[Candle]) -> dict:
    store.push_daily(symbol, candles)
    return {"stored": len(candles), "buffer_size": len(store.daily.get(symbol, []))}


@router.post("/candles/intraday")
def push_intraday_candles(symbol: str, candles: list[Candle]) -> dict:
    store.push_intraday(symbol, candles)
    return {"stored": len(candles), "buffer_size": len(store.intraday.get(symbol, []))}


@router.get("/signals", response_model=AnalyzeResponse)
def get_signals(symbol: str = "XAUUSD", timeframe: str = "5m") -> AnalyzeResponse:
    """Stateful mode: reads from the in-memory buffer built up via the
    /candles/daily and /candles/intraday push endpoints."""
    daily, intraday = store.get(symbol)
    if not intraday:
        raise HTTPException(
            status_code=404,
            detail=f"No intraday candles buffered for {symbol}. POST to /break-retest/candles/intraday first.",
        )
    return run_all_setups(symbol, timeframe, daily, intraday)


# --------------------------------------------------------------------------
# Live mode endpoints (Yahoo Finance polling)
# --------------------------------------------------------------------------
class LiveStatus(BaseModel):
    running: bool
    source: LiveSource
    symbol: str
    poll_interval_seconds: int
    last_bar_time: Optional[datetime]
    last_poll_at: Optional[datetime]
    last_error: Optional[str]
    connected_websocket_clients: int
    mt5_available: bool


@router.get("/live/status", response_model=LiveStatus)
def live_status() -> LiveStatus:
    running = _poller_task is not None and not _poller_task.done()
    return LiveStatus(
        running=running,
        source=_live_state["source"],
        symbol=_live_state["symbol"],
        poll_interval_seconds=POLL_INTERVAL_SECONDS,
        last_bar_time=_live_state["last_bar_time"],
        last_poll_at=_live_state["last_poll_at"],
        last_error=_live_state["last_error"],
        connected_websocket_clients=len(ws_manager.active),
        mt5_available=MT5_AVAILABLE,
    )


@router.post("/live/start")
async def live_start(source: LiveSource = DEFAULT_LIVE_SOURCE, symbol: Optional[str] = None) -> LiveStatus:
    if source == LiveSource.MT5 and not MT5_AVAILABLE:
        raise HTTPException(
            status_code=400,
            detail="source='mt5' requires the MetaTrader5 package, which is only importable on "
                   "Windows with a running MT5 terminal. This environment doesn't have it -- "
                   "run this service on your Windows machine instead, or use source='yfinance'.",
        )
    try:
        start_live_poller(source, symbol)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return live_status()


@router.post("/live/stop")
async def live_stop() -> LiveStatus:
    stop_live_poller()
    return live_status()


@router.get("/signals/live", response_model=AnalyzeResponse)
def get_live_signals() -> AnalyzeResponse:
    """Poll this whenever you want the latest computed levels + signals from
    the background Yahoo Finance feed. Reflects the last completed poll --
    does not block waiting for a fresh fetch."""
    latest = _live_state["latest"]
    if latest is None:
        raise HTTPException(
            status_code=503,
            detail="Live poller has no data yet. Check /break-retest/live/status, "
                   "or POST /break-retest/live/start if it isn't running.",
        )
    return latest


@router.get("/signals/live/recent", response_model=list[Signal])
def get_recent_live_signals(limit: int = 20) -> list[Signal]:
    """Just the newly-triggered signals from recent polls (most recent
    first), rather than the full recomputed level/signal snapshot."""
    return _live_state["recent_signals"][:limit]


@router.websocket("/ws/signals")
async def ws_signals(websocket: WebSocket):
    """Connect here to receive new signals pushed the moment they trigger,
    instead of polling GET /signals/live. Message shape:
        {"type": "signal", "data": {...Signal fields...}}
        {"type": "error", "message": "..."}
    """
    await ws_manager.connect(websocket)
    try:
        while True:
            # We don't expect inbound messages, but need to await something
            # to detect disconnects promptly.
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)


# --------------------------------------------------------------------------
# Standalone app (only used when running this file directly with uvicorn)
# --------------------------------------------------------------------------
@asynccontextmanager
async def _lifespan(app: FastAPI):
    start_live_poller()
    try:
        yield
    finally:
        stop_live_poller()


app = FastAPI(title="Break & Retest Signal Service", version="1.1.0", lifespan=_lifespan)
app.include_router(router)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8100)
