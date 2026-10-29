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
  - STATELESS by default: POST candle data in, get signals out. This keeps it
    decoupled from your MT5/OANDA/Yahoo provider layer -- you feed it OHLC
    bars from whichever provider you like.
  - An OPTIONAL lightweight in-memory candle store is included so you can
    push bars incrementally (e.g. from a websocket feed) and just poll
    /signals afterward, instead of resending full history every call.
  - Reuses your existing naming conventions (PIP_SIZE, STOP_LOSS_PIPS,
    RISK_REWARD_RATIO, "5m" timeframe strings) so the output is easy to
    reconcile with your other services, even though this module does not
    import from your app package.

Run standalone:
    uvicorn break_retest_service:app --port 8100 --reload

Mount into an existing FastAPI app instead:
    from break_retest_service import router as break_retest_router
    app.include_router(break_retest_router)
"""

from __future__ import annotations

from datetime import datetime, date, time as dtime, timezone
from enum import Enum
from typing import Optional
from zoneinfo import ZoneInfo

from fastapi import APIRouter, FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

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
# Standalone app (only used when running this file directly with uvicorn)
# --------------------------------------------------------------------------
app = FastAPI(title="Break & Retest Signal Service", version="1.0.0")
app.include_router(router)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8100)
