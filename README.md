# Paper Trading System

A small, modular paper-trading service for XAUUSD (gold) and forex. It pulls
OHLCV candles from a pluggable data provider (Yahoo Finance, MT5, or OANDA),
computes indicators (EMA, RSI, ADX/DI), runs a rule-based strategy, and
exposes the result over a FastAPI HTTP API — plus logs every signal to a
local SQLite database.

## Project layout

```
paper-trading/
├── app/
│   ├── main.py                  FastAPI app: /signal, /history, /health
│   ├── worker.py                Standalone script: fetch data → run strategy → print signal
│   ├── config.py                Symbol, timeframe, indicator periods, provider selection
│   │
│   ├── providers/                Market data sources (all implement DataProvider)
│   │   ├── base.py               Abstract interface every provider must follow
│   │   ├── factory.py            create_provider(name) → picks the right provider
│   │   ├── yahoo_provider.py     Yahoo Finance via yfinance (no credentials needed)
│   │   ├── mt5_provider.py       MetaTrader5 terminal (Windows + MT5 installed only)
│   │   └── oanda_provider.py     OANDA v3 REST API (practice or live account)
│   │
│   ├── indicators/                Pure indicator math (pandas/numpy only, no I/O)
│   │   ├── trend.py               ema()
│   │   ├── momentum.py            rsi()
│   │   ├── adx.py                 adx_di()  → (+DI, -DI, ADX)
│   │   ├── volume.py              obv(), cmf(), relative_volume()
│   │   └── volatile.py            (placeholder for ATR / Bollinger / Keltner later)
│   │
│   ├── strategy/                  Trading logic
│   │   ├── base.py                Strategy ABC + generic TradingSignal dataclass
│   │   └── ema_rsi_adx.py         EMARSIADXStrategy: the actual BUY/SELL/HOLD rules
│   │
│   ├── models/
│   │   └── signal.py               TradingSignal dataclass actually returned by the strategy
│   │                                (symbol, action, price, time, ema, rsi, adx, ±DI, ...)
│   │
│   └── database/
│       └── database.py             SQLite: init_db(), save_signal(), get_recent_signals()
│
└── tests/
    ├── conftest.py                Puts the project root on sys.path (see "Running tests")
    ├── test_indicators.py         Unit tests for indicator math (synthetic data)
    ├── test_provider.py           Factory + provider tests (network calls mocked)
    ├── test_strategy.py           Strategy decision-logic tests (BUY/SELL/HOLD branches)
    ├── test_worker.py             worker.run() tested with a fake provider
    └── test_api.py                FastAPI endpoints via TestClient
```

## How the pieces fit together

```
              ┌──────────────┐
              │   config.py  │  SYMBOL, TIMEFRAME, EMA/RSI/ADX periods, DATA_PROVIDER
              └──────┬───────┘
                     │
     ┌───────────────┴───────────────┐
     │                               │
┌────▼─────┐                  ┌──────▼───────┐
│  main.py │  (FastAPI app)   │  worker.py   │  (standalone script)
└────┬─────┘                  └──────┬───────┘
     │  create_provider(...)          │  create_provider(...)
     └───────────────┬─────────────────┘
                      │
              ┌───────▼────────┐
              │ providers/      │  get_history(symbol, timeframe, bars) → DataFrame
              │ factory.py      │  (OHLCV, DatetimeIndex)
              └───────┬────────┘
                      │
              ┌───────▼────────┐
              │ strategy/        │  strategy.generate_signal(symbol, df)
              │ ema_rsi_adx.py   │    1. prepare(df)   → adds EMA/RSI/ADX/+DI/-DI columns
              │                  │       (calls indicators/*)
              │                  │    2. apply BUY/SELL/HOLD rules to the last row
              └───────┬────────┘
                      │  TradingSignal
              ┌───────▼────────┐
              │ database/        │  save_signal(...) / get_recent_signals(...)
              │ database.py      │  (SQLite, data/papertrade.db)
              └────────────────┘
```

- **Provider swap is transparent** — `DATA_PROVIDER` in `config.py` (or an explicit
  argument to `create_provider("yahoo"|"mt5"|"oanda")`) decides where candles
  come from. Everything downstream (indicators, strategy, API) only ever talks
  to the `DataProvider` interface, never a specific provider.
- **Indicators are pure functions** — no I/O, no provider knowledge, just
  pandas/numpy in → pandas/numpy out. That's what makes them independently
  unit-testable.
- **The strategy owns the decision logic** — `prepare()` computes indicator
  columns, `generate_signal()` reads the last row and returns a
  `TradingSignal`. Swapping in a different rule set later just means adding a
  new class in `strategy/` that implements the same `Strategy` interface.

## Running the API

```bash
cd paper-trading
pip install -r requirements.txt   # fastapi, uvicorn, pandas, numpy, yfinance, requests
uvicorn app.main:app --reload
```

- `GET /signal` — fetches fresh candles, runs the strategy, saves and returns
  the signal.
- `GET /history?limit=20` — recent signals from SQLite.
- `GET /health` — liveness check.

## Running the worker script

```bash
python -m app.worker
```

Fetches candles for `SYMBOL`/`TIMEFRAME` (from `config.py`) via Yahoo Finance
by default, runs the strategy once, and prints the resulting signal.

## Running the tests

The tests are real `pytest` tests — network calls and MT5/OANDA connections
are mocked out, so the whole suite runs offline in a few seconds.

```bash
cd paper-trading
python -m pytest tests/ -v
```

**Run this from the `paper-trading` folder** (the parent of both `app/` and
`tests/`), not from inside `tests/`. `tests/conftest.py` also adds the
project root to `sys.path` automatically, so the suite works even if pytest
is invoked from a different working directory.

Do **not** run individual test files as scripts (`python -m test_indicators`)
— they're pytest modules now (they use `assert`, fixtures, and mocking), not
standalone scripts. Use `pytest`.

### What's covered

| File                  | Covers |
|-----------------------|--------|
| `test_indicators.py`  | EMA, RSI, ADX/±DI, OBV, CMF, relative volume — math correctness on synthetic OHLCV data |
| `test_provider.py`    | `create_provider()` selection/errors, OANDA missing-key guard, Yahoo `get_history()` shape (yfinance call mocked) |
| `test_strategy.py`    | BUY / SELL / HOLD branches of `EMARSIADXStrategy` (indicators forced via monkeypatch), plus one end-to-end run with real indicator math |
| `test_worker.py`      | `worker.run()` with a fake provider — checks it returns a valid `TradingSignal` |
| `test_api.py`         | `/health`, `/signal`, `/history` via FastAPI `TestClient`, with a scratch SQLite DB per test |

## Configuration

All tunable parameters live in `app/config.py`:

- `DATA_PROVIDER` — `"yahoo"`, `"mt5"`, or `"oanda"`
- `SYMBOL`, `TIMEFRAME`
- `EMA_LEN`, `RSI_LEN`, `RSI_OB`/`RSI_OS`, `ADX_LEN`, `ADX_SMOOTH`, `ADX_LEVEL`
- OANDA credentials are read from environment variables (`OANDA_API_KEY`,
  `OANDA_ACCOUNT_ID`, `OANDA_ENV`) rather than hardcoded — set these before
  using `DATA_PROVIDER = "oanda"`.

## Notes on the providers

- **Yahoo** (`yahoo_provider.py`) — works out of the box, no credentials.
  Maps `"XAUUSD"` → the Yahoo ticker `"XAUUSD=X"` and translates
  `"1h"`/`"4h"` timeframes to yfinance's `"60m"` interval.
- **MT5** (`mt5_provider.py`) — only works on Windows with the MetaTrader5
  terminal and package installed. Importing this module elsewhere won't
  crash the app; it only raises if you actually try to use it.
- **OANDA** (`oanda_provider.py`) — talks to the practice or live v3 REST API
  depending on `OANDA_ENV`. Requires `OANDA_API_KEY` (and `OANDA_ACCOUNT_ID`
  for `get_current_price`).
