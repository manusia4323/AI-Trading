"""
mt5_signal_server.py

Runs your existing EMARSIADXStrategy against live MT5 data and broadcasts
BUY/SELL signals over WebSocket to connected clients - primarily the
SignalDisplayEA.mq5 Expert Advisor, which draws them as arrows on the MT5
chart.

This does NOT place any orders. It only generates and broadcasts signals.

Run from your paper-trading project root (same folder as app/):

    python mt5_signal_server.py

Requires: pip install websockets
"""

import asyncio
import json
import logging
from datetime import datetime, timezone

import websockets

from app.config import TradingConfig
from app.providers.mt5_provider import MT5Provider
from app.strategy.ema_rsi_adx import EMARSIADXStrategy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("mt5_signal_server")

HOST = "127.0.0.1"
PORT = 8765

POLL_SECONDS = 30   # how often to re-check for a new signal
BARS = 300

clients = set()


async def handle_client(websocket):
    clients.add(websocket)
    logger.info("EA connected (%d total)", len(clients))

    try:
        # We don't expect messages FROM the EA, just keep the connection
        # open and drop it cleanly on disconnect.
        async for _ in websocket:
            pass
    finally:
        clients.discard(websocket)
        logger.info("EA disconnected (%d total)", len(clients))


async def broadcast(message: dict):
    if not clients:
        logger.info("No EA connected - signal generated but not sent: %s", message)
        return

    payload = json.dumps(message)
    stale = []

    for ws in clients:
        try:
            await ws.send(payload)
        except websockets.ConnectionClosed:
            stale.append(ws)

    for ws in stale:
        clients.discard(ws)

    logger.info("Broadcast to %d client(s): %s", len(clients), payload)


async def signal_loop():
    provider = MT5Provider()
    strategy = EMARSIADXStrategy()

    provider.connect()
    logger.info("Connected to MT5")

    last_action = None

    try:
        while True:
            try:
                df = provider.get_history(
                    TradingConfig.SYMBOL,
                    TradingConfig.TIMEFRAME,
                    BARS,
                )

                signal = strategy.generate_signal(TradingConfig.SYMBOL, df)

                logger.info(
                    "%s @ %.5f - %s",
                    signal.action,
                    signal.price,
                    signal.reason,
                )

                # Only broadcast BUY/SELL, and only when it's a genuinely
                # new signal (not the same action repeated every cycle
                # while a trend continues).
                if signal.action in ("BUY", "SELL") and signal.action != last_action:

                    message = {
                        "type": "signal",
                        "symbol": signal.symbol,
                        "action": signal.action,
                        "price": signal.price,
                        "stop_loss": signal.stop_loss,
                        "take_profit": signal.take_profit,
                        "time": (
                            signal.time.isoformat()
                            if hasattr(signal.time, "isoformat")
                            else str(signal.time)
                        ),
                        "reason": signal.reason,
                    }

                    await broadcast(message)

                last_action = signal.action

            except RuntimeError as exc:
                # MT5Provider raises RuntimeError on connection issues
                # (see mt5_provider.py) - try to reconnect next cycle
                # rather than crashing the whole server.
                logger.error("MT5 error, will retry: %s", exc)

                try:
                    provider.disconnect()
                except Exception:
                    pass

                try:
                    provider.connect()
                except Exception:
                    logger.error("Reconnect attempt failed, retrying later")

            except Exception:
                logger.exception("Unexpected error in signal loop")

            await asyncio.sleep(POLL_SECONDS)

    finally:
        provider.disconnect()


async def main():
    async with websockets.serve(handle_client, HOST, PORT):
        logger.info("WebSocket server listening on ws://%s:%d", HOST, PORT)
        await signal_loop()


if __name__ == "__main__":
    asyncio.run(main())
