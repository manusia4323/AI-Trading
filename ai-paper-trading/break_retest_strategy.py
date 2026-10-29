from typing import Optional
import logging

import pandas as pd

from app.config import TradingConfig
from app.models.signal import TradingSignal
from app.strategy.base import Strategy
from app.strategy.parameter import StrategyParameter

logger = logging.getLogger(__name__)


class BreakRetestStrategy(Strategy):
    """
    Price-action strategy, no indicators: three setups combined.

      1. Previous day high/low break + retest
      2. Opening range break + retest (first bar of each calendar day at
         this dataframe's own timeframe - see note in _compute_opening_range_levels)
      3. Order block retest (last opposite-colored candle before an impulse move)

    Ported from break_retest_service.py / backtest_adx_rsi_ema_indicator.py's
    build_break_retest_signals(), adapted to this app's Strategy interface so
    it can be selected/configured alongside EMA/RSI/ADX from the dashboard.
    """

    @classmethod
    def schema(cls) -> list[StrategyParameter]:

        return [

            StrategyParameter(
                key="retest_tolerance_pips",
                label="Retest Tolerance (Pips)",
                type="number",
                default=15.0,
                minimum=1.0,
                maximum=100.0,
                step=0.5,
            ),

            StrategyParameter(
                key="ob_tolerance_pips",
                label="Order Block Tolerance (Pips)",
                type="number",
                default=8.0,
                minimum=1.0,
                maximum=50.0,
                step=0.5,
            ),

            StrategyParameter(
                key="ob_lookback_bars",
                label="Order Block Lookback (Bars)",
                type="number",
                default=20,
                minimum=5,
                maximum=100,
                step=1,
            ),

            # Boolean toggles rendered as 0/1 number inputs since the dashboard's
            # strategy editor only special-cases type == "number" (see
            # strategy-editor.js) - a "boolean" input type isn't handled there.
            StrategyParameter(
                key="enable_prev_day",
                label="Enable Previous Day H/L (1=on, 0=off)",
                type="number",
                default=1,
                minimum=0,
                maximum=1,
                step=1,
            ),

            StrategyParameter(
                key="enable_opening_range",
                label="Enable Opening Range (1=on, 0=off)",
                type="number",
                default=1,
                minimum=0,
                maximum=1,
                step=1,
            ),

            StrategyParameter(
                key="enable_order_block",
                label="Enable Order Block (1=on, 0=off)",
                type="number",
                default=1,
                minimum=0,
                maximum=1,
                step=1,
            ),

            StrategyParameter(
                key="stop_loss_pips",
                label="Stop Loss (Pips)",
                type="number",
                default=50,
                minimum=10,
                maximum=5000,
                step=10,
            ),

            StrategyParameter(
                key="risk_reward_ratio",
                label="Risk Reward Ratio",
                type="number",
                default=2.0,
                minimum=0.5,
                maximum=10.0,
                step=0.1,
            ),

        ]

    def __init__(
        self,
        config: dict,
    ):

        super().__init__()

        config = config or {}

        #
        # Strategy Parameters
        #

        self.retest_tolerance_pips = float(
            config.get("retest_tolerance_pips", 15.0)
        )

        self.ob_tolerance_pips = float(
            config.get("ob_tolerance_pips", 8.0)
        )

        self.ob_lookback_bars = int(
            config.get("ob_lookback_bars", 20)
        )

        self.enable_prev_day = bool(
            int(config.get("enable_prev_day", 1))
        )

        self.enable_opening_range = bool(
            int(config.get("enable_opening_range", 1))
        )

        self.enable_order_block = bool(
            int(config.get("enable_order_block", 1))
        )

        self.stop_loss_pips = float(
            config.get("stop_loss_pips", 50)
        )

        self.risk_reward_ratio = float(
            config.get("risk_reward_ratio", 2.0)
        )

    @property
    def name(self):

        return "Break & Retest"

    @property
    def minimum_bars(self):

        # Order-block lookback + the 3-bar break/retest/reaction window, plus
        # headroom. Previous-day and opening-range levels degrade gracefully
        # (simply absent -> no signal from that setup) rather than erroring
        # if fewer than 2 calendar days of history happen to be available.
        return max(
            50,
            self.ob_lookback_bars + 10,
        )

    #
    # Indicators / levels
    #

    def prepare(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        logger.info(
            "Computing break & retest levels..."
        )

        df = df.copy()

        df = self._compute_prev_day_levels(df)

        df = self._compute_opening_range_levels(df)

        pdh_long = pdh_short = pd.Series(False, index=df.index)
        or_long = or_short = pd.Series(False, index=df.index)
        ob_long = ob_short = pd.Series(False, index=df.index)

        if self.enable_prev_day:

            pdh_long, pdh_short = self._break_retest_reaction(
                df,
                "prev_day_high",
                "prev_day_low",
            )

        if self.enable_opening_range:

            or_long, or_short = self._break_retest_reaction(
                df,
                "opening_range_high",
                "opening_range_low",
            )

            # don't fire on the opening-range bar itself
            or_long = or_long & ~df["_is_first_bar_of_day"]
            or_short = or_short & ~df["_is_first_bar_of_day"]

        if self.enable_order_block:

            ob_long_list, ob_short_list = self._order_block_signals(df)

            ob_long = pd.Series(ob_long_list, index=df.index)
            ob_short = pd.Series(ob_short_list, index=df.index)

        df["buy_signal"] = pdh_long | or_long | ob_long

        df["sell_signal"] = pdh_short | or_short | ob_short

        setup = pd.Series("", index=df.index, dtype=object)
        setup = setup.mask(pdh_long | pdh_short, "previous_day_high_low_retest")
        setup = setup.mask(or_long | or_short, "opening_range_retest")
        setup = setup.mask(ob_long | ob_short, "order_block_retest")
        df["setup"] = setup

        return df.drop(columns="_is_first_bar_of_day", errors="ignore")

    def _compute_prev_day_levels(self, df: pd.DataFrame) -> pd.DataFrame:

        daily = df.resample("1D").agg({"High": "max", "Low": "min"}).dropna()
        daily["prev_day_high"] = daily["High"].shift(1)
        daily["prev_day_low"] = daily["Low"].shift(1)

        session_date = df.index.normalize()
        df["prev_day_high"] = session_date.map(daily["prev_day_high"])
        df["prev_day_low"] = session_date.map(daily["prev_day_low"])

        return df

    def _compute_opening_range_levels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Opening range = first bar of each calendar day at this dataframe's
        own timeframe. XAUUSD trades ~24/5 so there's no single universal
        "market open" - this sidesteps timezone mismatches across providers.
        If you specifically want a strict NY 09:30 opening range, filter df
        to session-open bars before this runs."""

        df = df.copy()
        df["_date"] = df.index.normalize()
        df["opening_range_high"] = df.groupby("_date")["High"].transform("first")
        df["opening_range_low"] = df.groupby("_date")["Low"].transform("first")
        df["_is_first_bar_of_day"] = ~df["_date"].duplicated(keep="first")

        return df.drop(columns="_date")

    def _break_retest_reaction(
        self,
        df: pd.DataFrame,
        level_high_col: str,
        level_low_col: str,
    ):
        """Vectorized break -> retest -> reaction pattern:
          bar t-2: price breaks through the level
          bar t-1: price comes back to retest it
          bar t  : reaction candle confirms (closes back on the breakout side)
        """

        tol = self.retest_tolerance_pips * TradingConfig.PIP_SIZE

        broke_above = df["High"].shift(2) > df[level_high_col]
        retested_above = df["Low"].shift(1) <= df[level_high_col] + tol
        reaction_long = (df["Close"] > df[level_high_col]) & (df["Close"] > df["Open"])
        long_signal = broke_above & retested_above & reaction_long

        broke_below = df["Low"].shift(2) < df[level_low_col]
        retested_below = df["High"].shift(1) >= df[level_low_col] - tol
        reaction_short = (df["Close"] < df[level_low_col]) & (df["Close"] < df["Open"])
        short_signal = broke_below & retested_below & reaction_short

        return long_signal.fillna(False), short_signal.fillna(False)

    def _order_block_signals(self, df: pd.DataFrame):
        """Loop-based (not vectorized) since each bar needs to look back for
        the most recent opposite-colored candle. O(n * lookback), same cost
        class as the standalone backtest script's version."""

        tol = self.ob_tolerance_pips * TradingConfig.PIP_SIZE
        is_up = (df["Close"] > df["Open"]).to_numpy()
        is_down = (df["Close"] < df["Open"]).to_numpy()
        open_ = df["Open"].to_numpy()
        high = df["High"].to_numpy()
        low = df["Low"].to_numpy()
        close = df["Close"].to_numpy()
        n = len(df)
        lookback = self.ob_lookback_bars

        long_signal = [False] * n
        short_signal = [False] * n

        for i in range(1, n):

            window_start = max(0, i - lookback)

            # --- bearish order block: last up-close candle before this bar, for shorts ---
            ob_idx = None
            for j in range(i - 1, window_start - 1, -1):
                if is_up[j]:
                    ob_idx = j
                    break
            if ob_idx is not None:
                zone_top = max(open_[ob_idx], close[ob_idx])
                zone_bottom = low[ob_idx]
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
                zone_bottom = min(open_[ob_idx], close[ob_idx])
                zone_top = high[ob_idx]
                touched = (zone_bottom - tol) <= low[i] <= (zone_top + tol)
                strong_reaction = is_up[i] and close[i] > close[i - 1]
                if touched and strong_reaction:
                    long_signal[i] = True

        return long_signal, short_signal

    #
    # Signal
    #

    def generate_signal(

        self,

        symbol: str,

        df: pd.DataFrame,

    ) -> Optional[TradingSignal]:

        if "buy_signal" not in df.columns:

            df = self.prepare(df)

        last = df.iloc[-1]

        price = float(last["Close"])

        action = "HOLD"

        reason = "No setup"

        stop_loss = None

        take_profit = None

        if bool(last["buy_signal"]):

            action = "BUY"

            reason = f"Break & retest: {last['setup']}"

        elif bool(last["sell_signal"]):

            action = "SELL"

            reason = f"Break & retest: {last['setup']}"

        #
        # Stop Loss / Take Profit
        #

        if action in ("BUY", "SELL"):

            sl_distance = (

                self.stop_loss_pips

                * TradingConfig.PIP_SIZE

            )

            tp_distance = (

                sl_distance

                * self.risk_reward_ratio

            )

            if action == "BUY":

                stop_loss = (

                    price - sl_distance

                )

                take_profit = (

                    price + tp_distance

                )

            else:

                stop_loss = (

                    price + sl_distance

                )

                take_profit = (

                    price - tp_distance

                )

            print(
                f"******** {action} ******** "
                f"{df.index[-1]} "
                f"setup={last['setup']} "
                f"price={price:.2f}"
            )

        print(
            f"{df.index[-1]} | "
            f"Action={action} | "
            f"setup={last['setup'] or '-'}"
        )

        return TradingSignal(

            symbol=symbol,

            action=action,

            price=price,

            time=df.index[-1],

            reason=reason,

            stop_loss=stop_loss,

            take_profit=take_profit,

        )

    def generate_dataframe(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        return self.prepare(df)
