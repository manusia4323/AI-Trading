from app.factories.backtest_factory import create_backtest_engine
from app.config import TradingConfig


def main():

    engine = create_backtest_engine()

    try:

        result = engine.run(
            TradingConfig.SYMBOL,
            TradingConfig.TIMEFRAME,
            5000,
        )
    
    finally:
        engine.close()

    print(result.statistics)


if __name__ == "__main__":
    main()