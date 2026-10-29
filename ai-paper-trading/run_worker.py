import logging

from app.database.database import init_database
from app.workers.trading_worker import TradingWorker


def main():

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    init_database() 
    worker = TradingWorker()
    worker.run()


if __name__ == "__main__":
    main()