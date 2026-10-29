from app.database import Base
from app.database import engine

# IMPORTANT
from app.database.database import init_database
import app.database.models


def main():

    init_database()
    
    Base.metadata.create_all(bind=engine)

    print("Database created successfully.")


if __name__ == "__main__":
    main()