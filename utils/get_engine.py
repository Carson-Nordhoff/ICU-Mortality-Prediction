from sqlalchemy import Engine, create_engine
from utils.db_config import (
    DB_NAME,
    DB_USER,
    DB_PASSWORD,
    DB_HOST,
    DB_PORT
)

#gets an sqlalchemy engine to interpret between python and sql
def get_engine() -> Engine:
    return create_engine(
        f"postgresql+psycopg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )