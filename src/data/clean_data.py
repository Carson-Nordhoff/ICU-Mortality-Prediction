import psycopg
from utils.directories import ALTER_TABLES, CLEAN_DATA
from utils.db_config import DB_NAME, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT
from utils.logger import get_logger

from psycopg_pool import ConnectionPool

clean_data_logger = get_logger(__name__)

pool = ConnectionPool(
    conninfo=f"dbname={DB_NAME} user={DB_USER} password={DB_PASSWORD} host={DB_HOST} port={DB_PORT}",
    min_size=2,
    max_size=10,
    open=True
)

def clean_data(file_path):

    clean_data_logger.info(f"Reading SQL code from {file_path}.")

    with pool.connection() as conn:
        with conn.cursor() as cur:

            clean_data_logger.info("Dropping existing clean_mimic_data table if present.")
            cur.execute("DROP TABLE IF EXISTS clean_mimic_data;")

            with open(file_path, 'r') as f:
                sql_query = f.read()
                cur.execute(sql_query)

def build_icu_table():

    clean_data_logger.info(f"Altering raw data tables...")
    clean_data(ALTER_TABLES)
    clean_data_logger.info("Successfully altered tables.")

    clean_data_logger.info(f"Cleaning raw data tables...")
    clean_data(CLEAN_DATA)
    clean_data_logger.info("Successfully created final table.")

    pool.close()