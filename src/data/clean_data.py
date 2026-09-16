import psycopg
from utils.directories import SQL_CODE
from utils.db_config import DB_NAME, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT
from utils.logger import get_logger

clean_data_logger = get_logger(__name__)


def clean_data():

    clean_data_logger.info(f"Reading SQL code from {SQL_CODE}.")
    with open(SQL_CODE, 'r') as f:
        sql_query = f.read()

    #connect to postgres database
    conn = psycopg.connect(
        dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT
    )

    #execute the sql code
    with conn.cursor() as cur:
        clean_data_logger.info("Dropping existing clean_mimic_data table if present.")
        cur.execute("DROP TABLE IF EXISTS clean_mimic_data;")
        clean_data_logger.info("Running SQL code to build clean_mimic_data table.")
        cur.execute(sql_query)
    conn.commit()
    conn.close()
    clean_data_logger.info("clean_mimic_data created successfully.")