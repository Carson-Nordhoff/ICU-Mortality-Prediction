from utils.directories import CSV_DIRS
from utils.logger import get_logger
from sqlalchemy import create_engine
from utils.db_config import DB_NAME, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT
import pandas as pd
import os

data_logger = get_logger(__name__)

#gets an sqlalchemy engine to interpret between python and sql
def get_engine():
    return create_engine(
        f"postgresql+psycopg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )

#load data from raw csvs
def load_data():

    raw_data = []

    data_logger.info("Loading raw data...")
    for csv_file in CSV_DIRS:

        data_logger.info(f"Successfully loaded {csv_file}")
        df = pd.read_csv(csv_file)
        df.columns = df.columns.str.lower()
        raw_data.append(df)

    return raw_data

#inspects head and info per csv
def inspect_data(data):
    for raw_csv in data:
        data_logger.info(f"Data inspection for {raw_csv}")
        data_logger.info(raw_csv.head())
        data_logger.info(raw_csv.info())

#inserts data into postgre
def insert_raw_data(data):

    engine = get_engine()
    data_logger.info("Inserting data into mimic3 database.")
    data[0].to_sql('admissions', engine, if_exists='replace', index=False)
    data_logger.info("admissions loaded successfully.")

def load_clean_mimic_data():

    engine = get_engine()

    data_logger.info("Getting cleaned data from mimic3 database.")
    df = pd.read_sql("SELECT * FROM admissions", engine)
    data_logger.info("Loaded cleaned data from mimic3 database.")

    return df