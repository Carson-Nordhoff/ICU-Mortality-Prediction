#TO-DO Explicit data type given when reading CSVs to avoid multiple-datatype error
#instead of low_memory = False?

from utils.directories import CSV_DIRS
from utils.logger import get_logger
from utils.db_config import (
    DB_NAME,
    DB_USER,
    DB_PASSWORD,
    DB_HOST,
    DB_PORT
)

from sqlalchemy import create_engine, Engine
from pathlib import Path
import pandas as pd
import io

data_logger = get_logger(__name__)

#gets an sqlalchemy engine to interpret between python and sql
def get_engine() -> Engine:
    return create_engine(
        f"postgresql+psycopg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )

#load data from raw csvs
def load_data() -> dict[str, pd.DataFrame]:

    raw_data = {}

    data_logger.info("Loading raw data...")
    for csv_dir in CSV_DIRS:

        data_logger.info(f"Successfully loaded {csv_dir}")
        df = pd.read_csv(csv_dir, low_memory=False)
        df.columns = df.columns.str.lower()
        csv_name = Path(csv_dir).stem.lower()
        raw_data[csv_name] = df

    return raw_data

#inspects head and info per csv
def inspect_data(data: dict[str, pd.DataFrame]):
    for raw_csv in data.values():
        data_logger.info(f"Data inspection for {raw_csv}")
        data_logger.info(raw_csv.head())
        data_logger.info(f"DataFrame Info:\n {format_dataframe_info(raw_csv)}")

def format_dataframe_info(df: pd.DataFrame) -> str:

    buffer = io.StringIO()
    df.info(buf=buffer)
    df_info_str = buffer.getvalue()

    return df_info_str

#inserts data into postgre
def insert_raw_data(data: dict[str, pd.DataFrame]):

    engine = get_engine()
    data_logger.info("Inserting data into mimic3 database.")

    for df_name, raw_df in data.items():

        raw_df.to_sql(df_name, engine, if_exists='replace', index=False)
        data_logger.info(f"{df_name} loaded successfully.")

    engine.dispose()

def load_clean_mimic_data() -> pd.DataFrame:

    engine = get_engine()

    data_logger.info("Getting cleaned data from mimic3 database.")
    df = pd.read_sql("SELECT * FROM clean_mimic_data", engine)

    #TO-DO add a check that warns against
    #data misalignment with the manually specified data
    data_logger.info(f'Columns: {df.columns.to_list()}')

    data_logger.info("Loaded cleaned data from mimic3 database.")

    engine.dispose()

    return df