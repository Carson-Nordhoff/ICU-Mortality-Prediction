from utils.directories import CSV_DIR
from utils.logger import get_logger
import pandas as pd

data_logger = get_logger(__name__)

#load data from raw csvs
def load_data():

    raw_data = []

    data_logger.info("Loading raw data...")
    for csv_file in CSV_DIR:

        data_logger.info(f"Successfully loaded {csv_file}")
        df = pd.read_csv(csv_file)
        df.columns = df.columns.str.lower()
        raw_data.append(df)

    return raw_data

def inspect_data(data):
    for raw_csv in data:
        data_logger.info(f"Data inspection for {raw_csv}")
        data_logger.info(raw_csv.head())
        data_logger.info(raw_csv.info())


