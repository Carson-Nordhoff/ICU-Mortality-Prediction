from src.data import clean_data
from utils.directories import ensure_directories
from src.data.load_data import load_data, inspect_data, insert_raw_data, load_clean_mimic_data
from src.data.clean_data import clean_data
from utils.logger import get_logger

pipeline_logger = get_logger(__name__)

def pipeline():

    ensure_directories()

    raw_data = load_data()
    inspect_data(raw_data)

    insert_raw_data(raw_data)

    clean_data()
    df = load_clean_mimic_data()

    pipeline_logger.info(df.info())
    pipeline_logger.info(df.head())


if __name__ == "__main__":
    pipeline()