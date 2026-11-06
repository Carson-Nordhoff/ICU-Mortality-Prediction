import pandas as pd
from utils.get_engine import get_engine
from utils.logger import get_ml_system_logger

processed_data_logger = get_ml_system_logger(__name__)

def load_clean_mimic_data() -> pd.DataFrame:

    engine = get_engine()

    processed_data_logger.info("Getting cleaned data from mimic3 database.")
    df = pd.read_sql("SELECT * FROM clean_mimic_data", engine)

    #TO-DO add a check that warns against
    #data misalignment with the manually specified data
    processed_data_logger.info(f'Columns: {df.columns.to_list()}')

    processed_data_logger.info("Loaded cleaned data from mimic3 database.")

    engine.dispose()

    return df