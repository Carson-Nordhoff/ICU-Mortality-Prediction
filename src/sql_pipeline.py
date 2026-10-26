#THIS IS A TEMPORARY FILE TO MAKE IT EASIER TO RUN ITERATIONS WHILE TESTING CODE
#IT WILL EVENTUALLY BE RE-ABSORBED INTO THE PIPELINE

from src.data.load_data import (
    load_data,
    inspect_data,
    insert_raw_data
)
from src.data.clean_data import build_icu_table

from utils.directories import ensure_directories

def process_data():

    ensure_directories()

    raw_data = load_data()
    inspect_data(raw_data)

    insert_raw_data(raw_data)

    build_icu_table()

if __name__ == "__main__":
    process_data()