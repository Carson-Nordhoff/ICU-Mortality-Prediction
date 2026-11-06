#TO-DO
#MAKE READING IN DATA MORE EFFICIENT
#MAINLY CHART EVENTS?

#Add alternate quick run vs. full run options?

from src.data.load_data import (
    create_blank_tables,
    write_data_to_postgres,
    load_data,
    inspect_data,
    insert_raw_data
)
from src.data.clean_data import build_icu_table

from utils.directories import ensure_directories

def process_data():

    ensure_directories()

    #create the blank tables
    create_blank_tables()

    #raw_data = load_data()
    #inspect_data(raw_data)

    #insert_raw_data(raw_data)

    write_data_to_postgres()

    build_icu_table()

if __name__ == "__main__":
    process_data()