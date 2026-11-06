#TO-DO
#MAKE READING IN DATA MORE EFFICIENT
#MAINLY CHART EVENTS?

#Add alternate quick run vs. full run options?

from utils.directories import ensure_directories
from src.data.data_ingestion.load_data import peak_data, create_blank_tables, write_data_to_postgres
from src.data.data_ingestion.clean_data import build_icu_table

def process_data():

    ensure_directories()

    peak_data()

    #create the blank tables
    #create_blank_tables()

    #raw_data = load_data()
    #inspect_data(raw_data)

    #insert_raw_data(raw_data)

    #write_data_to_postgres()

    #build_icu_table()

if __name__ == "__main__":
    process_data()