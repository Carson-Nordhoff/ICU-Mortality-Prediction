from src.data import clean_data
from utils.directories import ensure_directories
from src.data.load_data import load_data, inspect_data, insert_raw_data
from src.data.clean_data import clean_data


def pipeline():

    ensure_directories()

    raw_data = load_data()
    inspect_data(raw_data)

    insert_raw_data(raw_data)

    clean_data()

if __name__ == "__main__":
    pipeline()