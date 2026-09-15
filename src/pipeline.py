from utils.directories import ensure_directories
from src.data.load_data import load_data, inspect_data

def pipeline():

    ensure_directories()

    raw_data = load_data()
    inspect_data(raw_data)

if __name__ == "__main__":
    pipeline()