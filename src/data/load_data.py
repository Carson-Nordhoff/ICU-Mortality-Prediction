from utils.directories import CSV_DIR
import pandas as pd

#load data from raw csvs
def load_data():

    raw_data = []

    for csv_file in CSV_DIR:
        df = pd.read_csv(csv_file)
        df.columns = df.columns.str.lower()
        raw_data.append(df)

    return raw_data

def inspect_data(data):
    for raw_csv in data:
        print(raw_csv.info())
        print(raw_csv.head())


