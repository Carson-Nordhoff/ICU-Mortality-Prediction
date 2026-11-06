#TO-DO Explicit data type given when reading CSVs to avoid multiple-datatype error
#instead of low_memory = False?

import time
import gzip

from utils.directories import CSV_DIRS, CREATE_TABLES
from utils.logger import get_sql_system_logger
from utils.get_engine import get_engine

from pathlib import Path
import os

data_logger = get_sql_system_logger(__name__)

def run_slq_file(sql_path: Path):

    sql = Path(sql_path).read_text(encoding="utf-8")

    engine = get_engine()

    try:
        with engine.begin() as conn:
            conn.exec_driver_sql(sql)

        data_logger.info(f"Successfully created blank tables.")

    finally:
        engine.dispose()

def create_blank_tables():
    run_slq_file(CREATE_TABLES)

def write_data_to_postgres():

    engine = get_engine()
    conn = engine.raw_connection()

    chunk_size = 10000000 #10MB

    try:
        with (conn.cursor() as cur):
            for csv_path in CSV_DIRS:

                table = csv_path.name.split(".")[0].lower()  # handles .csv and .csv.gz

                start = time.time()
                prev_time = time.time()

                #ISSUES WITH LOADING TAKING LONGER PER ITERATION

                cur.execute(f"TRUNCATE {table}")

                raw_opener = open(csv_path, "rb")
                gz_opener = gzip.GzipFile(fileobj=raw_opener)

                file_size = os.path.getsize(csv_path) # size of file in bytes
                data_logger.info(f'File Size: {file_size}')

                data_logger.info(f"Reading file: {table}. Size: {file_size / 1000000:.3f}MB")

                with raw_opener, gz_opener as f, cur.copy(
                    f"COPY {table} FROM STDIN WITH (FORMAT csv, HEADER true)"
                ) as copy:
                    while chunk := f.read(chunk_size):  # 10 MB at a time
                        copy.write(chunk)

                        amount_read = raw_opener.tell() #amount of bytes read
                        percent_read = amount_read / file_size * 100

                        curr_time = time.time()

                        eta = (file_size - amount_read) / chunk_size * ((curr_time-prev_time)/60)

                        data_logger.info(
                            f"Loaded {percent_read:.3f}% of data "
                            f"| Chunk Time: {(curr_time-prev_time):.3f} "
                            f"| Elapsed: {(curr_time-start)/60:.3f} Minutes "
                            f"| Remaining: {eta:.3f} Minutes."
                        )
                        prev_time = curr_time

                data_logger.info(f"Successfully copied {table} to postgres. Time to execute: {(time.time()-start):.3f} seconds.")

        conn.commit()

    finally:
        conn.close()
        engine.dispose()

#load data from raw csvs
# def load_data() -> dict[str, pd.DataFrame]:
#
#     raw_data = {}
#
#     data_logger.info("Loading raw data...")
#     for csv_dir in CSV_DIRS:
#
#         data_logger.info(f"Successfully loaded {csv_dir}")
#         df = pd.read_csv(csv_dir, low_memory=False)
#         df.columns = df.columns.str.lower()
#         csv_name = Path(csv_dir).stem.lower()
#         raw_data[csv_name] = df
#
#     return raw_data

#inspects head and info per csv
# def inspect_data(data: dict[str, pd.DataFrame]):
#     for raw_csv in data.values():
#         data_logger.info(f"Data inspection for {raw_csv}")
#         data_logger.info(raw_csv.head())
#         data_logger.info(f"DataFrame Info:\n {format_dataframe_info(raw_csv)}")

#inserts data into postgre
# def insert_raw_data(data: dict[str, pd.DataFrame]):
#
#     engine = get_engine()
#     data_logger.info("Inserting data into mimic3 database.")
#
#     for df_name, raw_df in data.items():
#
#         raw_df.to_sql(df_name, engine, if_exists='replace', index=False)
#         data_logger.info(f"{df_name} loaded successfully.")
#
#     engine.dispose()