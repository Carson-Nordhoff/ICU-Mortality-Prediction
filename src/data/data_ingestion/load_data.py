#TO-DO Explicit data type given when reading CSVs to avoid multiple-datatype error
#instead of low_memory = False?

import time
import gzip
import pandas as pd

from utils.directories import CSV_DIRS, CREATE_TABLES
from utils.logger import get_sql_system_logger
from utils.get_engine import get_engine
from utils.format_dataframe import format_dataframe_info
from utils.read_yaml import general_config

from pathlib import Path
import os

NROWS = general_config()["peak_data"]["nrows"]
CHUNK_SIZE = general_config()["chunking"]["chunk_size"]

data_logger = get_sql_system_logger(__name__)

def peak_data() -> None:

    data_logger.info("Taking a peak at raw data.")
    for csv_dir in CSV_DIRS:

        df = pd.read_csv(csv_dir, nrows=NROWS)

        csv_name = Path(csv_dir).stem.lower()

        data_logger.info(f"Data inspection for {csv_name}")
        data_logger.info(f"DataFrame Head: {df.head()}")
        data_logger.info(f"DataFrame Info:\n {format_dataframe_info(df)}")

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
                    while chunk := f.read(CHUNK_SIZE):
                        copy.write(chunk)

                        amount_read = raw_opener.tell() #amount of bytes read
                        percent_read = amount_read / file_size * 100

                        curr_time = time.time()

                        eta = (file_size - amount_read) / CHUNK_SIZE * ((curr_time-prev_time)/60)

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