#TO-DO Explicit data type given when reading CSVs to avoid multiple-datatype error
#instead of low_memory = False?

#get rid of rows with error=1??

import time
import pandas as pd
import gzip, subprocess

from utils.directories import CSV_DIRS, CREATE_TABLES
from utils.logger import get_sql_system_logger
from utils.get_engine import get_engine
from utils.format_dataframe import format_dataframe_info
from utils.read_yaml import general_config
from utils.db_config import *

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

def write_data_to_postgres(csv_path: Path, table: str):

    cmd = [
        "psql", "-h", DB_HOST, "-p", str(DB_PORT), "-U", DB_USER, "-d", DB_NAME, #env variables
        "-v", "ON_ERROR_STOP=1", "-1", #tells postgres to stop on the first error and exit with a failure code
        "-c", f"TRUNCATE {table}", #empty the table
        "-c", f"\\copy {table} FROM pstdin WITH (FORMAT csv, HEADER true)", #copy the data
    ]

    env = {**os.environ, "PGPASSWORD": DB_PASSWORD} #copies current env variables, add password

    file_size = os.path.getsize(csv_path) #get compressed file size for %loaded calcs
    start = time.time()
    prev = time.time()

    #opens the file twice
    #raw is used for %loaded calcs
    #gzip is used for reading the zipped file
    with open(csv_path, "rb") as raw, gzip.GzipFile(fileobj=raw) as f:

        #starts psql and passes cmd, pipeline, and environment details
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, env=env)


        try:

            #reads CHUNK_SIZE MB a time
            while chunk := f.read(CHUNK_SIZE):

                #sends the chunk into the pipeline
                #but waits for postgres to read before sending more
                proc.stdin.write(chunk)

                now = time.time()

                pct = raw.tell() / file_size * 100
                elapsed_min = (now - start) / 60
                eta_min = elapsed_min / pct * (100 - pct)
                chunk_speed = now - prev

                data_logger.info(
                    f"Loading: {table} | {pct:.3f}% | Elapsed {elapsed_min:.3f} Min | Chunk Speed: {chunk_speed:.3f} Sec | {eta_min:.3f} Min Remaining"
                )
                prev = now

        #catches if psql exits early
        #but ignores it for the RuntimeError
        except BrokenPipeError:
            pass

        #closes connection at EOF
        finally:
            proc.stdin.close()

        if proc.wait() != 0:
            raise RuntimeError(f"psql load failed for {table}")

    data_logger.info(f"{table} loaded in {(time.time() - start) / 60:.3f} min.")

def copy_tables():

    for csv in CSV_DIRS:
        write_data_to_postgres(csv, csv.name.split(".")[0].lower())