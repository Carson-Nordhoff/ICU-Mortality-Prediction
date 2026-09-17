from pathlib import Path
import os

ROOT_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = ROOT_DIR / "data"
ADMISSIONS = RAW_DATA_DIR / "ADMISSIONS.csv"
PATIENTS = RAW_DATA_DIR / "PATIENTS.csv"

CSV_DIRS = [ADMISSIONS, PATIENTS]

SQL_DIR = ROOT_DIR / "sql"
SQL_CODE = SQL_DIR / "clean_data.sql"

ARTIFACT_DIR = ROOT_DIR / "artifacts"
LOG_DIR = ARTIFACT_DIR / "logs"
PIPELINE_LOG_DIR = LOG_DIR / "pipeline_logs.log"

def ensure_directories():

    for dir in [RAW_DATA_DIR]:
        Path(dir).mkdir(parents=True, exist_ok=True)