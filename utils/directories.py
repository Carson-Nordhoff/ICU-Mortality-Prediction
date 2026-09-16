from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = ROOT_DIR / "data"
ADMISSIONS = RAW_DATA_DIR / "ADMISSIONS.csv"
#PATIENTS = RAW_DATA_DIR / "PATIENTS.csv"

CSV_DIRS = [ADMISSIONS]

SQL_DIR = ROOT_DIR / "sql"
SQL_CODE = SQL_DIR / "create_table.sql"

def ensure_directories():

    for dir in [RAW_DATA_DIR]:
        Path(dir).mkdir(parents=True, exist_ok=True)