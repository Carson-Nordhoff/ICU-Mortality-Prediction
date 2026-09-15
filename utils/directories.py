from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"
ADMISSIONS = DATA_DIR / "ADMISSIONS.csv"
PATIENTS = DATA_DIR / "PATIENTS.csv"

CSV_DIR = [ADMISSIONS, PATIENTS]

def ensure_directories():

    for dir in [DATA_DIR]:
        Path(dir).mkdir(parents=True, exist_ok=True)