from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"
ADMISSIONS_DIR = DATA_DIR / "ADMISSIONS.csv"

def ensure_directories():

    for dir in [DATA_DIR]:
        Path(dir).mkdir(parents=True, exist_ok=True)