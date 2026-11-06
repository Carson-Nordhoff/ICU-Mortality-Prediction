from pathlib import Path
import os

ROOT_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = ROOT_DIR / "data"

ADMISSIONS = RAW_DATA_DIR / "ADMISSIONS.csv.gz"
PATIENTS = RAW_DATA_DIR / "PATIENTS.csv.gz"
ICUSTAYS = RAW_DATA_DIR / "ICUSTAYS.csv.gz"
CHARTEVENTS = RAW_DATA_DIR / "CHARTEVENTS.csv.gz"
LABEVENTS = RAW_DATA_DIR / "LABEVENTS.csv.gz"

CSV_DIRS = [
    ADMISSIONS,
    PATIENTS,
    ICUSTAYS,
    CHARTEVENTS,
    LABEVENTS]

SQL_DIR = ROOT_DIR / "sql"
ALTER_TABLES = SQL_DIR / "alter_tables.sql"
CLEAN_DATA = SQL_DIR / "clean_data.sql"
CREATE_TABLES = SQL_DIR / "postgres_create_tables.sql"

ARTIFACT_DIR = ROOT_DIR / "artifacts"
LOG_DIR = ARTIFACT_DIR / "logs"

ML_PIPELINE_LOG_DIR = LOG_DIR / "ml_pipeline_logs.log"
SQL_PIPELINE_LOG_DIR = LOG_DIR / "sql_pipeline_logs.log"
MODELS_DIR = ARTIFACT_DIR / "models"
METRICS_DIR = ARTIFACT_DIR / "metrics"

TRAINED_MODEL = MODELS_DIR / "trained_model.pkl"

CONFIGS_DIR = ROOT_DIR / "configs"
MODEL_SELECTION_YAML = CONFIGS_DIR / "model_selection.yaml"
COARSE_HYPERPARAMETER_YAML = CONFIGS_DIR / "coarse_hyperparameters.yaml"
GENERAL_CONFIG_YAML = CONFIGS_DIR / "general.yaml"

def ensure_directories():

    for dir in [RAW_DATA_DIR, ARTIFACT_DIR, LOG_DIR, MODELS_DIR, METRICS_DIR]:
        Path(dir).mkdir(parents=True, exist_ok=True)