from utils.directories import TRAINED_MODEL
from utils.logger import get_logger
import joblib

save_model_logger = get_logger(__name__)

def save_model(model, fold):
    joblib.dump(model, f'{TRAINED_MODEL}{fold}')
    save_model_logger.info(f"Saved model to {TRAINED_MODEL}{fold}")

def load_model(fold):

    clf = joblib.load(f'{TRAINED_MODEL}{fold}')
    save_model_logger.info(f"Loaded model from {TRAINED_MODEL}{fold}")

    return clf