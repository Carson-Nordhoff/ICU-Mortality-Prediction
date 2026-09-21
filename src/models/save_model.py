from utils.directories import TRAINED_MODEL
from utils.logger import get_logger
import joblib

save_model_logger = get_logger(__name__)

def save_model(model):
    joblib.dump(model, TRAINED_MODEL)
    save_model_logger.info(f"Saved model to {TRAINED_MODEL}")

def load_model():

    clf = joblib.load(TRAINED_MODEL)
    save_model_logger.info(f"Loaded model from {TRAINED_MODEL}")

    return clf