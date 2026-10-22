from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from utils.logger import get_logger

get_model_logger = get_logger(__name__)

MODELS = ["logistic_regression", "random_forest", "xgboost"]

#add type hinting for return type?

def get_model(name: str, scale_pos_weight: float):

    match name:
        case "logistic_regression":
            return LogisticRegression(
                        class_weight='balanced',
                        random_state=42
                    )
        case "random_forest":
            return RandomForestClassifier(
                        class_weight='balanced',
                        random_state=42
                    )
        case "xgboost":
            return xgb.XGBClassifier(
                        scale_pos_weight = scale_pos_weight,
                        random_state=42
                    )
        case _:
            get_model_logger.critical("No model matches provided model.")
            raise ValueError("No model matches provided model")