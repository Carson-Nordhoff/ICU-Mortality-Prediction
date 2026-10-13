from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

MODELS = ["Logistic Regression", "Random Forest", "XGBoost"]

def get_model(name, scale_pos_weight):

    match name:
        case "Logistic Regression":
            return LogisticRegression(
                        max_iter=1000,
                        class_weight='balanced'
                    )
        case "Random Forest":
            return RandomForestClassifier(
                        n_estimators=100,
                        class_weight='balanced'
                    )
        case "XGBoost":
            return xgb.XGBClassifier(
                    learning_rate = 0.3,
                    max_depth = 6,
                    subsample = 1.0,
                    colsample_bytree = 1.0,
                    random_state = 42,
                    scale_pos_weight = scale_pos_weight
            )