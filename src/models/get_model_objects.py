from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

from utils.logger import get_logger

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.pipeline import Pipeline

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

def get_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    return ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

def get_pipeline(model_name: str, scale_pos_weight: float, numeric_features: list[str], categorical_features: list[str]) -> Pipeline:

    return Pipeline(steps=[
        ('preprocessor', get_preprocessor(numeric_features, categorical_features)),
        (model_name, get_model(model_name, scale_pos_weight))
    ])