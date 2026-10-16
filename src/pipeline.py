#MODEL TO-DO#

#Handle mortality class imbalance
#Adjust system to multiple model analysis for optimal model

from src.data.load_data import (
    load_data,
    inspect_data,
    insert_raw_data,
    load_clean_mimic_data
)
from src.data.clean_data import build_icu_table
from src.models.train import get_preprocessor, train_default_models
from src.models.get_model import MODELS, get_model
from src.models.evaluate import evaluate
from src.models.model_selection import top_k_models
from src.models.save_model import save_model, load_model

from utils.directories import ensure_directories
from utils.logger import get_logger

from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedGroupKFold

import io

from utils.read_yaml import hyper_parameter_yaml

pipeline_logger = get_logger(__name__)

vitals = [
    "avg_heart_rate",
    "avg_systolic_bp",
    "avg_diastolic_bp",
    "avg_respiratory_rate",
    "avg_body_temp",
    "avg_spo2"
]
labs = [
    "avg_creatinine",
    "avg_bun",
    "avg_wbc",
    "avg_hemoglobin",
    "avg_platelets",
    "avg_sodium",
    "avg_potassium",
    "avg_glucose",
    "avg_lactate"
]

numeric_features = ["age"] + vitals + labs

categorical_features = [
    "marital_status",
    "religion",
    "language",
    "insurance",
    "admission_type",
    "first_careunit",
    "gender"
]
binary_features = []

def pipeline():

    ensure_directories()

    raw_data = load_data()
    inspect_data(raw_data)

    insert_raw_data(raw_data)

    build_icu_table()
    df = load_clean_mimic_data()

    buffer = io.StringIO()
    df.info(buf=buffer)
    df_info_str = buffer.getvalue()
    pipeline_logger.info("DataFrame Info:\n%s", df_info_str)
    pipeline_logger.info(df.head())

    pipeline_logger.info(f'Missingness per column:\n {df.isnull().mean().round(3)}')

    X = df.drop(columns=['icu_expire_flag'])
    y = df['icu_expire_flag']
    all_data_scale_pos_weight = sum(y == 0) / sum(y == 1) #don't remove need for commented out final pipeline

    groups = df['subject_id']

    pipeline_logger.info(f'Class balance for icu_expire_flag: {y.value_counts().to_dict()}')
    pipeline_logger.info(f'Percentile balance for icu_expire_flag: {y.value_counts(normalize=True).round(3).to_dict()}')

    sgkf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)

    default_models = train_default_models(X, y, groups, sgkf, numeric_features, categorical_features)

    top_models = top_k_models(default_models)
    pipeline_logger.info(f'top models: {top_models}')

    #IN THE MIDDLE OF IMPLEMENTING HYPERPARAMETER TUNING

    #FIRST PASS DONE: Get top k performing models on baseline
    #SECOND PASS: General hypertune top k models to get best model
    #THIRD PASS: Large compute hypertune search for best model


    #
    # final_clf = Pipeline(steps=[
    #         ('preprocessor', get_preprocessor(numeric_features, categorical_features)),
    #         ('classifier', get_model(best_model_name, all_data_scale_pos_weight))
    # ])
    # final_clf.fit(X, y)
    #
    # save_model(final_clf)
    # load_check = load_model()

if __name__ == "__main__":
    pipeline()