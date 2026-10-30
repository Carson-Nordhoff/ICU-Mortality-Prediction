#MODEL TO-DO#

from src.data.load_data import (
    format_dataframe_info,
    load_clean_mimic_data
)
from src.models.train import get_preprocessor, train_default_models, make_fine_tune_params, fine_hypertune, coarse_tune_models
from src.models.get_model_objects import MODELS
from src.models.icu_mortality_model_selection import top_k_models
from src.models.save_model import save_model, load_model

from src.models.evaluate import evaluate

from src.models.get_model_objects import get_model

from utils.directories import ensure_directories
from utils.logger import get_logger

from sklearn.model_selection import StratifiedGroupKFold

from sklearn.pipeline import Pipeline

from utils.read_yaml import formatted_hyper_parameter_yaml
from utils.read_yaml import model_selection_yaml
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

    df = load_clean_mimic_data()

    pipeline_logger.info(f"DataFrame Info:\n {format_dataframe_info(df)}")
    pipeline_logger.info(df.head())

    pipeline_logger.info(f'Missingness per column:\n {df.isnull().mean().round(3)}')

    X = df.drop(columns=['icu_expire_flag'])
    y = df['icu_expire_flag']
    all_data_scale_pos_weight = sum(y == 0) / sum(y == 1)

    groups = df['subject_id']

    pipeline_logger.info(f'Class balance for icu_expire_flag: {y.value_counts().to_dict()}')
    pipeline_logger.info(f'Percentile balance for icu_expire_flag: {y.value_counts(normalize=True).round(3).to_dict()}')

    #Full Data split
    #To keep a chunk of data for final metric test
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    all_train_idx, all_test_idx = next(sgkf.split(X, y, groups))

    X_train, X_final_test = X.iloc[all_train_idx], X.iloc[all_test_idx]
    y_train, y_final_test = y.iloc[all_train_idx], y.iloc[all_test_idx]
    groups_train, groups_final_test = groups.iloc[all_train_idx], groups.iloc[all_test_idx]

    cv_sgkf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)

    #FIRST PASS: Train all default models
    trained_default_models = train_default_models(X_train, y_train, groups_train, cv_sgkf, numeric_features, categorical_features, MODELS)

    #Get top k performing models
    selection_config = model_selection_yaml()
    top_models = top_k_models(trained_default_models, selection_config, selection_config["num_models_hyperparameter_tune"])

    #SECOND PASS: General hypertune top k models to get best model
    hyperparameters = formatted_hyper_parameter_yaml()

    hypertuned_models, performances = coarse_tune_models(X_train, y_train, groups_train, cv_sgkf, numeric_features, categorical_features, top_models, hyperparameters)

    #Select best model from coarse search
    top_model = top_k_models(performances, selection_config, selection_config["num_top_models"])[0]
    pipeline_logger.info("OVERALL BEST MODEL")
    pipeline_logger.info(top_model)


    #THIRD PASS: Large compute hypertune search for best model
    pipeline_logger.info(hypertuned_models[top_model].best_params_)

    parameters = hyper_parameter_yaml()

    params = make_fine_tune_params(hypertuned_models, top_model, parameters)

    best_model_hypertuned, best_model_performance = fine_hypertune(X_train, y_train, groups_train, cv_sgkf, numeric_features, categorical_features, top_model, params)

    #Last train and test for top model with optimal params
    #THIS CAN USE THE GET_PIPELINE MODULARIZED FUNC
    #BUT FIRST SEE BELOW
    #CHECK DON'T I NEED TO PASS THE TRAIN CHECK THE GROUPS TEST?
    preprocessor = get_preprocessor(numeric_features, categorical_features)

    model123 = get_model(top_model, all_data_scale_pos_weight)

    best_clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        (top_model, model123)
    ])

    best_clf.set_params(**best_model_hypertuned.best_params_)

    best_clf.fit(X_train, y_train) #Put in groups_train??
    y_preds = best_clf.predict(X_final_test)
    y_probs = best_clf.predict_proba(X_final_test)[:, 1]

    pipeline_logger.info("Final Model Performance")
    evaluate(y_final_test, y_preds, y_probs, "best_model", top_model, None)

    #Train a final optimal model on all the data.
    preprocessor = get_preprocessor(numeric_features, categorical_features)

    final_clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        (top_model, get_model(top_model, all_data_scale_pos_weight))
    ])

    final_clf.fit(X, y)
    save_model(final_clf)
    load_check = load_model()

if __name__ == "__main__":
    pipeline()