#MODEL TO-DO#

#IN THE MIDDLE OF MAKING FINE HYPERTUNING
#AMOUNT OF STEPS BASED ON "PRIORITY" OF PARAM
#ALSO CHECK IF THE NUMBER IS WITHIN MIN/MAX
#IF LESS THEN MIN THEN MIN
#IF MORE THAN MAX THEN MAX

import time
import json

from utils.git_commit_hash import commit_hash
from utils.format_dataframe import format_dataframe_info
from src.data.load_processed_data import load_clean_mimic_data
from src.models.train import (
    train_default_models,
    make_fine_tune_params,
    hypertune,
    coarse_tune_models
)
from src.models.get_model_objects import MODELS, get_model, get_preprocessor
from src.models.icu_mortality_model_selection import top_k_models
from src.models.save_model import save_model, load_model
from src.models.evaluate import evaluate

from utils.directories import ensure_directories
from utils.logger import get_ml_system_logger
from utils.read_yaml import model_selection_yaml, hyper_parameter_yaml, formatted_hyper_parameter_yaml

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline

import mlflow


ml_pipeline_logger = get_ml_system_logger(__name__)

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

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Experiment 10")

    mlflow_tags = {
        "git commit hash": commit_hash,
        "ml.note.content": "View dashboard: [Training Report](https://physionet.org/content/mimiciii/1.4/)",
        "SQL Version": "PostgreSQL 18.6",
        "Random Seed": "42",
        "Features": json.dumps(categorical_features + numeric_features)
    }

    with mlflow.start_run(run_name=f"ml_pipeline_run_{time.ctime()}", tags=mlflow_tags):

        ensure_directories()

        df = load_clean_mimic_data()

        ml_pipeline_logger.info(f"DataFrame Info:\n {format_dataframe_info(df)}")
        ml_pipeline_logger.info(df.head())

        ml_pipeline_logger.info(f'Missingness per column:\n {df.isnull().mean().round(3)}')

        X = df.drop(columns=['icu_expire_flag'])
        y = df['icu_expire_flag']
        all_data_scale_pos_weight = sum(y == 0) / sum(y == 1)

        groups = df['subject_id']

        ml_pipeline_logger.info(f'Class balance for icu_expire_flag: {y.value_counts().to_dict()}')
        ml_pipeline_logger.info(f'Percentile balance for icu_expire_flag: {y.value_counts(normalize=True).round(3).to_dict()}')

        #Full Data split
        #To keep a chunk of data for final metric test
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

        all_train_idx, all_test_idx = next(sgkf.split(X, y, groups))

        X_train, X_final_test = X.iloc[all_train_idx], X.iloc[all_test_idx]
        y_train, y_final_test = y.iloc[all_train_idx], y.iloc[all_test_idx]
        groups_train, groups_final_test = groups.iloc[all_train_idx], groups.iloc[all_test_idx]

        cv_sgkf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)

        #FIRST PASS: Train all default models
        trained_default_models = train_default_models(X_train, y_train, groups_train, cv_sgkf, numeric_features, categorical_features, MODELS, mlflow_tags)

        #Get top k performing models
        selection_config = model_selection_yaml()
        top_models = top_k_models(trained_default_models, selection_config, selection_config["num_models_hyperparameter_tune"])

        #SECOND PASS: General hypertune top k models to get best model
        hyperparameters = formatted_hyper_parameter_yaml()

        hypertuned_models, performances = coarse_tune_models(X_train, y_train, groups_train, cv_sgkf, numeric_features, categorical_features, top_models, hyperparameters, "randomized_search", mlflow_tags)

        #Select best model from coarse search
        top_model = top_k_models(performances, selection_config, selection_config["num_top_models"])[0]

        #THIRD PASS: Large compute hypertune search for best model

        parameters = hyper_parameter_yaml()

        params = make_fine_tune_params(hypertuned_models, top_model, parameters)

        best_model_hypertuned, best_model_performance = hypertune(X_train, y_train, groups_train, cv_sgkf, numeric_features, categorical_features, top_model, params, "grid_search", "fine_tune_", mlflow_tags)

        #Last train and test for top model with optimal params
        #THIS CAN USE THE GET_PIPELINE MODULARIZED FUNC
        #BUT FIRST SEE BELOW
        #CHECK DON'T I NEED TO PASS THE TRAIN CHECK THE GROUPS TEST?

        best_clf = Pipeline(steps=[
            ('preprocessor', get_preprocessor(numeric_features, categorical_features)),
            (top_model, get_model(top_model, all_data_scale_pos_weight))
        ])

        best_clf.set_params(**best_model_hypertuned.best_params_)

        with mlflow.start_run(run_name="final_test_" + top_model, nested=True, tags=mlflow_tags):

            best_clf.fit(X_train, y_train) #Put in groups_train??
            y_preds = best_clf.predict(X_final_test)
            y_probs = best_clf.predict_proba(X_final_test)[:, 1]

            ml_pipeline_logger.info("Final Model Performance")

            # log the parameters
            mlflow.log_params(best_model_hypertuned.best_params_)

            # set a tagv
            mlflow.set_tags({"stage": "final_test", "model": top_model})

            acc, precision, recall, f1, avg_precision, roc_auc, brier_score = evaluate(y_final_test, y_preds, y_probs, "best_model", top_model, None)

            # log the model
            mlflow.sklearn.log_model(
                sk_model=best_clf,
                name="final_model_" + top_model,
                input_example=None,
                signature=None,
                skops_trusted_types=["numpy.dtype", "sklearn.tree._tree.Tree", "xgboost.core.Booster", "xgboost.sklearn.XGBClassifier"]
            )

            mlflow.log_metrics({
                "accuracy": acc,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "avg_precision": avg_precision,
                "roc_auc": roc_auc,
                "brier_score": brier_score
            })

        with mlflow.start_run(run_name="final_model", nested=True, tags=mlflow_tags):

            # set a tag
            mlflow.set_tags({"stage": "final", "model": top_model})

            #Train a final optimal model on all the data.

            final_clf = Pipeline(steps=[
                ('preprocessor', get_preprocessor(numeric_features, categorical_features)),
                (top_model, get_model(top_model, all_data_scale_pos_weight))
            ])

            final_clf.set_params(**best_model_hypertuned.best_params_)
            final_clf.fit(X, y)

            # log the model
            mlflow.sklearn.log_model(
                sk_model=final_clf,
                name="final_model_" + top_model,
                input_example=None,
                signature=None,
                skops_trusted_types=["numpy.dtype", "sklearn.tree._tree.Tree", "xgboost.core.Booster", "xgboost.sklearn.XGBClassifier"]
            )

        save_model(final_clf)
        load_check = load_model()

if __name__ == "__main__":
    pipeline()