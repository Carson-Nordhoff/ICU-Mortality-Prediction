from utils.logger import get_logger, get_ml_system_logger

from src.models.get_model_objects import get_pipeline, get_hypertuner
from sklearn.model_selection import StratifiedGroupKFold, RandomizedSearchCV, GridSearchCV
from src.models.evaluate import evaluate

import pandas as pd
import mlflow

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "avg_precision": "average_precision",
    "roc_auc": "roc_auc",
    "brier_score": "neg_brier_score",
}

from utils.mlflow_config import set_mlflow

training_logger = get_ml_system_logger(__name__)

numeric = int | float

def contain_range(nums: list[numeric], min_: numeric, max_: numeric) -> list[numeric]:
    return [min_ if n < min_ else max_ if n > max_ else n for n in nums]

def make_multiplicative_range(val: numeric, range: list[numeric], param_min: numeric, param_max: numeric) -> list[numeric]:

    nums = [val * modifier for modifier in range]

    return contain_range(nums, param_min, param_max)

def make_additive_range(val: numeric, modifier: numeric, param_min: numeric, param_max: numeric) -> list[numeric]:

    nums = [val - modifier, val, val + modifier]

    return contain_range(nums, param_min, param_max)

def make_fine_tune_params(hypertuned_models, top_model, parameters):

    params = {}

    for parameter_name, value in hypertuned_models[top_model].best_params_.items():

        info = parameter_name.split("__")

        param_specs = parameters[top_model][info[1]]

        param_range = []

        if param_specs["fine_tune"] == "Y" and param_specs["testing_tune"] == "Y":

            match parameter_name:

                case "xgboost__gamma" | "xgboost__reg_alpha":

                    if abs(value) <= 0.05:
                        param_range = param_specs["alt"]
                    else:
                        param_range = make_multiplicative_range(value, param_specs["mult"], param_specs["min"], param_specs["max"])

                case "xgboost__min_child_weight":

                    if value <= 5:
                        param_range = make_additive_range(value, param_specs["add"], param_specs["min"], param_specs["max"])
                    else:
                        param_range = make_multiplicative_range(value, param_specs["mult"], param_specs["min"], param_specs["max"])

                case _:

                    if param_specs["mult"] == "N/A" and param_specs["add"] == "N/A":
                        param_range = param_specs["alt"]

                    elif param_specs["mult"] == "N/A" and param_specs["add"]:
                        param_range = make_additive_range(value, param_specs["add"], param_specs["min"], param_specs["max"])

                    elif param_specs["mult"] and param_specs["add"] == "N/A":
                        param_range = make_multiplicative_range(value, param_specs["mult"], param_specs["min"], param_specs["max"])

                    else:
                        training_logger.error(f"Cannot fine hypertune on empty parameter list for param {parameter_name}")
                        raise ValueError(f"Cannot fine hypertune on empty parameter list for param {parameter_name}")

        else:
            param_range = [value]

        params[parameter_name] = param_range

    return params

def cross_validate_model(X: pd.DataFrame, y: pd.DataFrame, groups: pd.Series[str], sgkf: StratifiedGroupKFold, numeric_features: list[str], categorical_features: list[str], model_name: str, mlflow_run_name, mlflow_tags) -> dict[str, dict[str, float]]:

    performance = {}

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):

        training_logger.info(f'----------Fold: {fold}----------')

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        groups_train, groups_val = groups.iloc[train_idx], groups.iloc[val_idx]

        training_logger.info(f'Train balance: {y_train.value_counts(normalize=True).round(3).to_dict()}')
        training_logger.info(f'Validation balance: {y_val.value_counts(normalize=True).round(3).to_dict()}')

        scale_pos_weight = sum(y_train == 0) / sum(y_train == 1)

        train_subjects = set(groups_train)
        val_subjects = set(groups_val)
        assert train_subjects.isdisjoint(val_subjects), f"Leaked subject_ids across train/validation"

        clf = get_pipeline(model_name, scale_pos_weight, numeric_features, categorical_features)

        with mlflow.start_run(run_name="default_cv_" + model_name + f"_fold_{fold}", nested=True, tags=mlflow_tags):

            clf.fit(X_train, y_train)  # Pass this groups_train?
            preds = clf.predict(X_val)
            probs = clf.predict_proba(X_val)[:, 1]

            #log the parameters
            mlflow.log_params(clf.named_steps[model_name].get_params())

            #set a tag
            mlflow.set_tags({"stage": "default", "model": model_name, "fold": fold})

            acc, precision, recall, f1, avg_precision, roc_auc, brier_score = evaluate(y_val, preds, probs, "default", model_name, fold)

            mlflow.log_metrics({
                "accuracy": acc,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "avg_precision": avg_precision,
                "roc_auc": roc_auc,
                "brier_score": brier_score
            })

            performance[(model_name, fold)] = {
                'clf': clf,
                'metrics': {
                    'accuracy': acc,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'avg_precision': avg_precision,
                    'roc_auc': roc_auc,
                    'brier_score': brier_score
                }
            }

    top_model = sorted(performance, key=lambda key: performance[key]["metrics"]["recall"], reverse=True)[0]

    return {model_name: performance[top_model]}

def train_default_models(X, y, groups, sgkf, numeric_features, categorical_features, MODELS, mlflow_tags):

    default_model_performances = {}

    with mlflow.start_run(run_name="default_model_cv", nested=True, tags=mlflow_tags):

        for model_name in MODELS:
            training_logger.info(f"Performing Cross-Validation for default model: {model_name}")

            with mlflow.start_run(run_name="default_cv_" + model_name, nested=True, tags=mlflow_tags):
                performance = cross_validate_model(X, y, groups, sgkf, numeric_features, categorical_features, model_name, "default", mlflow_tags)
                default_model_performances.update(performance)

    return default_model_performances

def hypertune(X, y, groups, sgkf, numeric_features, categorical_features, model_name, params, tuner_name: str, STAGE, mlflow_tags):

    scale_pos_weight = sum(y==0) / sum(y==1)

    clf = get_pipeline(model_name, scale_pos_weight, numeric_features, categorical_features)

    training_logger.info(f"STEP: {STAGE}_{model_name}")

    tuner = get_hypertuner(tuner_name, clf, params, sgkf)

    with mlflow.start_run(run_name=STAGE + model_name, nested=True, tags=mlflow_tags):

        tuner.fit(X, y, groups=groups)

        index = tuner.best_index_

        metrics = {}
        str_output = f"Model: {model_name} | "
        for measurement in scoring:

            if measurement == "brier_score":
                result = -tuner.cv_results_[f"mean_test_{measurement}"][index]

            else:
                result = tuner.cv_results_[f"mean_test_{measurement}"][index]

            metrics[measurement] = result
            str_output += f"{measurement}: {result:.3f} | "

        #log the parameters
        mlflow.log_params(tuner.best_params_)

        # set a tag
        mlflow.set_tags({"stage": STAGE, "model": model_name})

        #log the model
        mlflow.sklearn.log_model(
            sk_model=tuner.best_estimator_,
            name=STAGE + model_name,
            input_example=None,
            signature=None,
            skops_trusted_types=["numpy.dtype", "sklearn.tree._tree.Tree", "xgboost.core.Booster", "xgboost.sklearn.XGBClassifier"]
        )

        mlflow.log_metrics(metrics)

        training_logger.info(str_output)

        performance = {
            model_name: {
                'clf': tuner.best_estimator_,
                'metrics': metrics
            }
        }

    return tuner, performance

def coarse_tune_models(X, y, groups, cv_sgkf, numeric_features, categorical_features, top_models, hyperparameters, tuner_name, mlflow_tags):

    hypertuned_models = {}
    performances = {}

    with mlflow.start_run(run_name="coarse_tuning", nested=True, tags=mlflow_tags):

        for model_name in top_models:
            hypertuned_models[model_name], performance = hypertune(X, y, groups, cv_sgkf, numeric_features, categorical_features, model_name, hyperparameters[model_name], tuner_name, "coarse_tuning_", mlflow_tags)
            performances.update(performance)

    return hypertuned_models, performances
