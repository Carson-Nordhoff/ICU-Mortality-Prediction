from utils.logger import get_logger

from src.models.get_model_objects import get_pipeline, get_hypertuner
from sklearn.model_selection import StratifiedGroupKFold, RandomizedSearchCV, GridSearchCV
from src.models.evaluate import evaluate

import pandas as pd

training_logger = get_logger(__name__)

def make_float_param_list(val: float, step: int, steps_per_side: int) -> list[float]:

    mult = 1 + (step/100)
    div = 1 - (step/100)

    back = [val * (div**i) for i in range(1, steps_per_side+1)][::-1]
    front = [val * (mult**i) for i in range(1, steps_per_side+1)]

    return back + [val] + front

def make_int_param_list(val: int, step: int, steps_per_side: int) -> list[int]:

    back = [val - (i * step) for i in range(1, steps_per_side+1)][::-1]
    front = [val + (i * step) for i in range(1, steps_per_side+1)]

    return back + [val] + front

def make_fine_tune_params(hypertuned_models, top_model, parameters):

    params = {}

    for parameter_name, value in hypertuned_models[top_model].best_params_.items():

        info = parameter_name.split("__")

        type = parameters[top_model][info[1]]["type"]

        range = []
        if type == "randint":
            range = make_int_param_list(value, 1, 0)

        elif type == "uniform":
            range = make_float_param_list(value, 5, 0)

        elif type == "choice":
            range = parameters[top_model][info[1]]["values"]

        if range is None or not range:
            training_logger.error(f"Cannot fine hypertune on empty parameter list for param {parameter_name}")
            raise ValueError(f"Cannot fine hypertune on empty parameter list for param {parameter_name}")

        params[parameter_name] = range

        return params

def cross_validate_model(X: pd.DataFrame, y: pd.DataFrame, groups: pd.Series[str], sgkf: StratifiedGroupKFold, numeric_features: list[str], categorical_features: list[str], model_name: str) -> dict[int, dict[str, float]]:

    performance = {}

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):

        training_logger.info(f'----------Fold: {fold}----------')

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        training_logger.info(f'Train balance: {y_train.value_counts(normalize=True).round(3).to_dict()}')
        training_logger.info(f'Validation balance: {y_val.value_counts(normalize=True).round(3).to_dict()}')

        scale_pos_weight = sum(y_train == 0) / sum(y_train == 1)

        train_subjects = set(groups.iloc[train_idx])
        val_subjects = set(groups.iloc[val_idx])
        assert train_subjects.isdisjoint(val_subjects), f"Leaked subject_ids across train/validation"

        clf = get_pipeline(model_name, scale_pos_weight, numeric_features, categorical_features)

        clf.fit(X_train, y_train)
        preds = clf.predict(X_val)
        probs = clf.predict_proba(X_val)[:, 1]

        acc, precision, recall, f1, avg_precision, roc_auc, brier_score = evaluate(y_val, preds, probs, "default", model_name, fold)

        performance[(model_name, fold)] = {
            'clf': clf,
            'acc': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'avg_precision': avg_precision,
            'roc_auc': roc_auc,
            'brier_score': brier_score
        }

    return performance

def train_default_models(X, y, groups, sgkf, numeric_features, categorical_features, MODELS):

    default_model_performances = {}

    for model_name in MODELS:
        training_logger.info(f"Performing Cross-Validation for default model: {model_name}")
        performance = cross_validate_model(X, y, groups, sgkf, numeric_features, categorical_features, model_name)
        default_model_performances.update(performance)

    return default_model_performances

def hypertune(X, y, groups, sgkf, numeric_features, categorical_features, model_name, params, tuner_name: str):

    #MANUALLY GRAB A SPLIT USING THE SGKF

    outer_sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    train_idx, val_idx = next(outer_sgkf.split(X, y, groups))

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    groups_train, groups_val = groups.iloc[train_idx], groups.iloc[val_idx]

    scale_pos_weight = sum(y_train==0) / sum(y_train==1)

    clf = get_pipeline(model_name, scale_pos_weight, numeric_features, categorical_features)

    training_logger.info("------------------")
    training_logger.info(f"Model Name: {model_name}")

    tuner = get_hypertuner(tuner_name, clf, params, sgkf)

    tuner.fit(X_train, y_train, groups=groups_train)

    best_model = tuner.best_estimator_

    y_preds = best_model.predict(X_val)
    y_probs = best_model.predict_proba(X_val)[:, 1]

    acc, precision, recall, f1, avg_precision, roc_auc, brier_score = evaluate(y_val, y_preds, y_probs, "coarse_hypertune", model_name, None)

    training_logger.info(
        f"Model: {model_name} | "
        f"Accuracy: {acc:.3f} | "
        f"Precision: {precision:.3f} | "
        f"Recall: {recall:.3f} | "
        f"F1 Score: {f1:.3f} | "
        f"Average Precision: {avg_precision:.3f} | "
        f"ROC AUC: {roc_auc:.3f} | "
        f"Brier Score: {brier_score:.3f} |"
    )

    performance = {
        (model_name, 0): {
            'clf': clf, #Shouldn't this be best_model not clf??
            'acc': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'avg_precision': avg_precision,
            'roc_auc': roc_auc,
            'brier_score': brier_score
        }
    }

    return tuner, performance

def coarse_tune_models(X, y, groups, cv_sgkf, numeric_features, categorical_features, top_models, hyperparameters, tuner_name):

    hypertuned_models = {}
    performances = {}

    for model_name in top_models:
        hypertuned_models[model_name], performance = hypertune(X, y, groups, cv_sgkf, numeric_features, categorical_features, model_name, hyperparameters[model_name], tuner_name)
        performances.update(performance)

    return hypertuned_models, performances
