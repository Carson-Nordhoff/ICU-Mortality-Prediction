from utils.logger import get_logger

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedGroupKFold, RandomizedSearchCV
from src.models.get_model import get_model
from src.models.evaluate import evaluate

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    brier_score_loss,
)

training_logger = get_logger(__name__)

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

        preprocessor = get_preprocessor(numeric_features, categorical_features)

        clf = Pipeline(steps=[
            ('preprocessor', preprocessor),
            (model_name, get_model(model_name, scale_pos_weight))
        ])

        clf.fit(X_train, y_train)
        preds = clf.predict(X_val)
        probs = clf.predict_proba(X_val)[:, 1]

        acc, precision, recall, f1, avg_precision, roc_auc, brier_score = evaluate(y_val, preds, probs, fold)

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
        training_logger.info(f"Model: {model_name}")
        performance = cross_validate_model(X, y, groups, sgkf, numeric_features, categorical_features, model_name)
        default_model_performances.update(performance)

    return default_model_performances

def coarse_hypertune(X, y, groups, sgkf, numeric_features, categorical_features, model_name, params):

    #MANUALLY GRAB A SPLIT USING THE SGKF

    outer_sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    train_idx, val_idx = next(outer_sgkf.split(X, y, groups))

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    groups_train, groups_val = groups.iloc[train_idx], groups.iloc[val_idx]

    scale_pos_weight = sum(y_train==0) / sum(y_train==1)

    preprocessor = get_preprocessor(numeric_features, categorical_features)

    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        (model_name, get_model(model_name, scale_pos_weight))
    ])

    training_logger.info("------------------")
    training_logger.info(f"Model Name: {model_name}")

    randomized_search = RandomizedSearchCV(
        estimator=clf,
        param_distributions=params[model_name],
        cv=sgkf,
        n_iter=10,
        scoring="recall",
        error_score="raise"
    )

    randomized_search.fit(X_train, y_train, groups=groups_train)

    best_model = randomized_search.best_estimator_

    y_preds = best_model.predict(X_val)
    y_probs = best_model.predict_proba(X_val)[:, 1]

    acc = accuracy_score(y_val, y_preds)
    precision = precision_score(y_val, y_preds)
    recall = recall_score(y_val, y_preds)
    f1 = f1_score(y_val, y_preds)
    avg_precision = average_precision_score(y_val, y_probs)
    roc_auc = roc_auc_score(y_val, y_probs)
    brier_score = brier_score_loss(y_val, y_probs)

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
            'clf': clf,
            'acc': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'avg_precision': avg_precision,
            'roc_auc': roc_auc,
            'brier_score': brier_score
        }
    }

    return randomized_search, performance