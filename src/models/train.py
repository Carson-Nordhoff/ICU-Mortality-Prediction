from utils.logger import get_logger

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.pipeline import Pipeline
from src.models.get_model import MODELS, get_model
from src.models.evaluate import evaluate

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

def train_default_models(X, y, groups, sgkf, numeric_features, categorical_features):

    models = {}

    for fold, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups)):

        training_logger.info(f'----------Fold: {fold}----------')

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        training_logger.info(f'Train balance: {y_train.value_counts(normalize=True).round(3).to_dict()}')
        training_logger.info(f'Test balance: {y_test.value_counts(normalize=True).round(3).to_dict()}')

        scale_pos_weight = sum(y_train == 0) / sum(y_train == 1)

        train_subjects = set(groups.iloc[train_idx])
        test_subjects = set(groups.iloc[test_idx])
        assert train_subjects.isdisjoint(test_subjects), f"Leaked subject_ids across train/test"

        preprocessor = get_preprocessor(numeric_features, categorical_features)

        for model_name in MODELS:

            clf = Pipeline(steps=[
                ('preprocessor', preprocessor),
                (model_name, get_model(model_name, scale_pos_weight))
            ])

            clf.fit(X_train, y_train)
            preds = clf.predict(X_test)
            probs = clf.predict_proba(X_test)[:, 1]

            acc, precision, recall, f1, avg_precision, roc_auc, brier_score = evaluate(y_test, preds, probs, fold)

            models[(fold, model_name)] = {
                'clf': clf,
                'acc': acc,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'avg_precision': avg_precision,
                'roc_auc': roc_auc,
                'brier_score': brier_score
            }

    return models