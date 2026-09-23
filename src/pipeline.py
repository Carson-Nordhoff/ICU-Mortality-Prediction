#MODEL TO-DO#

#Handle mortality class imbalance
#Adjust system to multiple model analysis for optimal model

from src.data import clean_data
from src.data.load_data import load_data, inspect_data, insert_raw_data, load_clean_mimic_data
from src.data.clean_data import clean_data
from src.models.train_model import get_preprocessor
from src.models.evaluate import evaluate
from src.models.save_model import save_model, load_model

from utils.directories import ensure_directories
from utils.logger import get_logger

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedGroupKFold

pipeline_logger = get_logger(__name__)



numeric_features = [
    "age",
    "avg_heart_rate",
    "avg_systolic_bp",
    "avg_diastolic_bp",
    "avg_respiratory_rate",
    "avg_body_temp",
    "avg_spo2"
]
categorical_features = [
    "marital_status",
    "religion",
    "language",
    "insurance",
    "admission_type",
    "first_careunit"
]
binary_features = []

def pipeline():

    ensure_directories()

    raw_data = load_data()
    inspect_data(raw_data)

    insert_raw_data(raw_data)

    clean_data()
    df = load_clean_mimic_data()

    pipeline_logger.info(df.info())
    pipeline_logger.info(df.head())
    pipeline_logger.info(f'Missingness per column:\n {df.isnull().mean().round(3)}')

    X = df.drop(columns=['hospital_expire_flag'])
    y = df['hospital_expire_flag']
    groups = df['subject_id']

    pipeline_logger.info(f'Class balance for hospital_expire_flag: {y.value_counts().to_dict()}')
    pipeline_logger.info(f'Percentile balance for hospital_expire_flag: {y.value_counts(normalize=True).round(3).to_dict()}')

    sgkf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)

    models = {}

    for fold, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups)):

        pipeline_logger.info(f'----------Fold: {fold}----------')

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        pipeline_logger.info(f'Train balance: {y_train.value_counts(normalize=True).round(3).to_dict()}')
        pipeline_logger.info(f'Test balance: {y_test.value_counts(normalize=True).round(3).to_dict()}')

        train_subjects = set(groups.iloc[train_idx])
        test_subjects = set(groups.iloc[test_idx])
        assert train_subjects.isdisjoint(test_subjects), f"Leaked subject_ids across train/test"

        preprocessor = get_preprocessor(numeric_features, categorical_features)

        clf = Pipeline(steps=[
            ('preprocessor', preprocessor),
            (f'classifier', LogisticRegression(class_weight='balanced'))
        ])

        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        probs = clf.predict_proba(X_test)[:, 1]

        acc, precision, recall, f1, avg_precision, roc_auc = evaluate(y_test, preds, probs)

        models[fold] = {
            'model': clf,
            'acc': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'avg_precision': avg_precision,
            'roc_auc': roc_auc
        }

    dummy_model = models[0]

    save_model(dummy_model, 0)
    dummy_model = load_model(0)

if __name__ == "__main__":
    pipeline()