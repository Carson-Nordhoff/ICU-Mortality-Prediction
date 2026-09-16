from src.data import clean_data
from src.data.load_data import load_data, inspect_data, insert_raw_data, load_clean_mimic_data
from src.data.clean_data import clean_data
from utils.directories import ensure_directories
from utils.logger import get_logger

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

pipeline_logger = get_logger(__name__)

def pipeline():

    ensure_directories()

    raw_data = load_data()
    inspect_data(raw_data)

    insert_raw_data(raw_data)

    clean_data()
    df = load_clean_mimic_data()

    pipeline_logger.info(df.info())
    pipeline_logger.info(df.head())

    X = df.drop(columns=['hospital_expire_flag'])
    y = df['hospital_expire_flag']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    numeric_features = []
    categorical_features = ["marital_status", "religion", "language", "insurance", "admission_type"]
    binary_features = []

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression())
    ])

    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    acc = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    pipeline_logger.info(f"Accuracy: {acc}")
    pipeline_logger.info(f"Precision: {precision}")
    pipeline_logger.info(f"Recall: {recall}")
    pipeline_logger.info(f"F1 Score: {f1}")

if __name__ == "__main__":
    pipeline()