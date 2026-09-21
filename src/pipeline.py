#MODEL TO-DO#

#Add GroupShuffleSplit (prevent patient/entity level leakage)
#Handle mortality class imbalance
#Adjust system to multiple model analysis for optimal model

from src.data import clean_data
from src.data.load_data import load_data, inspect_data, insert_raw_data, load_clean_mimic_data
from src.data.clean_data import clean_data
from src.models.train_model import get_preprocessor
from src.models.evaluate import evaluate
from utils.directories import ensure_directories
from utils.logger import get_logger

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

pipeline_logger = get_logger(__name__)

numeric_features = ["age"]
categorical_features = ["marital_status", "religion", "language", "insurance", "admission_type", "first_careunit"]
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

    X = df.drop(columns=['hospital_expire_flag'])
    y = df['hospital_expire_flag']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = get_preprocessor(numeric_features, categorical_features)

    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression())
    ])

    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    evaluate(y_test, preds)

if __name__ == "__main__":
    pipeline()