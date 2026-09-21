from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils.logger import get_logger

evaluate_logger = get_logger(__name__)

def evaluate(y_test, y_preds):

    acc = accuracy_score(y_test, y_preds)
    precision = precision_score(y_test, y_preds)
    recall = recall_score(y_test, y_preds)
    f1 = f1_score(y_test, y_preds)

    evaluate_logger.info(f"Accuracy: {acc}")
    evaluate_logger.info(f"Precision: {precision}")
    evaluate_logger.info(f"Recall: {recall}")
    evaluate_logger.info(f"F1 Score: {f1}")

    return acc, precision, recall, f1