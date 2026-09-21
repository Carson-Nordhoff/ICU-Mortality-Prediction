from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score, roc_auc_score
from utils.logger import get_logger

evaluate_logger = get_logger(__name__)

def evaluate(y_test, y_preds, y_probas):

    acc = accuracy_score(y_test, y_preds)
    precision = precision_score(y_test, y_preds)
    recall = recall_score(y_test, y_preds)
    f1 = f1_score(y_test, y_preds)
    avg_precision = average_precision_score(y_test, y_probas)
    roc_auc = roc_auc_score(y_test, y_probas)

    evaluate_logger.info(
        f"Accuracy: {acc:.3f} | Precision: {precision:.3f} | Recall: {recall:.3f} | F1 Score: {f1:.3f} | Average Precision: {avg_precision:.3f} | ROC AUC: {roc_auc:.3f} |"
    )

    return acc, precision, recall, f1, avg_precision, roc_auc