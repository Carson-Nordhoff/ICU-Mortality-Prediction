import json
import matplotlib.pyplot as plt

from utils.directories import METRICS_DIR
from utils.logger import get_logger

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    brier_score_loss,
)
from sklearn.calibration import CalibrationDisplay, calibration_curve

import os

evaluate_logger = get_logger(__name__)

def plot_calibration(y_test: list[int], y_probas: list[float]):

    evaluate_logger.info("Calibration Diagram (Calibration Curve)")
    fig, ax = plt.subplots(figsize=(6, 6))
    display = CalibrationDisplay.from_predictions(
        y_test, y_probas, n_bins=10, name="Logistic Regression", ax=ax
    )

    plt.title("Reliability Diagram (Calibration Curve)")
    plt.show()

type SevenFloats = tuple[
    float,
    float,
    float,
    float,
    float,
    float,
    float
]

def evaluate(y_test: list[int], y_preds: list[int], y_probas: list[float], step: str, model: str, fold: int | None) -> SevenFloats:

    acc = accuracy_score(y_test, y_preds)
    precision = precision_score(y_test, y_preds)
    recall = recall_score(y_test, y_preds)
    f1 = f1_score(y_test, y_preds)
    avg_precision = average_precision_score(y_test, y_probas)
    roc_auc = roc_auc_score(y_test, y_probas)
    brier_score = brier_score_loss(y_test, y_probas)

    #plot_calibration(y_test, y_probas)

    metrics = {
        'fold': fold,
        'acc': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'avg_precision': avg_precision,
        'roc_auc': roc_auc,
        'brier_score': brier_score
    }

    evaluate_logger.info(
        f"Accuracy: {acc:.3f} | "
        f"Precision: {precision:.3f} | "
        f"Recall: {recall:.3f} | "
        f"F1 Score: {f1:.3f} | "
        f"Average Precision: {avg_precision:.3f} | "
        f"ROC AUC: {roc_auc:.3f} | "
        f"Brier Score: {brier_score:.3f} |"
    )

    file_name = ""
    match step:

        case "default":
            file_name = f'{step}_{model}_{fold}_metrics.json'

        case "coarse_hypertune":
            file_name = f'{step}_{model}_metrics.json'

        case "fine_hypertune":
            file_name = f'{step}_{model}_metrics.json'

        case "best_model":
            file_name = f'{step}_{model}_metrics.json'

        case _:
            evaluate_logger.error(f"Cannot save metrics as step is not recognized. Step: {step}")
            raise ValueError(f"Cannot save metrics as step is not recognized. Step: {step}")

    file_path = os.path.join(METRICS_DIR, file_name)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    return acc, precision, recall, f1, avg_precision, roc_auc, brier_score