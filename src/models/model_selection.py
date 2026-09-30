from utils.read_yaml import model_selection_yaml
from utils.logger import get_logger

model_selection_logger = get_logger(__name__)

selection_config = model_selection_yaml()

def best_model_from_folds(models):

    candidates = {}

    for fold, model in models.items():

        if model["recall"] >= selection_config["thresholds"]["recall"]["min"]:
            candidates[fold] = model

    if not candidates:
        model_selection_logger.info("No model passed minimum thresholds.")
        raise ValueError("No model passed minimum thresholds.")

    max_recall_fold = max(models, key=lambda k: models[k]["recall"])
    max_recall_model = candidates[max_recall_fold]

    model_selection_logger.info(f"----------BEST MODEL----------")
    model_selection_logger.info(f"Best model on fold: {max_recall_fold}")
    model_selection_logger.info(
        f"Accuracy: {max_recall_model["acc"]:.3f} | "
        f"Precision: {max_recall_model["precision"]:.3f} | "
        f"Recall: {max_recall_model["recall"]:.3f} | "
        f"F1 Score: {max_recall_model["f1"]:.3f} | "
        f"Average Precision: {max_recall_model["avg_precision"]:.3f} | "
        f"ROC AUC: {max_recall_model["roc_auc"]:.3f} | "
        f"Brier Score: {max_recall_model["brier_score"]:.3f} |"
    )

    return max_recall_model