from utils.logger import get_ml_system_logger

model_selection_logger = get_ml_system_logger(__name__)

from utils.read_yaml import model_selection_yaml

CONFIG = model_selection_yaml()["thresholds"]
THRESHOLDS = [threshold for threshold in CONFIG]
BRIER_MAX = CONFIG["brier_score"]["max"]

def meets_thresholds(data: dict[str, float]) -> bool:

    for threshold in THRESHOLDS:
        if data[threshold] < CONFIG[threshold]["min"]:
            return False

    if data["brier_score"] > BRIER_MAX:
        return False

    return True

def top_k_models(trained_models: dict[tuple[str, int], dict], selection_config, num_models: int) -> list[str]:

    candidates = {}

    for (model_name, fold), data in trained_models.items():

        if meets_thresholds(data["metrics"]):
            candidates[(model_name, fold)] = data

    if not candidates:
        model_selection_logger.critical("No model passed minimum thresholds.")
        raise ValueError("No model passed minimum thresholds.")

    if len(candidates) < num_models:
        model_selection_logger.error(f"Not enough candidate models passed minimum thresholds. Expected: {num_models} Got: {len(candidates)}")
        raise ValueError(f"Not enough candidate models passed minimum thresholds. Expected: {num_models} Got: {len(candidates)}")

    top_k_performing_keys = sorted(candidates, key=lambda key: candidates[key]["metrics"]["recall"], reverse=True)[:num_models]
    top_k_performance_models = [key[0] for key in top_k_performing_keys]

    return top_k_performance_models