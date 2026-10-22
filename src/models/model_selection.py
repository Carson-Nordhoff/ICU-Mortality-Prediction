from utils.read_yaml import model_selection_yaml
from utils.logger import get_logger

model_selection_logger = get_logger(__name__)

selection_config = model_selection_yaml()
num_models = selection_config["num_models_hyperparameter_tune"]

def top_k_models(trained_models: dict[tuple[str, int], dict]) -> list[str]:

    candidates = {}

    model_selection_logger.info("-----GET TOP K MODELS-----")

    for (model_name, fold), data in trained_models.items():

        model_selection_logger.info(model_name)

        if data["recall"] >= selection_config["thresholds"]["recall"]["min"]:
            candidates[(model_name, fold)] = data

    if not candidates:
        model_selection_logger.critical("No model passed minimum thresholds.")
        raise ValueError("No model passed minimum thresholds.")

    if len(candidates) < num_models:
        model_selection_logger.error(f"Not enough candidate models passed minimum thresholds. Expected: {num_models} Got: {len(candidates)}")
        raise ValueError(f"Not enough candidate models passed minimum thresholds. Expected: {num_models} Got: {len(candidates)}")

    top_k_performing_keys = sorted(candidates, key=lambda key: candidates[key]["recall"], reverse=True)[:num_models]
    top_k_performance_models = [key[0] for key in top_k_performing_keys]

    return top_k_performance_models