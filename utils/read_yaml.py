import yaml
from utils.logger import get_logger
from utils.directories import MODEL_SELECTION_YAML, COARSE_HYPERPARAMETER_YAML

from scipy.stats import uniform, randint

from collections import defaultdict

yaml_logger = get_logger(__name__)

def read_yaml(path):

    yaml_logger.info(f'Reading YAML file: {path}')
    with open(path, 'r') as f:
        config = yaml.safe_load(f)

    return config

def model_selection_yaml():
    return read_yaml(MODEL_SELECTION_YAML)

def hyper_parameter_yaml():
    return read_yaml(COARSE_HYPERPARAMETER_YAML)

def formatted_hyper_parameter_yaml():

    params = hyper_parameter_yaml()

    formatted_params = defaultdict(dict)

    for model_name, param in params.items():

        for param_name, spec in param.items():

            if spec["type"] == "uniform":
                functionality = uniform(spec["bounds"][0], spec["bounds"][1])

            elif spec["type"] == "randint":
                functionality = randint(spec["bounds"][0], spec["bounds"][1])

            elif spec["type"] == "choice":
                functionality = spec["values"]

            else:
                yaml_logger.info(f"Error. Spec Type: {spec["type"]}")
                yaml_logger.error("Cannot process hyperparameter that is not type uniform or randint")
                raise ValueError("Cannot process hyperparameter that is not type uniform or randint")

            formatted_params[model_name][f"{model_name}__{param_name}"] = functionality

    return formatted_params
