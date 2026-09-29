import yaml
from utils.logger import get_logger
from utils.directories import MODEL_SELECTION_YAML

yaml_logger = get_logger(__name__)

def read_yaml(path):

    yaml_logger.info(f'Reading YAML file: {path}')
    with open(path, 'r') as f:
        config = yaml.safe_load(f)

    return config

def model_selection_yaml():
    return read_yaml(MODEL_SELECTION_YAML)