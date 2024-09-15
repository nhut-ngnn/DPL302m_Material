import os
import yaml

CONFIG_PATH = 'utils/config.yml'

def original_validation_dir(config_path=CONFIG_PATH):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config['original_validation_dir']

def validation_dir(config_path=CONFIG_PATH):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config['validation_dir']

def test_dir(config_path=CONFIG_PATH):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config['test_dir']

def train_dir(config_path=CONFIG_PATH):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config['train_dir']

def model_dir(config_path=CONFIG_PATH):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config['models_path']