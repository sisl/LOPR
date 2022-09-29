import os
import yaml

def load_config(config_name):
    CONFIG_PATH = ""
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)
    return config
