import yaml
from utils import config_updater


class Config:
    def __init__(self):
        with open("config_file.yaml", "r") as yaml_file:
            config_dct = yaml.safe_load(yaml_file)
        for key, value in config_dct.items():
            setattr(self, key, value)

        config_updater(self)
