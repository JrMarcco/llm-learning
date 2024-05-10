import yaml


class ConfLoader:
    def __init__(self, path):
        self.conf_path = path

    def load_config(self):
        with open(self.conf_path, "r") as f:
            config = yaml.safe_load(f)
        return config

