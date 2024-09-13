import yaml
from pathlib import Path


class ObjDict(dict):
    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)


def load_config(file_name: str, config_name: str):
    """
    Load the configuration file.
    This function does not accept any parameters.
    Returns:
        config: An ObjDict object containing the configuration information loaded from the configuration file.
    """
    # Get the directory of the current file
    parent_dir = Path(__file__).resolve().parent
    # Safely load the contents of the configuration file
    config_file = yaml.safe_load((parent_dir / file_name).read_text())
    # Map the "smarts" section of the configuration file to an ObjDict object
    config = ObjDict(config_file[config_name])
    return config