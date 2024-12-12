from pathlib import Path
import re
import torch
import yaml
from nn import BaseModel

class Model:
    def __init__(self, phi: str, num_classes: int):
        dep_mul, wid_mul, deep_mul = yaml_model_load('./config.yaml', phi)
        base_channels = int(wid_mul * 64)  # 64 default width
        base_depth = max(round(dep_mul * 3), 1)  # 3 default depth
        self.model = BaseModel(num_classes, base_channels, base_depth, deep_mul)


def yaml_model_load(path, phi):
    """Load a model from a YAML file."""
    path = Path(path)
    d = yaml_load(path) 
    return d['scales'][phi]

def yaml_load(file="data.yaml", append_filename=False):
    """
    Load YAML data from a file.

    Args:
        file (str, optional): File name. Default is 'data.yaml'.
        append_filename (bool): Add the YAML filename to the YAML dictionary. Default is False.

    Returns:
        (dict): YAML data and file name.
    """
    assert Path(file).suffix in {".yaml", ".yml"}, f"Attempting to load non-YAML file {file} with yaml_load()"
    with open(file, errors="ignore", encoding="utf-8") as f:
        s = f.read()  # string
        if not s.isprintable():
            s = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]+", "", s)

        # Add YAML filename to dict and return
        data = yaml.safe_load(s) or {}  # always return a dict (yaml.safe_load() may return None for empty files)
        if append_filename:
            data["yaml_file"] = str(file)
        return data
