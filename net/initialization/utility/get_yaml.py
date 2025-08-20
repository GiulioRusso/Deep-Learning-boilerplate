import os
import yaml


def get_yaml(yaml_path: str = "./configs/paths.yaml") -> dict:
    """
    Load the content of the specified YAML file.

    :param yaml_path: Path to the YAML file.
    :return: Dictionary with YAML file configurations.
    """
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"YAML file not found: '{yaml_path}'")

    try:
        with open(yaml_path, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Failed to parse YAML file '{yaml_path}': {e}")

    if not isinstance(data, dict):
        raise ValueError(f"YAML file '{yaml_path}' must contain a dictionary at the root level.")

    return data