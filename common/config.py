from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"


def load_yaml(name: str) -> dict:
    path = CONFIGS_DIR / name
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_all():
    return {
        "town": load_yaml("town.yaml"),
        "scenarios": load_yaml("scenarios.yaml"),
        "bev": load_yaml("bev.yaml"),
    }
