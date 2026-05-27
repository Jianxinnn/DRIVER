import os
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CONFIG = {
    "cache_dir": "./output/cache",
    "pdb_path": "./output/pdb_predict/",
    "msas_path": "./output/msa_queries",
    "max_workers": 4,
}


def configure_runtime_environment():
    endpoint = os.environ.get("DRIVER_HF_ENDPOINT")
    if endpoint and "HF_ENDPOINT" not in os.environ:
        os.environ["HF_ENDPOINT"] = endpoint

    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("USE_FLAX", "0")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def repo_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def load_runtime_config(config_path=None):
    path = Path(config_path or os.environ.get("DRIVER_CONFIG") or REPO_ROOT / "config.yaml")
    if not path.is_absolute():
        path = REPO_ROOT / path

    config = DEFAULT_CONFIG.copy()
    if path.exists():
        with path.open("r") as handle:
            loaded = yaml.safe_load(handle) or {}
        config.update(loaded)

    for key in ("cache_dir", "pdb_path", "msas_path"):
        config[key] = str(repo_path(config[key]))
    config["max_workers"] = int(config.get("max_workers", DEFAULT_CONFIG["max_workers"]))
    return config
