"""Load dataset configurations from YAML."""
import os
import yaml

_script_dir = os.path.dirname(os.path.abspath(__file__))
_CONFIG_PATH = os.path.join(_script_dir, "datasets.yaml")

_config_cache = {}


def load_datasets_config(config_path=None):
    """Load the full datasets YAML config. Cached per path after first load."""
    global _config_cache
    path = os.path.abspath(config_path) if config_path else _CONFIG_PATH
    if path not in _config_cache:
        with open(path, "r") as f:
            _config_cache[path] = yaml.safe_load(f)
    return _config_cache[path]


def get_dataset_config(dataset_name, config_path=None):
    """
    Get config for a dataset. Returns dict with all keys from YAML.
    For datasets that need runtime input_dims (cremad, urfunny), caller should
    merge those in after loading the dataset.
    """
    configs = load_datasets_config(config_path)
    if dataset_name not in configs:
        raise KeyError(f"Unknown dataset: {dataset_name}. Available: {list(configs.keys())}")
    cfg = dict(configs[dataset_name])
    cfg["dataset_name"] = dataset_name
    return cfg


def get_checkpoint_path(dataset_name, seed, config_path=None):
    """Build checkpoint path from config pattern."""
    cfg = get_dataset_config(dataset_name, config_path)
    pattern = cfg.get("checkpoint_pattern")
    if not pattern:
        return cfg.get("checkpoint_path", "")
    rank = cfg.get("rank") or cfg.get("shared_rank") or cfg.get("specific_rank") or 10
    return pattern.format(
        ckpts_dir="./ckpts",
        dataset=dataset_name,
        lr=cfg.get("lr", 0.001),
        bs=cfg.get("batch_size", 256),
        rank=rank,
        prune=cfg.get("prune", 0.1),
        seed=seed,
    )
