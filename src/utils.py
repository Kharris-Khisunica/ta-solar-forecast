"""Utilitas bersama: path repo, konfigurasi, dan seed global (T02.4–T02.5)."""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "config.yaml"


def load_config(path: Path | str = CONFIG_PATH) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def path(key: str, cfg: dict | None = None) -> Path:
    """Path absolut dari configs/config.yaml -> paths.<key>; folder dibuat bila belum ada."""
    cfg = cfg or load_config()
    p = ROOT / cfg["paths"][key]
    p.mkdir(parents=True, exist_ok=True)
    return p


def set_seed(seed: int | None = None) -> int:
    """Seed untuk random, numpy, dan tensorflow (bila terpasang). Panggil di awal setiap skrip."""
    if seed is None:
        seed = int(load_config()["seed"])
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        tf.keras.utils.set_random_seed(seed)
    except ImportError:
        pass
    return seed
