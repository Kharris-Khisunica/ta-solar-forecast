import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from utils import load_config, set_seed  # noqa: E402


def test_config_has_required_keys():
    cfg = load_config()
    assert cfg["data"]["split"]["train"] == [2016, 2017, 2018]
    assert cfg["forecast"]["issue_time_utc"] == "05:00"
    assert -90 <= cfg["location"]["lat"] <= 90 and -180 <= cfg["location"]["lon"] <= 180


def test_set_seed_is_reproducible():
    set_seed(42); a = (random.random(), np.random.rand())
    set_seed(42); b = (random.random(), np.random.rand())
    assert a == b
