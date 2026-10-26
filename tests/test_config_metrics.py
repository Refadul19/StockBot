import datetime as dt

import pandas as pd
import pytest

from gapbot.config import Config
from gapbot.metrics import max_drawdown_pct, split_days


def test_config_roundtrip(tmp_path):
    c = Config().with_overrides({"exit.target_r": 3.0})
    p = tmp_path / "c.yaml"
    c.save(p)
    assert Config.load(p) == c


def test_unknown_keys_are_rejected():
    with pytest.raises(ValueError):
        Config.from_dict({"exit": {"taget_r": 2}})
    with pytest.raises(ValueError):
        Config().with_overrides({"exit.nope": 1})


def test_drawdown():
    assert max_drawdown_pct(pd.Series([100, 120, 90, 130])) == pytest.approx(-25.0)


def test_split_days_holds_out_the_end():
    days = [dt.date(2025, 1, d) for d in range(1, 11)]
    a, b = split_days(days, 0.3)
    assert a == days[:7] and b == days[7:]


def test_unquoted_yaml_times_are_understood():
    import yaml
    c = Config.from_dict(yaml.safe_load("entry:\n  no_entry_before: 10:00\nexit:\n  flatten_at: 15:50\n"))
    assert c.entry.no_entry_before == "10:00" and c.exit.flatten_at == "15:50"
