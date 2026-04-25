from __future__ import annotations

from pathlib import Path

import yaml

from src.features import registry as registry_module


def test_horizon_families_yaml_has_valid_schema(monkeypatch) -> None:
    monkeypatch.setattr(registry_module.settings, "ENABLE_SHORTING_FEATURES", True)
    monkeypatch.setattr(registry_module.settings, "ENABLE_ANALYST_PROXY_FEATURES", True)

    registry = registry_module.FeatureRegistry()
    valid_families = {definition.category for definition in registry.list_features()}

    path = Path("configs/research/horizon_families.yaml")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert set(payload["horizons"]) == {"1d", "5d", "20d", "60d"}
    for horizon, config in payload["horizons"].items():
        assert isinstance(config["families"], list)
        assert isinstance(config["excluded_families"], list)
        assert isinstance(config["rationale"], str)
        assert config["families"]
        assert set(config["families"]).issubset(valid_families)
        assert set(config["excluded_families"]).issubset(valid_families)
        assert set(config["families"]).isdisjoint(config["excluded_families"])
