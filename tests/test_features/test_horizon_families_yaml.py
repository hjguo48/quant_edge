from __future__ import annotations

from pathlib import Path

import yaml

from src.features import registry as registry_module


def _load_horizon_payload(monkeypatch):
    monkeypatch.setattr(registry_module.settings, "ENABLE_SHORTING_FEATURES", True)
    monkeypatch.setattr(registry_module.settings, "ENABLE_ANALYST_PROXY_FEATURES", True)

    registry = registry_module.FeatureRegistry()
    valid_families = {definition.category for definition in registry.list_features()}

    path = Path("configs/research/horizon_families.yaml")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload, registry, valid_families


def test_horizon_families_yaml_has_valid_schema(monkeypatch) -> None:
    payload, _, valid_families = _load_horizon_payload(monkeypatch)

    assert set(payload["horizons"]) == {"1d", "5d", "20d", "60d"}
    for horizon, config in payload["horizons"].items():
        assert isinstance(config["families"], list)
        assert isinstance(config["excluded_families"], list)
        assert isinstance(config["rationale"], str)
        assert config["families"]
        assert set(config["families"]).issubset(valid_families)
        assert set(config["excluded_families"]).issubset(valid_families)
        assert set(config["families"]).isdisjoint(config["excluded_families"])


def test_horizon_families_yaml_uses_registry_supported_horizons_for_annotated_families(monkeypatch) -> None:
    payload, registry, _ = _load_horizon_payload(monkeypatch)

    registered_by_family = {}
    for definition in registry.list_features():
        registered_by_family.setdefault(definition.category, set()).add(definition.name)

    family_horizon_metadata = {
        "shorting": registry_module._SHORTING_FEATURE_METADATA,
        "analyst_proxy": registry_module._ANALYST_PROXY_FEATURE_METADATA,
    }

    for horizon, config in payload["horizons"].items():
        for family in config["families"]:
            metadata = family_horizon_metadata.get(family)
            if metadata is None:
                continue
            assert family in registered_by_family
            supported = [
                feature_name
                for feature_name, feature_metadata in metadata.items()
                if feature_name in registered_by_family[family]
                and horizon in feature_metadata["horizon_applicability"]
            ]
            assert supported, f"{family} has no registry-backed feature annotated for horizon {horizon}"
