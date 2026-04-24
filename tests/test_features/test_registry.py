from __future__ import annotations

from src.features import registry as registry_module

SHORTING_FEATURES = {
    "short_sale_ratio_1d",
    "short_sale_ratio_5d",
    "short_sale_accel",
    "abnormal_off_exchange_shorting",
}

ANALYST_PROXY_FEATURES = {
    "net_grade_change_5d",
    "net_grade_change_20d",
    "net_grade_change_60d",
    "upgrade_count",
    "downgrade_count",
    "consensus_upside",
    "target_price_drift",
    "target_dispersion_proxy",
    "coverage_change_proxy",
    "financial_health_trend",
}


def _names(registry: registry_module.FeatureRegistry, category: str | None = None) -> set[str]:
    return {definition.name for definition in registry.list_features(category)}


def test_shorting_registry_default_off(monkeypatch) -> None:
    monkeypatch.setattr(registry_module.settings, "ENABLE_SHORTING_FEATURES", False)
    monkeypatch.setattr(registry_module.settings, "ENABLE_ANALYST_PROXY_FEATURES", False)

    registry = registry_module.FeatureRegistry()

    assert SHORTING_FEATURES.isdisjoint(_names(registry))
    assert _names(registry, "shorting") == set()


def test_analyst_proxy_registry_default_off(monkeypatch) -> None:
    monkeypatch.setattr(registry_module.settings, "ENABLE_SHORTING_FEATURES", False)
    monkeypatch.setattr(registry_module.settings, "ENABLE_ANALYST_PROXY_FEATURES", False)

    registry = registry_module.FeatureRegistry()

    assert ANALYST_PROXY_FEATURES.isdisjoint(_names(registry))
    assert _names(registry, "analyst_proxy") == set()


def test_feature_registry_count_updated_when_week5_flags_enabled(monkeypatch) -> None:
    monkeypatch.setattr(registry_module.settings, "ENABLE_SHORTING_FEATURES", False)
    monkeypatch.setattr(registry_module.settings, "ENABLE_ANALYST_PROXY_FEATURES", False)
    base_registry = registry_module.FeatureRegistry()
    base_count = len(base_registry)

    monkeypatch.setattr(registry_module.settings, "ENABLE_SHORTING_FEATURES", True)
    monkeypatch.setattr(registry_module.settings, "ENABLE_ANALYST_PROXY_FEATURES", True)
    enabled_registry = registry_module.FeatureRegistry()

    assert len(enabled_registry) == base_count + 14


def test_enable_flags_flip_shorting_only(monkeypatch) -> None:
    monkeypatch.setattr(registry_module.settings, "ENABLE_SHORTING_FEATURES", True)
    monkeypatch.setattr(registry_module.settings, "ENABLE_ANALYST_PROXY_FEATURES", False)

    registry = registry_module.FeatureRegistry()
    names = _names(registry)

    assert SHORTING_FEATURES.issubset(names)
    assert ANALYST_PROXY_FEATURES.isdisjoint(names)


def test_compute_fn_callable_for_all_week5_features(monkeypatch) -> None:
    monkeypatch.setattr(registry_module.settings, "ENABLE_SHORTING_FEATURES", True)
    monkeypatch.setattr(registry_module.settings, "ENABLE_ANALYST_PROXY_FEATURES", True)

    registry = registry_module.FeatureRegistry()

    for feature_name in SHORTING_FEATURES | ANALYST_PROXY_FEATURES:
        feature = registry.get_feature(feature_name)
        assert callable(feature.compute_fn)
