from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime
from decimal import Decimal
import ast
import json
from pathlib import Path
from typing import Any

import yaml
from sqlalchemy import create_engine

from src.config import settings


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_DATE_TAG = "20260425"

FEATURE_SOURCES = (
    ("technical", "src/features/technical.py", "TECHNICAL_FEATURE_NAMES"),
    ("fundamental", "src/features/fundamental.py", "FUNDAMENTAL_FEATURE_NAMES"),
    ("macro", "src/features/macro.py", "MACRO_FEATURE_NAMES"),
    ("alternative", "src/features/alternative.py", "ALTERNATIVE_FEATURE_NAMES"),
    ("sector_rotation", "src/features/sector_rotation.py", "SECTOR_ROTATION_FEATURE_NAMES"),
    ("composite", "src/features/pipeline.py", "COMPOSITE_FEATURE_NAMES"),
)


def get_engine():
    return create_engine(settings.database_url)


def load_yaml(rel_path: str) -> dict[str, Any]:
    with (REPO_ROOT / rel_path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_current_state() -> dict[str, Any]:
    return load_yaml("configs/research/current_state.yaml")


def load_family_registry() -> dict[str, Any]:
    return load_yaml("configs/research/family_registry.yaml")


def extract_feature_sets() -> dict[str, set[str]]:
    feature_sets: dict[str, set[str]] = {}
    for domain, rel_path, var_name in FEATURE_SOURCES:
        source = (REPO_ROOT / rel_path).read_text(encoding="utf-8")
        tree = ast.parse(source, filename=rel_path)
        found = None
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == var_name:
                        found = set(ast.literal_eval(node.value))
                        break
            if found is not None:
                break
        if found is None:
            raise RuntimeError(f"Could not locate {var_name} in {rel_path}")
        feature_sets[domain] = found
    return feature_sets


def get_feature_domain(feature_name: str, feature_sets: dict[str, set[str]]) -> str | None:
    base_name = feature_name[len("is_missing_") :] if feature_name.startswith("is_missing_") else feature_name
    for domain, names in feature_sets.items():
        if base_name in names:
            return domain
    return None


def infer_feature_contract(
    feature_name: str,
    feature_sets: dict[str, set[str]],
    family_map: dict[str, str],
) -> dict[str, Any]:
    base_name = feature_name[len("is_missing_") :] if feature_name.startswith("is_missing_") else feature_name
    family = family_map.get(feature_name) or family_map.get(base_name) or infer_family_from_name(base_name)
    domain = get_feature_domain(feature_name, feature_sets)
    if domain == "technical":
        source_tables = ["stock_prices"]
        if base_name in {"turnover_rate"}:
            source_tables.append("fundamentals_pit")
        lag_rule = "价格类特征依赖日频 OHLCV；当前实现按 stock_prices 的 T+1 knowledge_time 可见。"
        pit_rule = "只使用 knowledge_time <= as_of 的价格行。"
    elif domain == "fundamental":
        source_tables = ["fundamentals_pit", "stock_prices"]
        lag_rule = "财报类特征在 filing/knowledge_time 生效后可见，价格分母按 as_of 的 PIT 价格对齐。"
        pit_rule = "fundamentals_pit.event_time <= trade_date 且 knowledge_time <= as_of。"
    elif domain == "macro":
        source_tables = ["macro_series_pit"]
        if base_name in {"sp500_breadth", "market_ret_20d"}:
            source_tables.append("stock_prices")
        lag_rule = "宏观序列遵循各自发布时间；以 knowledge_time <= as_of 为准。"
        pit_rule = "只取最新 knowledge_time <= as_of 的宏观观测。"
    elif domain == "alternative":
        if base_name.startswith(("earnings_", "surprise_", "pead_")):
            source_tables = ["earnings_estimates", "stock_prices"]
            lag_rule = "财报事件按 earnings_estimates.knowledge_time 生效，价格交互项随价格 T+1 更新。"
            pit_rule = "earnings_estimates.knowledge_time <= as_of。"
        elif base_name in {"eps_revision_direction", "revenue_revision_pct", "analyst_coverage"}:
            source_tables = ["analyst_estimates"]
            lag_rule = "分析师代理特征按 analyst_estimates.knowledge_time 生效。"
            pit_rule = "analyst_estimates.knowledge_time <= as_of。"
        elif base_name.startswith(("short_interest_", "short_squeeze_", "crowding_unwind_")):
            source_tables = ["short_interest", "stock_prices", "stocks"]
            lag_rule = "短仓数据遵循 settlement_date + 2 个工作日左右的可见时点，价格交互项为 T+1。"
            pit_rule = "short_interest.knowledge_time <= as_of。"
        elif base_name.startswith("insider_"):
            source_tables = ["insider_trades", "fundamentals_pit", "stock_prices"]
            lag_rule = "内部人特征在 filing_date / accepted filing 可见后生效。"
            pit_rule = "insider_trades.knowledge_time <= as_of。"
        elif base_name.startswith(("days_since_last_", "recent_8k_count_", "has_recent_8k_", "filing_")):
            source_tables = ["sec_filings"]
            lag_rule = "SEC filing metadata 在 accepted_date 对应的 knowledge_time 后可见。"
            pit_rule = "sec_filings.knowledge_time <= as_of。"
        else:
            source_tables = ["stock_prices"]
            lag_rule = "价格派生事件特征随价格 T+1 更新。"
            pit_rule = "stock_prices.knowledge_time <= as_of。"
    elif domain == "sector_rotation":
        source_tables = ["stock_prices", "stocks"]
        lag_rule = "板块 ETF 轮动特征依赖 ETF/股票日线；按价格 T+1 knowledge_time 生效。"
        pit_rule = "只使用 stock_prices 中 knowledge_time <= as_of 的 ETF 与个股价格。"
    elif domain == "composite":
        source_tables = ["derived_from_base_features"]
        lag_rule = "组合特征继承底层输入中最慢的 lag。"
        pit_rule = "所有底层输入必须先满足各自 knowledge_time <= as_of。"
    else:
        source_tables = ["unknown"]
        lag_rule = "未在冻结研究协议中注册；需人工确认。"
        pit_rule = "未注册特征，无法自动确认。"

    return {
        "family": family,
        "domain": domain or "unregistered_live_only",
        "source_tables": source_tables,
        "lag_rule": lag_rule,
        "pit_rule": pit_rule,
    }


def infer_family_from_name(feature_name: str) -> str:
    name = feature_name[len("is_missing_") :] if feature_name.startswith("is_missing_") else feature_name
    if name.endswith("_sector_rel") or name.startswith(("sector_rel_", "sector_volume_", "sector_pressure", "stock_vs_sector")):
        return "sector_network_diffusion"
    if name.startswith(("vix", "yield_", "credit_", "ffr")):
        return "macro_regime"
    if name.startswith(("short_interest_", "short_squeeze_", "crowding_unwind_", "insider_")):
        return "shorting_crowding"
    if name.startswith(("earnings_", "surprise_", "pead_", "days_since_last_", "recent_8k_count_", "has_recent_8k_", "filing_")):
        return "event_earnings_sec"
    if name.startswith(("eps_revision_", "revenue_revision_", "consensus_")) or name == "analyst_coverage":
        return "analyst_expectations_proxy"
    if name.startswith(("pe_", "pb_", "ps_", "ev_", "fcf_")) or name in {
        "dividend_yield",
        "roe",
        "roa",
        "gross_margin",
        "operating_margin",
        "revenue_growth_yoy",
        "earnings_growth_yoy",
        "debt_to_equity",
        "current_ratio",
        "eps_surprise",
        "accruals",
        "asset_growth",
    }:
        return "fundamental_quality"
    if name.startswith(("amihud", "turnover_rate", "volume_", "vwap_deviation", "residual_", "idio_vol", "overnight_gap")):
        return "liquidity_microstructure"
    return "price_momentum"


def json_default(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return str(value)


def write_json_report(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=json_default)


def summarize_issues(issues: list[dict[str, Any]]) -> tuple[int, int]:
    critical = sum(1 for issue in issues if issue.get("severity") == "critical")
    warnings = sum(1 for issue in issues if issue.get("severity") == "warning")
    return critical, warnings

