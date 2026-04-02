from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from src.features.fundamental import FUNDAMENTAL_FEATURE_NAMES, compute_fundamental_features
from src.features.macro import MACRO_FEATURE_NAMES, compute_macro_features
from src.features.pipeline import COMPOSITE_FEATURE_NAMES, compute_composite_features
from src.features.technical import TECHNICAL_FEATURE_NAMES, compute_technical_features


@dataclass(frozen=True)
class FeatureDefinition:
    name: str
    category: str
    description: str
    compute_fn: Callable[..., pd.DataFrame]


class FeatureRegistry:
    def __init__(self) -> None:
        self._registry: dict[str, FeatureDefinition] = {}
        self._register_defaults()

    def register(
        self,
        name: str,
        category: str,
        description: str,
        compute_fn: Callable[..., pd.DataFrame],
    ) -> None:
        self._registry[name] = FeatureDefinition(
            name=name,
            category=category,
            description=description,
            compute_fn=compute_fn,
        )

    def get_feature(self, name: str) -> FeatureDefinition:
        try:
            return self._registry[name]
        except KeyError as exc:
            raise KeyError(f"Feature {name!r} is not registered.") from exc

    def list_features(self, category: str | None = None) -> list[FeatureDefinition]:
        definitions = list(self._registry.values())
        if category is not None:
            definitions = [definition for definition in definitions if definition.category == category]
        return sorted(definitions, key=lambda definition: (definition.category, definition.name))

    def _register_defaults(self) -> None:
        for name, description in _TECHNICAL_FEATURE_METADATA.items():
            self.register(name, "technical", description, compute_technical_features)
        for name, description in _FUNDAMENTAL_FEATURE_METADATA.items():
            self.register(name, "fundamental", description, compute_fundamental_features)
        for name, description in _MACRO_FEATURE_METADATA.items():
            self.register(name, "macro", description, compute_macro_features)
        for name, description in _COMPOSITE_FEATURE_METADATA.items():
            self.register(name, "composite", description, compute_composite_features)


_TECHNICAL_FEATURE_METADATA = {
    "ret_5d": "Five-day trailing return based on adjusted close.",
    "ret_10d": "Ten-day trailing return based on adjusted close.",
    "ret_20d": "Twenty-day trailing return based on adjusted close.",
    "ret_60d": "Sixty-day trailing return based on adjusted close.",
    "high_52w_ratio": "Close divided by the trailing 52-week high.",
    "low_52w_ratio": "Close divided by the trailing 52-week low.",
    "momentum_rank_20d": "Cross-sectional percentile rank of 20-day momentum.",
    "momentum_rank_60d": "Cross-sectional percentile rank of 60-day momentum.",
    "vol_20d": "Annualized 20-day realized volatility.",
    "vol_60d": "Annualized 60-day realized volatility.",
    "atr_14": "14-day average true range divided by close.",
    "gk_vol": "20-day Garman-Klass volatility estimator.",
    "vol_rank": "Cross-sectional percentile rank of 20-day volatility.",
    "vol_change": "Ratio of 20-day to 60-day realized volatility.",
    "volume_ratio_5d": "Five-day average volume divided by 20-day average volume.",
    "volume_ratio_20d": "Twenty-day average volume divided by 60-day average volume.",
    "obv_slope": "Normalized 20-day rolling slope of on-balance volume.",
    "vwap_deviation": "Close minus approximate daily VWAP divided by VWAP.",
    "amihud": "20-day average Amihud illiquidity measure.",
    "turnover_rate": "Daily volume divided by shares outstanding.",
    "rsi_14": "14-day relative strength index.",
    "rsi_28": "28-day relative strength index.",
    "macd_signal": "MACD signal line using 12/26/9 parameters.",
    "macd_histogram": "MACD histogram using 12/26/9 parameters.",
    "bb_width": "20-day Bollinger Band width using two standard deviations.",
    "bb_position": "Relative close position inside 20-day Bollinger Bands.",
    "adx_14": "14-day average directional index.",
    "stoch_k": "14-day stochastic oscillator %K.",
    "stoch_d": "Three-day moving average of stochastic %K.",
    "cci_20": "20-day commodity channel index.",
}

_FUNDAMENTAL_FEATURE_METADATA = {
    "pe_ratio": "Price divided by trailing-twelve-month EPS.",
    "pb_ratio": "Price divided by book value per share.",
    "ps_ratio": "Price divided by trailing-twelve-month revenue per share.",
    "ev_ebitda": "Enterprise value divided by trailing EBITDA.",
    "fcf_yield": "Free-cash-flow yield on market capitalization.",
    "dividend_yield": "Annualized dividend per share divided by price.",
    "roe": "Net income divided by shareholder equity.",
    "roa": "Net income divided by total assets.",
    "gross_margin": "Gross profit divided by revenue.",
    "operating_margin": "Operating income divided by revenue.",
    "revenue_growth_yoy": "Year-over-year quarterly revenue growth.",
    "earnings_growth_yoy": "Year-over-year quarterly net income growth.",
    "debt_to_equity": "Total debt divided by shareholder equity.",
    "current_ratio": "Current assets divided by current liabilities.",
    "eps_surprise": "Reported EPS surprise versus consensus EPS.",
}

_MACRO_FEATURE_METADATA = {
    "vix": "MACRO_REGIME: latest VIX level as of the PIT date.",
    "vix_change_5d": "MACRO_REGIME: five-day percentage change in VIX.",
    "vix_rank": "MACRO_REGIME: percentile rank of VIX over the trailing 252-day history.",
    "yield_10y": "MACRO_REGIME: latest 10-year Treasury yield.",
    "yield_spread_10y2y": "MACRO_REGIME: 10-year yield minus 2-year yield or fallback short-rate proxy.",
    "credit_spread": "MACRO_REGIME: latest BAA10Y minus AAA10Y credit spread level.",
    "credit_spread_change": "MACRO_REGIME: 20-day change in credit spread.",
    "ffr": "MACRO_REGIME: latest federal funds rate.",
    "sp500_breadth": "MACRO_REGIME: share of S&P 500 members with positive trailing 20-day returns.",
    "market_ret_20d": "MACRO_REGIME: trailing 20-day return of SPY.",
}

_COMPOSITE_FEATURE_METADATA = {
    "ret_vol_interaction_20d": "20-day return interacted with 20-day volume ratio.",
    "ret_vol_interaction_60d": "60-day return interacted with 20-day volume ratio.",
    "mom_vol_adj_20d": "20-day return adjusted by 20-day volatility.",
    "mom_vol_adj_60d": "60-day return adjusted by 60-day volatility.",
    "value_mom_pe": "Momentum combined with P/E valuation.",
    "value_mom_pb": "Momentum combined with P/B valuation.",
    "quality_value_roe_pb": "ROE adjusted by P/B valuation.",
    "quality_value_roa_pe": "ROA adjusted by P/E valuation.",
    "fcf_mom_20d": "Free-cash-flow yield interacted with 20-day momentum.",
    "leverage_vol_20d": "Debt-to-equity interacted with 20-day volatility.",
    "valuation_spread_pb_pe": "Relative spread between P/B and P/E signals.",
    "margin_quality_combo": "Combined gross and operating margin quality signal.",
    "profitability_combo": "Combined ROE and ROA profitability signal.",
    "liquidity_momentum": "Liquidity ratio interacted with 20-day momentum.",
    "mean_reversion_combo": "Bollinger positioning minus RSI-based overbought signal.",
    "trend_confirmation": "MACD histogram combined with stochastic confirmation.",
    "risk_sentiment": "MACRO_REGIME composite: market return adjusted by VIX change.",
    "spread_stress": "MACRO_REGIME composite: credit spread relative to curve slope stress.",
    "breadth_momentum": "MACRO_REGIME composite: breadth and market momentum composite.",
    "macro_risk_on": "MACRO_REGIME composite: curve slope minus credit spread risk-on composite.",
}

assert len(TECHNICAL_FEATURE_NAMES) == 30
assert len(FUNDAMENTAL_FEATURE_NAMES) == 15
assert len(MACRO_FEATURE_NAMES) == 10
assert len(COMPOSITE_FEATURE_NAMES) == 20
