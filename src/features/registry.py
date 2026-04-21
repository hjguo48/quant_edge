from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from src.features.alternative import ALTERNATIVE_FEATURE_NAMES
from src.features.fundamental import FUNDAMENTAL_FEATURE_NAMES, compute_fundamental_features
from src.features.intraday import INTRADAY_FEATURE_NAMES, compute_intraday_features
from src.features.macro import MACRO_FEATURE_NAMES, compute_macro_features
from src.features.pipeline import COMPOSITE_FEATURE_NAMES, compute_composite_features
from src.features.pipeline import compute_alternative_features_batch
from src.features.sector_rotation import SECTOR_ROTATION_FEATURE_NAMES, compute_sector_rotation_features
from src.features.technical import TECHNICAL_FEATURE_NAMES, compute_technical_features
from src.features.trade_microstructure import (
    TRADE_MICROSTRUCTURE_FEATURE_NAMES,
    compute_trade_microstructure_features,
)


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
        for name, (category, description) in _ALTERNATIVE_FEATURE_METADATA.items():
            self.register(name, category, description, compute_alternative_features_batch)
        for name, description in _SECTOR_ROTATION_FEATURE_METADATA.items():
            self.register(name, "sector_rotation", description, compute_sector_rotation_features)
        for name, description in _COMPOSITE_FEATURE_METADATA.items():
            self.register(name, "composite", description, compute_composite_features)
        for name, description in _INTRADAY_FEATURE_METADATA.items():
            self.register(name, "intraday", description, compute_intraday_features)
        for name, description in _TRADE_MICROSTRUCTURE_FEATURE_METADATA.items():
            self.register(name, "trade_microstructure", description, compute_trade_microstructure_features)


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
    "residual_momentum": "Blitz, Huij & Martens (2011) 60-day sum of SPY market-model residual returns using a 252-day regression window.",
    "idio_vol": "Ang, Hodrick, Xing & Zhang (2006) 60-day standard deviation of SPY market-model residual returns using a 252-day regression window.",
    "stock_beta_252": "Rolling 252-day beta to SPY from the PIT-safe market model.",
    "residual_ret_5d": "Five-day return net of rolling SPY beta exposure.",
    "residual_ret_10d": "Ten-day return net of rolling SPY beta exposure.",
    "vol_scaled_reversal_5d": "Negative five-day return divided by annualized 20-day volatility.",
    "above_20dma": "Binary flag for close above the trailing 20-day moving average.",
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
    "accruals": "Sloan (1996) accrual anomaly: (net income - operating cash flow) divided by total assets.",
    "asset_growth": "Cooper, Gulen & Schill (2008) year-over-year total asset growth using t versus t-4 quarters.",
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

_ALTERNATIVE_FEATURE_METADATA = {
    "earnings_surprise_latest": (
        "earnings",
        "Most recent quarterly EPS surprise versus estimate using PIT-visible earnings history.",
    ),
    "earnings_surprise_avg_4q": (
        "earnings",
        "Average EPS surprise across the latest four PIT-visible quarters.",
    ),
    "earnings_beat_streak": (
        "earnings",
        "Consecutive quarterly EPS beats in the latest PIT-visible history.",
    ),
    "earnings_surprise_recency": (
        "earnings",
        "Latest EPS surprise decayed by days since the latest PIT-visible fiscal earnings date.",
    ),
    "earnings_beat_recency": (
        "earnings",
        "EPS beat streak decayed by days since the latest PIT-visible fiscal earnings date.",
    ),
    "earnings_surprise_recency_20d": (
        "earnings",
        "Latest EPS surprise decayed with a 20-day event half-life proxy.",
    ),
    "earnings_beat_recency_30d": (
        "earnings",
        "Beat streak decayed with a 30-day event half-life proxy.",
    ),
    "surprise_flip_qoq": (
        "earnings",
        "Change in latest EPS surprise versus the prior fiscal quarter surprise.",
    ),
    "surprise_vs_history": (
        "earnings",
        "Latest EPS surprise minus the recent four-quarter average surprise.",
    ),
    "pead_setup": (
        "earnings",
        "Post-earnings drift setup using decayed surprise and positive five-day momentum.",
    ),
    "eps_revision_direction": (
        "analyst",
        "Direction of the latest quarterly EPS consensus revision.",
    ),
    "revenue_revision_pct": (
        "analyst",
        "Percentage change in the latest quarterly revenue consensus versus the prior quarter target.",
    ),
    "analyst_coverage": (
        "analyst",
        "Number of analysts contributing to the current quarterly EPS consensus.",
    ),
    "short_interest_ratio": (
        "short_interest",
        "Latest days-to-cover or short-interest ratio from PIT-visible short-interest filings.",
    ),
    "short_interest_change": (
        "short_interest",
        "Change in short interest versus the prior reported settlement period.",
    ),
    "short_interest_sector_rel": (
        "short_interest",
        "Sector-relative z-score of the latest PIT-visible days-to-cover reading.",
    ),
    "short_interest_change_20d": (
        "short_interest",
        "Bi-weekly change in days-to-cover over the latest reporting interval.",
    ),
    "short_interest_abnormal_1y": (
        "short_interest",
        "Latest days-to-cover normalized by the stock's trailing one-year history.",
    ),
    "short_squeeze_setup": (
        "short_interest",
        "High short-interest names with positive short-term price and volume pressure.",
    ),
    "crowding_unwind_risk": (
        "short_interest",
        "High short-interest names in negative 20-day drawdowns.",
    ),
    "insider_net_buy_ratio": (
        "insider",
        "Net insider buy ratio over the trailing 90-day PIT-visible filing window.",
    ),
    "insider_buy_value": (
        "insider",
        "Total dollar value of insider purchases over the trailing 90-day PIT-visible filing window.",
    ),
    "insider_cluster_buy": (
        "insider",
        "Binary flag for three or more distinct insider buyers in the trailing 90-day PIT-visible window.",
    ),
    "insider_buy_intensity_20d": (
        "insider",
        "Role-weighted decayed insider purchases over 20 days scaled by market capitalization.",
    ),
    "insider_net_intensity_60d": (
        "insider",
        "Role-weighted decayed insider net buying over 60 days scaled by market capitalization.",
    ),
    "insider_cluster_buy_30d_w": (
        "insider",
        "Decayed weighted count of distinct insider buyers over the trailing 30 days.",
    ),
    "insider_abnormal_buy_90d": (
        "insider",
        "Current 90-day insider buy intensity relative to the stock's own historical baseline.",
    ),
    "insider_role_skew_30d": (
        "insider",
        "CEO/CFO buy intensity minus director/officer sell intensity over the trailing 30 days.",
    ),
    "days_since_last_8k": (
        "sec_filing",
        "Days since the latest PIT-visible 8-K filing.",
    ),
    "days_since_last_10q": (
        "sec_filing",
        "Days since the latest PIT-visible 10-Q filing.",
    ),
    "days_since_last_10k": (
        "sec_filing",
        "Days since the latest PIT-visible 10-K filing.",
    ),
    "recent_8k_count_5d": (
        "sec_filing",
        "Number of PIT-visible 8-K filings over the trailing 5 days.",
    ),
    "recent_8k_count_20d": (
        "sec_filing",
        "Number of PIT-visible 8-K filings over the trailing 20 days.",
    ),
    "recent_8k_count_60d": (
        "sec_filing",
        "Number of PIT-visible 8-K filings over the trailing 60 days.",
    ),
    "has_recent_8k_5d": (
        "sec_filing",
        "Binary indicator for at least one PIT-visible 8-K in the trailing 5 days.",
    ),
    "filing_burst_20d": (
        "sec_filing",
        "Count of PIT-visible SEC filings of all types over the trailing 20 days.",
    ),
    "overnight_gap": (
        "daily",
        "Current open versus previous close gap using PIT-visible daily bars.",
    ),
    "volume_surge": (
        "daily",
        "Current volume divided by the trailing 20-session average volume.",
    ),
}

_SECTOR_ROTATION_FEATURE_METADATA = {
    "sector_rel_ret_5d": "Sector ETF five-day return relative to SPY assigned by the stock's sector.",
    "sector_volume_surge": "Sector ETF volume divided by its trailing 20-day average volume.",
    "sector_pressure": "Rolling z-scored sector-relative return multiplied by sector ETF volume surge.",
    "stock_vs_sector_20d": "Stock 20-day return minus its mapped sector ETF 20-day return.",
    "sector_pressure_x_divergence": "Sector pressure interacted with the stock's 20-day divergence from its sector ETF.",
}

_INTRADAY_FEATURE_METADATA = {
    "gap_pct": "Open versus prior daily close return using the first regular-session minute bar.",
    "overnight_ret": "Alias for the regular-session opening gap versus the prior close.",
    "intraday_ret": "Regular-session close versus open return derived from minute aggregates.",
    "open_30m_ret": "First 30-minute regular-session return from the 09:30 open to the 09:59 close.",
    "last_30m_ret": "Last 30-minute regular-session return from the 15:30 open to the 15:59 close.",
    "realized_vol_1d": "Daily realized volatility from intraday minute log returns scaled by sqrt(390).",
    "volume_curve_surprise": "Average z-score of today's 30-minute volume buckets versus the prior 30 trading days.",
    "close_to_vwap": "15:59 regular-session close relative to the day VWAP synthesized from minute closes and volume.",
    "transactions_count_zscore": "Daily minute-aggregated transaction count z-score versus the prior 20 trading days.",
}

_TRADE_MICROSTRUCTURE_FEATURE_METADATA = {
    "trade_imbalance_proxy": "Lee-Ready tick-rule buy/sell imbalance over regular-session trades.",
    "large_trade_ratio": "Dollar volume share from trades above the configured block threshold.",
    "late_day_aggressiveness": "Late-day absolute imbalance ratio versus full regular-session imbalance.",
    "offhours_trade_ratio": "Pre/post-market volume share versus full analyzed-day volume.",
    "off_exchange_volume_ratio": "TRF/off-exchange dollar volume share using exchange, TRF id, or TRF timestamp.",
}

_COMPOSITE_FEATURE_METADATA = {
    "ret_vol_interaction_20d": "20-day return interacted with 20-day volume ratio.",
    "ret_vol_interaction_60d": "60-day return interacted with 20-day volume ratio.",
    "mom_vol_adj_20d": "20-day return adjusted by 20-day volatility.",
    "mom_vol_adj_60d": "60-day return adjusted by 60-day volatility.",
    "breadth_pct_above_20dma": "Stock above-20DMA participation relative to the cross-sectional breadth rate.",
    "return_dispersion_20d": "Stock 20-day relative return scaled by cross-sectional 20-day return dispersion.",
    "narrow_leadership_score": "Narrow-leadership context interacted with stock relative 20-day return.",
    "high_vix_x_beta": "Positive rolling VIX z-score interacted with stock 252-day beta.",
    "credit_widening_x_leverage": "Positive 20-day credit-spread widening interacted with debt-to-equity.",
    "curve_inverted_x_growth": "Negative yield-curve slope interacted with negative revenue growth.",
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

assert len(TECHNICAL_FEATURE_NAMES) == 37
assert len(FUNDAMENTAL_FEATURE_NAMES) == 17
assert len(MACRO_FEATURE_NAMES) == 10
assert len(ALTERNATIVE_FEATURE_NAMES) == 38
assert len(SECTOR_ROTATION_FEATURE_NAMES) == 5
assert len(INTRADAY_FEATURE_NAMES) == 9
assert len(TRADE_MICROSTRUCTURE_FEATURE_NAMES) == 5
assert len(COMPOSITE_FEATURE_NAMES) == 26
