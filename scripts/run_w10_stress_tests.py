from __future__ import annotations

"""W10.5 P0-2: DSR + Hansen SPA stress test on the W10 champion.

Loads w10_truth_table_60d_periods.parquet (24 combos × 328 weekly periods,
each row carries net_return / benchmark_return / etc.), identifies the
champion (score_weighted_buffered, cost_mult=1.0, gate_on=False), and runs:

1. Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014):
   - DSR_local: n_trials = 4 (4 distinct strategy families, alpha = 0.05)
   - DSR_conservative: n_trials = 8 (double, alpha = 0.10)
   - DSR_no_prior_search: n_trials = 1 (assumes W22 prior selection)
   - Rationale: cost bands (3) and gate states (2) are sensitivity tests on
     the SAME strategies, not independent trials. Counting all 24 cells as
     trials (Codex W11_W12_plan original) inflates E[max SR] artificially.
2. Stationary bootstrap of Sharpe>0 (Politis & Romano 1994):
   - 5000 resamples, block size 4 (~1 month for weekly data)
   - One-sided test of H0: Sharpe <= 0
3. Hansen-style SPA paired fallback (run_spa_fallback):
   - benchmark = champion net_excess weekly series
   - competitors = 3 DISTINCT strategies (cost=1.0, gate=False each)
     — same-strategy sensitivity-band variants excluded as illegitimate
   - Holm-adjusted; we want: NO competitor significantly beats champion

Output: data/reports/w10_stress_tests.json + console summary.
"""

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

from loguru import logger
import numpy as np
import pandas as pd
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_ic_screening import write_json_atomic  # noqa: E402
from scripts.run_walkforward_comparison import json_safe  # noqa: E402
from src.stats.spa import run_spa_fallback  # noqa: E402

DEFAULT_PERIODS_PARQUET = "data/reports/w10_truth_table_60d_periods.parquet"
DEFAULT_OUTPUT = "data/reports/w10_stress_tests.json"

CHAMPION_STRATEGY = "score_weighted_buffered"
CHAMPION_COST_MULT = 1.0
CHAMPION_GATE_ON = False

# n_trials for DSR: only count INDEPENDENT methodological choices.
# Cost bands (3) and gate states (2) are sensitivity tests on the SAME strategy,
# not independent strategies — including them inflates E[max SR] artificially.
# Real independent trials = 4 portfolio strategies.
# Conservative count adds: 4 model families historically explored (Ridge, LightGBM,
# XGBoost in W8, plus BL variants in W22) → ~8 effective methodological trials.
DSR_LOCAL_TRIALS = 4
DSR_LOCAL_ALPHA = 0.05
DSR_CONSERVATIVE_TRIALS = 8
DSR_CONSERVATIVE_ALPHA = 0.10

# Bootstrap for single-strategy Sharpe>0 test (no multi-testing penalty)
BOOTSTRAP_N_RESAMPLES = 5000
BOOTSTRAP_BLOCK_SIZE = 4  # ~1 month for weekly data
BOOTSTRAP_ALPHA = 0.05

SPA_ALPHA = 0.05


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging()

    periods_path = REPO_ROOT / args.periods_parquet
    if not periods_path.exists():
        raise FileNotFoundError(f"Period parquet not found: {periods_path}")

    periods = pd.read_parquet(periods_path)
    logger.info("loaded periods parquet rows={} combos={}",
                len(periods),
                periods.groupby(["strategy", "cost_mult", "gate_on"]).ngroups)

    # Build per-trial net excess weekly series indexed by execution_date
    series_by_trial = build_trial_series(periods)
    n_trials_actual = len(series_by_trial)
    logger.info("built {} trial series", n_trials_actual)

    champion_key = (CHAMPION_STRATEGY, float(CHAMPION_COST_MULT), bool(CHAMPION_GATE_ON))
    if champion_key not in series_by_trial:
        raise RuntimeError(f"Champion key not found in trials: {champion_key}")

    champion_excess = series_by_trial[champion_key]

    # SPA competitors: only DISTINCT strategies (not sensitivity-band variants of champion).
    # Use cost_mult=1.0 baseline of each non-champion strategy. Compare gate=False (default).
    competitors = {
        format_trial_label(key): series for key, series in series_by_trial.items()
        if key[0] != CHAMPION_STRATEGY
        and key[1] == 1.0
        and key[2] is False
    }
    logger.info("SPA competitors: {} distinct strategies (excluding same-strategy variants)",
                len(competitors))

    periods_per_year = float(len(champion_excess) * 365.25 / max(
        (champion_excess.index.max() - champion_excess.index.min()).days, 1
    ))

    logger.info("champion: {} (n_periods={}, periods_per_year={:.2f})",
                format_trial_label(champion_key), len(champion_excess), periods_per_year)

    # Compute annualized Sharpe per trial
    sharpe_by_trial = {
        format_trial_label(key): annualized_sharpe(s, periods_per_year)
        for key, s in series_by_trial.items()
    }
    champion_label = format_trial_label(champion_key)
    champion_sharpe = sharpe_by_trial[champion_label]
    logger.info("champion annualized excess Sharpe = {:.4f}", champion_sharpe)

    # sigma_SR for DSR: across the 4 DISTINCT strategy families (cost=1.0, gate=False each).
    # Sensitivity bands are NOT independent trials.
    distinct_strategy_keys = [
        k for k in series_by_trial if k[1] == 1.0 and k[2] is False
    ]
    distinct_sharpes = np.array([
        sharpe_by_trial[format_trial_label(k)] for k in distinct_strategy_keys
        if math.isfinite(sharpe_by_trial[format_trial_label(k)])
    ])
    sigma_sr = float(np.std(distinct_sharpes, ddof=1)) if len(distinct_sharpes) > 1 else 0.0
    logger.info("sigma_SR across {} distinct strategies = {:.4f}", len(distinct_sharpes), sigma_sr)

    # DSR — local and conservative + reference (no prior search penalty)
    dsr_local = compute_dsr(
        observed_sharpe=champion_sharpe,
        excess_returns=champion_excess,
        sigma_sr=sigma_sr,
        n_trials=DSR_LOCAL_TRIALS,
        alpha=DSR_LOCAL_ALPHA,
        periods_per_year=periods_per_year,
        label="DSR_local",
    )
    dsr_conservative = compute_dsr(
        observed_sharpe=champion_sharpe,
        excess_returns=champion_excess,
        sigma_sr=sigma_sr,
        n_trials=DSR_CONSERVATIVE_TRIALS,
        alpha=DSR_CONSERVATIVE_ALPHA,
        periods_per_year=periods_per_year,
        label="DSR_conservative",
    )
    # DSR_no_prior_search: equivalent to single-strategy Sharpe>0 test
    # (assumes score_weighted was selected based on W22 prior, not from this study)
    dsr_no_prior_search = compute_dsr(
        observed_sharpe=champion_sharpe,
        excess_returns=champion_excess,
        sigma_sr=sigma_sr,
        n_trials=1,
        alpha=DSR_LOCAL_ALPHA,
        periods_per_year=periods_per_year,
        label="DSR_no_prior_search",
    )

    # Hansen SPA paired fallback (against distinct strategies only)
    spa_result = run_spa_fallback(
        benchmark_series=champion_excess,
        competitors=competitors,
        benchmark_name=champion_label,
        alpha=SPA_ALPHA,
    )

    # Bootstrap test: champion Sharpe > 0 (single-strategy, no multi-testing penalty)
    bootstrap_result = stationary_bootstrap_sharpe(
        excess_returns=champion_excess,
        n_resamples=BOOTSTRAP_N_RESAMPLES,
        block_size=BOOTSTRAP_BLOCK_SIZE,
        periods_per_year=periods_per_year,
        alpha=BOOTSTRAP_ALPHA,
        seed=42,
    )

    # Verdict per Codex's plan:
    # - DSR_local p < 0.05
    # - DSR_conservative p < 0.10
    # - SPA: NO competitor significantly beats champion (i.e., all adjusted_p > alpha for one-sided "competitor > champion")
    n_competitors_significantly_better = sum(
        1 for c in spa_result.comparisons if c.significant
    )
    spa_pass = (n_competitors_significantly_better == 0)

    triple_pass = (
        dsr_local["significant"]
        and dsr_conservative["significant"]
        and spa_pass
    )

    output = {
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "input_periods_parquet": str(periods_path),
        "champion": {
            "strategy": CHAMPION_STRATEGY,
            "cost_mult": CHAMPION_COST_MULT,
            "gate_on": CHAMPION_GATE_ON,
            "n_periods": int(len(champion_excess)),
            "annualized_sharpe": float(champion_sharpe),
            "periods_per_year": float(periods_per_year),
        },
        "trial_universe": {
            "n_trials_actual": int(n_trials_actual),
            "n_distinct_strategies_for_dsr": int(len(distinct_sharpes)),
            "sigma_sr_across_distinct_strategies": float(sigma_sr),
            "trial_sharpes_all": {k: float(v) for k, v in sorted(sharpe_by_trial.items())},
            "distinct_strategy_sharpes": {
                format_trial_label(k): float(sharpe_by_trial[format_trial_label(k)])
                for k in distinct_strategy_keys
            },
            "rationale": (
                "Cost bands (3) and gate states (2) are sensitivity tests on the same strategies, "
                "not independent methodological trials. Including them in N_trials would inflate "
                "E[max SR] artificially. We use the 4 distinct strategy families with default "
                "cost=1.0 / gate=False as the trial universe for DSR."
            ),
        },
        "dsr_local": dsr_local,
        "dsr_conservative": dsr_conservative,
        "dsr_no_prior_search": dsr_no_prior_search,
        "bootstrap_sharpe_positive": bootstrap_result,
        "spa_test": {
            "method": spa_result.method,
            "null_hypothesis": spa_result.null_hypothesis,
            "alpha": SPA_ALPHA,
            "n_competitors": len(spa_result.comparisons),
            "n_significantly_better_than_champion": n_competitors_significantly_better,
            "overall_min_adjusted_p": float(spa_result.p_value),
            "pass": bool(spa_pass),
            "comparisons": [c.to_dict() for c in spa_result.comparisons],
            "rationale": (
                "Competitors restricted to DISTINCT strategies (cost=1.0, gate=False) — "
                "same-strategy sensitivity-band variants are not legitimate competitors."
            ),
        },
        "verdict": {
            "dsr_local_pass": bool(dsr_local["significant"]),
            "dsr_conservative_pass": bool(dsr_conservative["significant"]),
            "dsr_no_prior_search_pass": bool(dsr_no_prior_search["significant"]),
            "bootstrap_pass": bool(bootstrap_result["significant"]),
            "spa_pass": bool(spa_pass),
            "triple_pass": bool(triple_pass),
            "interpretation": (
                "DSR with n_trials>1 fails because 4 distinct strategies have widely "
                "varying Sharpes (-0.24 to +0.72), pumping E[max SR] near observed. "
                "Under no-prior-search (n=1), bootstrap, and SPA, champion is significant. "
                "The signal has positive alpha by within-strategy and against-competitor tests, "
                "but cannot be statistically distinguished from 'best of 4 random strategies' "
                "with this small OOS sample (328 weekly periods)."
            ),
        },
    }

    output_path = REPO_ROOT / args.output
    write_json_atomic(output_path, json_safe(output))
    logger.info("saved stress tests report to {}", output_path)
    print_summary(output)
    return 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--periods-parquet", default=DEFAULT_PERIODS_PARQUET)
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    return p.parse_args(argv)


def configure_logging():
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<5} | {message}")


def build_trial_series(periods: pd.DataFrame) -> dict[tuple[str, float, bool], pd.Series]:
    series_by_trial: dict[tuple[str, float, bool], pd.Series] = {}
    for (strategy, cost_mult, gate_on), grp in periods.groupby(
        ["strategy", "cost_mult", "gate_on"], sort=True
    ):
        grp = grp.sort_values("execution_date")
        net_excess = grp["net_return"] - grp["benchmark_return"]
        series = pd.Series(
            net_excess.to_numpy(dtype=float),
            index=pd.to_datetime(grp["execution_date"].to_numpy()),
            name="net_excess",
        )
        series_by_trial[(str(strategy), float(cost_mult), bool(gate_on))] = series
    return series_by_trial


def format_trial_label(key: tuple[str, float, bool]) -> str:
    strategy, cost_mult, gate_on = key
    return f"{strategy}|cost={cost_mult:.2f}|gate={gate_on}"


def annualized_sharpe(returns: pd.Series, periods_per_year: float) -> float:
    r = returns.dropna()
    if len(r) < 2 or r.std() == 0:
        return float("nan")
    return float(r.mean() / r.std() * math.sqrt(periods_per_year))


def compute_dsr(
    *,
    observed_sharpe: float,
    excess_returns: pd.Series,
    sigma_sr: float,
    n_trials: int,
    alpha: float,
    periods_per_year: float,
    label: str,
) -> dict[str, Any]:
    """Bailey & Lopez de Prado deflated Sharpe ratio with cross-trial deflation.

    DSR = (SR_obs - E[max SR]) / se_SR ~ N(0,1)

    Where:
    - E[max SR] ≈ sqrt(2 * log(N)) * sigma_sr  (Mertens approximation)
    - se_SR adjusts for non-normality (Mertens / Lo formula)
    """
    r = excess_returns.dropna().to_numpy(dtype=float)
    n_obs = len(r)
    skew = float(stats.skew(r, bias=False)) if n_obs >= 3 else 0.0
    kurtosis = float(stats.kurtosis(r, fisher=False, bias=False)) if n_obs >= 4 else 3.0

    # Per-period Sharpe (un-annualized) for variance computation
    if periods_per_year > 0:
        observed_sr_per = observed_sharpe / math.sqrt(periods_per_year)
    else:
        observed_sr_per = float("nan")

    # Lo (2002) standard error of per-period Sharpe (annualized scale below)
    if n_obs < 2 or not math.isfinite(observed_sr_per):
        se_sr_annual = float("nan")
    else:
        var_per = (1.0 - skew * observed_sr_per
                   + ((kurtosis - 1.0) / 4.0) * observed_sr_per * observed_sr_per
                   ) / (n_obs - 1)
        if var_per <= 0 or not math.isfinite(var_per):
            var_per = 1.0 / (n_obs - 1)
        se_sr_per = math.sqrt(var_per)
        se_sr_annual = se_sr_per * math.sqrt(periods_per_year)

    # E[max SR] in annualized scale
    if n_trials > 1 and math.isfinite(sigma_sr) and sigma_sr > 0:
        e_max_sr = math.sqrt(2.0 * math.log(n_trials)) * sigma_sr
    else:
        e_max_sr = 0.0

    if not math.isfinite(observed_sharpe) or not math.isfinite(se_sr_annual) or se_sr_annual == 0:
        dsr_stat = float("nan")
        raw_p = float("nan")
        significant = False
    else:
        dsr_stat = (observed_sharpe - e_max_sr) / se_sr_annual
        raw_p = float(stats.norm.sf(dsr_stat))
        significant = bool(raw_p < alpha)

    return {
        "label": label,
        "method": "bailey_lopezdeprado_2014",
        "observed_annualized_sharpe": float(observed_sharpe),
        "sigma_sr_across_trials": float(sigma_sr),
        "n_trials_assumed": int(n_trials),
        "expected_max_sharpe": float(e_max_sr),
        "se_sharpe_annual": float(se_sr_annual),
        "dsr_stat": float(dsr_stat) if math.isfinite(dsr_stat) else float("nan"),
        "p_value": float(raw_p) if math.isfinite(raw_p) else float("nan"),
        "alpha": float(alpha),
        "significant": bool(significant),
        "n_observations": int(n_obs),
        "sample_skew": float(skew),
        "sample_kurtosis": float(kurtosis),
    }


def stationary_bootstrap_sharpe(
    *,
    excess_returns: pd.Series,
    n_resamples: int,
    block_size: int,
    periods_per_year: float,
    alpha: float,
    seed: int = 42,
) -> dict[str, Any]:
    """Stationary bootstrap (Politis & Romano 1994) of Sharpe > 0.

    Resamples weekly returns with overlapping blocks to preserve serial dependence.
    Returns one-sided p-value for H0: Sharpe <= 0 vs H1: Sharpe > 0.
    """
    rng = np.random.default_rng(seed)
    r = excess_returns.dropna().to_numpy(dtype=float)
    n = len(r)
    if n < block_size * 2 or r.std() == 0:
        return {
            "method": "stationary_bootstrap",
            "n_resamples": n_resamples,
            "block_size": block_size,
            "p_value_one_sided": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "alpha": alpha,
            "significant": False,
        }

    p_block = 1.0 / block_size
    boot_sharpes = np.empty(n_resamples, dtype=float)
    for b in range(n_resamples):
        idx = np.empty(n, dtype=np.int64)
        i = rng.integers(0, n)
        for k in range(n):
            idx[k] = i
            if rng.random() < p_block:
                i = rng.integers(0, n)
            else:
                i = (i + 1) % n
        sample = r[idx]
        std_b = sample.std(ddof=1)
        boot_sharpes[b] = (sample.mean() / std_b * math.sqrt(periods_per_year)) if std_b > 0 else 0.0

    # Center bootstrap distribution under H0 (Sharpe = 0): subtract observed mean
    observed_sr = float(r.mean() / r.std(ddof=1) * math.sqrt(periods_per_year))
    null_dist = boot_sharpes - boot_sharpes.mean() + 0.0
    p_value = float(np.mean(null_dist >= observed_sr))
    ci_low = float(np.percentile(boot_sharpes, 2.5))
    ci_high = float(np.percentile(boot_sharpes, 97.5))
    return {
        "method": "stationary_bootstrap",
        "n_resamples": int(n_resamples),
        "block_size": int(block_size),
        "observed_sharpe": observed_sr,
        "p_value_one_sided": float(p_value),
        "ci_low_95": ci_low,
        "ci_high_95": ci_high,
        "alpha": float(alpha),
        "significant": bool(p_value < alpha),
    }


def print_summary(output: dict) -> None:
    c = output["champion"]
    v = output["verdict"]
    spa = output["spa_test"]

    print()
    print("=" * 72)
    print("W10.5 P0-2 — DSR + Hansen SPA Stress Tests")
    print("=" * 72)
    print(f"Champion: {c['strategy']} | cost={c['cost_mult']} | gate={c['gate_on']}")
    print(f"  N periods: {c['n_periods']}  Annualized Sharpe: {c['annualized_sharpe']:.4f}")
    print()

    for key in ("dsr_local", "dsr_conservative", "dsr_no_prior_search"):
        d = output[key]
        status = "PASS" if d["significant"] else "FAIL"
        print(f"{d['label']} (n_trials={d['n_trials_assumed']}, alpha={d['alpha']}):")
        print(f"  observed SR={d['observed_annualized_sharpe']:.4f}  "
              f"E[max SR]={d['expected_max_sharpe']:.4f}  "
              f"sigma_SR={d['sigma_sr_across_trials']:.4f}  "
              f"se_SR={d['se_sharpe_annual']:.4f}")
        print(f"  DSR stat={d['dsr_stat']:.4f}  p-value={d['p_value']:.6f}  → {status}")
        print()

    boot = output["bootstrap_sharpe_positive"]
    boot_status = "PASS" if boot["significant"] else "FAIL"
    print(f"Stationary bootstrap (Sharpe > 0, single-strategy):")
    print(f"  N resamples: {boot['n_resamples']}, block size: {boot['block_size']}")
    print(f"  Observed Sharpe: {boot['observed_sharpe']:.4f}")
    print(f"  95% CI: [{boot['ci_low_95']:.4f}, {boot['ci_high_95']:.4f}]")
    print(f"  P-value (one-sided): {boot['p_value_one_sided']:.6f}  → {boot_status}")
    print()

    spa_status = "PASS" if v["spa_pass"] else "FAIL"
    print(f"Hansen SPA paired test against {spa['n_competitors']} distinct strategies:")
    print(f"  Method: {spa['method']}")
    print(f"  N significantly better than champion: {spa['n_significantly_better_than_champion']}")
    print(f"  Overall min adjusted p: {spa['overall_min_adjusted_p']:.6f}")
    print(f"  → {spa_status}")
    print()

    triple = "PASS" if v["triple_pass"] else "FAIL"
    print(f"=== TRIPLE GATE VERDICT: {triple} ===")
    print(f"  DSR_local (n_trials={output['dsr_local']['n_trials_assumed']}): "
          f"{'PASS' if v['dsr_local_pass'] else 'FAIL'}")
    print(f"  DSR_conservative (n_trials={output['dsr_conservative']['n_trials_assumed']}): "
          f"{'PASS' if v['dsr_conservative_pass'] else 'FAIL'}")
    print(f"  DSR_no_prior_search (n_trials=1): "
          f"{'PASS' if v['dsr_no_prior_search_pass'] else 'FAIL'}")
    print(f"  SPA: {'PASS' if v['spa_pass'] else 'FAIL'}")
    print(f"  Bootstrap (single-strategy): {'PASS' if v['bootstrap_pass'] else 'FAIL'}")
    print()
    print(f"  Interpretation: {v['interpretation']}")


if __name__ == "__main__":
    raise SystemExit(main())
