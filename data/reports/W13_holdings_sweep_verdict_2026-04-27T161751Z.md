# W13 Holdings Sweep Verdict

**Generated:** 2026-04-27T16:17:51.108143+00:00
**Baseline:** baseline
**Baseline metrics:** net excess 7.34%, IR 0.72, max DD 14.04%, mean holdings 147

## Comparison Table

| Config | Mean Holdings | Net Excess (Δ vs base) | IR (Δ) | Max DD (Δ) | Turnover | CVaR Hit | Final Cash |
|---|---|---|---|---|---|---|---|
| `baseline` | 147 (post) / 147 (pre) | 7.34% | 0.72 | 14.04% | 2.65% | 0.0% | 0.0% |
| `cap50_only` | 50 (post) / 50 (pre) | 3.13% (-4.20%) | 0.33 (-0.39) | 18.88% (+4.84%) | 20.76% | 0.0% | 0.0% |
| `cap30_only` | 30 (post) / 30 (pre) | 3.50% (-3.83%) | 0.33 (-0.38) | 19.80% (+5.76%) | 24.10% | 0.0% | 0.0% |
| `cap20_only` | 20 (post) / 20 (pre) | -0.70% (-8.04%) | 0.11 (-0.60) | 20.26% (+6.22%) | 28.56% | 0.0% | 0.0% |
| `layer3_only` | 63 (post) / 69 (pre) | -3.11% (-10.45%) | -0.24 (-0.96) | 26.92% (+12.88%) | 10.90% | 65.9% | 24.5% |
| `cap50_plus_layer3` | 48 (post) / 54 (pre) | -6.85% (-14.18%) | -0.51 (-1.23) | 41.45% (+27.40%) | 20.13% | 65.9% | 24.6% |
| `cap30_plus_layer3` | 32 (post) / 34 (pre) | -5.41% (-12.75%) | -0.36 (-1.07) | 38.69% (+24.64%) | 22.72% | 63.4% | 23.4% |
| `cap20_plus_layer3` | 24 (post) / 23 (pre) | -7.49% (-14.83%) | -0.51 (-1.23) | 44.06% (+30.01%) | 25.03% | 62.5% | 23.2% |
| `layer3_medium` | 76 (post) / 78 (pre) | -1.67% (-9.01%) | -0.06 (-0.78) | 23.29% (+9.24%) | 9.10% | 18.3% | 6.6% |
| `layer3_loose` | 82 (post) / 83 (pre) | 3.96% (-3.38%) | 0.43 (-0.29) | 16.78% (+2.73%) | 7.97% | 12.2% | 3.1% |
| `cap30_plus_layer3_med` | 42 (post) / 42 (pre) | -1.71% (-9.04%) | -0.04 (-0.75) | 25.67% (+11.63%) | 24.80% | 17.7% | 6.8% |
| `cap30_plus_layer3_loose` | 41 (post) / 41 (pre) | 1.51% (-5.83%) | 0.20 (-0.51) | 18.55% (+4.51%) | 24.77% | 12.8% | 4.1% |

## Key Decompositions

- **Layer 3 alone** moves net excess by -10.45% and IR by -0.96.
- **30-stock cap alone** moves net excess by -3.83% and IR by -0.38.
- **30-stock cap + Layer 3** (live-style) moves net excess by -12.75% and IR by -1.07.

## Notes
- Baseline reproduces W10 truth-table `score_weighted_buffered` at cost_mult=1.0, gate_off.
- Layer 3 uses PIT-correct trailing returns (each period sees only history up to execution_date).
- CVaR config: 99% confidence, floor -5%, 20% gross haircut, max 3 rounds.
- Layer 3 sector deviation cap = 15% (research default; live W12 ran tighter ~9%).