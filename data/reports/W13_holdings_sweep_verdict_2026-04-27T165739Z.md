# W13 Holdings Sweep Verdict

**Generated:** 2026-04-27T16:57:39.034843+00:00
**Baseline:** baseline
**Baseline metrics:** net excess 7.34%, IR 0.72, max DD 14.04%, mean holdings 147

## Comparison Table

| Config | Mean Holdings | Net Excess (Δ vs base) | IR (Δ) | Max DD (Δ) | Turnover | CVaR Hit | Final Cash |
|---|---|---|---|---|---|---|---|
| `baseline` | 147 (post) / 147 (pre) | 7.34% | 0.72 | 14.04% | 2.65% | 0.0% | 0.0% |
| `cap50_only` | 50 (post) / 50 (pre) | 3.13% (-4.20%) | 0.33 (-0.39) | 18.88% (+4.84%) | 20.76% | 0.0% | 0.0% |
| `cap30_only` | 30 (post) / 30 (pre) | 3.50% (-3.83%) | 0.33 (-0.38) | 19.80% (+5.76%) | 24.10% | 0.0% | 0.0% |
| `cap20_only` | 20 (post) / 20 (pre) | -0.70% (-8.04%) | 0.11 (-0.60) | 20.26% (+6.22%) | 28.56% | 0.0% | 0.0% |
| `layer3_only` | 64 (post) / 68 (pre) | -3.22% (-10.55%) | -0.19 (-0.90) | 26.89% (+12.85%) | 11.25% | 24.1% | 9.7% |
| `cap50_plus_layer3` | 54 (post) / 58 (pre) | -6.13% (-13.47%) | -0.38 (-1.10) | 37.16% (+23.11%) | 21.83% | 23.8% | 9.7% |
| `cap30_plus_layer3` | 36 (post) / 37 (pre) | -3.84% (-11.18%) | -0.19 (-0.91) | 32.56% (+18.52%) | 24.91% | 28.0% | 10.6% |
| `cap20_plus_layer3` | 29 (post) / 27 (pre) | -6.18% (-13.51%) | -0.35 (-1.07) | 38.57% (+24.52%) | 26.89% | 30.5% | 11.4% |
| `layer3_medium` | 77 (post) / 80 (pre) | -1.71% (-9.04%) | -0.07 (-0.78) | 23.29% (+9.24%) | 9.10% | 16.2% | 7.4% |
| `layer3_loose` | 86 (post) / 87 (pre) | 1.34% (-6.00%) | 0.20 (-0.52) | 15.58% (+1.53%) | 7.52% | 15.5% | 5.1% |
| `cap30_plus_layer3_med` | 38 (post) / 38 (pre) | -2.26% (-9.59%) | -0.07 (-0.79) | 29.74% (+15.70%) | 24.90% | 15.9% | 7.3% |
| `cap30_plus_layer3_loose` | 39 (post) / 39 (pre) | -0.08% (-7.42%) | 0.08 (-0.63) | 21.35% (+7.31%) | 24.69% | 14.3% | 5.9% |

## Key Decompositions

- **Layer 3 alone** moves net excess by -10.55% and IR by -0.90.
- **30-stock cap alone** moves net excess by -3.83% and IR by -0.38.
- **30-stock cap + Layer 3** (live-style) moves net excess by -11.18% and IR by -0.91.

## Notes
- Baseline reproduces W10 truth-table `score_weighted_buffered` at cost_mult=1.0, gate_off.
- Layer 3 uses PIT-correct trailing returns (each period sees only history up to execution_date).
- CVaR config: 99% confidence, floor -5%, 20% gross haircut, max 3 rounds.
- Layer 3 sector deviation cap = 15% (research default; live W12 ran tighter ~9%).