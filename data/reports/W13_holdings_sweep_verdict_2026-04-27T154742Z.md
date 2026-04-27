# W13 Holdings Sweep Verdict

**Generated:** 2026-04-27T15:47:42.722768+00:00
**Baseline:** baseline
**Baseline metrics:** net excess 12.54%, IR 1.26, max DD 6.13%, mean holdings 74

## Comparison Table

| Config | Mean Holdings | Net Excess (Δ vs base) | IR (Δ) | Max DD (Δ) | Turnover | CVaR Hit | Final Cash |
|---|---|---|---|---|---|---|---|
| `baseline` | 74 (post) / 74 (pre) | 12.54% | 1.26 | 6.13% | 5.38% | 0.0% | 0.0% |
| `cap50_only` | 50 (post) / 50 (pre) | 14.30% (+1.77%) | 1.33 (+0.07) | 6.24% (+0.10%) | 12.81% | 0.0% | 0.0% |
| `cap30_only` | 30 (post) / 30 (pre) | 14.28% (+1.74%) | 1.21 (-0.05) | 7.51% (+1.37%) | 15.15% | 0.0% | 0.0% |
| `cap20_only` | 20 (post) / 20 (pre) | 6.63% (-5.91%) | 0.62 (-0.64) | 7.75% (+1.61%) | 19.40% | 0.0% | 0.0% |
| `layer3_only` | 64 (post) / 67 (pre) | 8.38% (-4.16%) | 0.95 (-0.31) | 5.29% (-0.84%) | 12.27% | 0.0% | 0.8% |
| `cap50_plus_layer3` | 54 (post) / 56 (pre) | 9.06% (-3.47%) | 0.91 (-0.35) | 7.65% (+1.51%) | 18.79% | 0.0% | 0.9% |
| `cap30_plus_layer3` | 32 (post) / 31 (pre) | 11.60% (-0.94%) | 1.02 (-0.25) | 8.06% (+1.93%) | 21.26% | 0.0% | 0.8% |
| `cap20_plus_layer3` | 24 (post) / 21 (pre) | 5.74% (-6.80%) | 0.58 (-0.69) | 8.24% (+2.10%) | 21.79% | 0.0% | 1.0% |

## Key Decompositions

- **Layer 3 alone** moves net excess by -4.16% and IR by -0.31.
- **30-stock cap alone** moves net excess by +1.74% and IR by -0.05.
- **30-stock cap + Layer 3** (live-style) moves net excess by -0.94% and IR by -0.25.

## Notes
- Baseline reproduces W10 truth-table `score_weighted_buffered` at cost_mult=1.0, gate_off.
- Layer 3 uses PIT-correct trailing returns (each period sees only history up to execution_date).
- CVaR config: 99% confidence, floor -5%, 20% gross haircut, max 3 rounds.
- Layer 3 sector deviation cap = 15% (research default; live W12 ran tighter ~9%).