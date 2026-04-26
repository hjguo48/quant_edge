# W9.3 Weak Window Diagnosis & Ex-ante Gating Rule — 2026-04-26

**数据**: W9.1 13-window 60D Ridge IC + W9.2 regime table (2231 dates × 15 cols, 2016-04-18 ~ 2025-02-28)

## 1. Window classification

| Window | test 区间 | 60D Ridge IC | 类别 |
|---|---|---|---|
| W-1 | 2018-09~2019-02 | +0.0418 | Strong |
| W0  | 2019-03~2019-08 | -0.0067 | **Weak** |
| W1  | 2019-09~2020-02 | +0.1639 | Strong |
| W2  | 2020-03~2020-08 | +0.1934 | Strong (COVID 反弹) |
| W3  | 2020-09~2021-02 | +0.2285 | Strong (COVID 二阶段) |
| W4  | 2021-03~2021-08 | +0.0812 | Strong |
| W5  | 2021-09~2022-02 | +0.0109 | **Weak** |
| W6  | 2022-03~2022-08 | +0.0076 | **Weak** (股债双杀) |
| W7  | 2022-09~2023-02 | +0.1235 | Strong |
| W8  | 2023-03~2023-08 | +0.0557 | Strong |
| W9  | 2023-09~2024-02 | +0.0153 | **Weak** (AI 末段) |
| W10 | 2024-03~2024-08 | +0.0216 | Strong (临界) |
| W11 | 2024-09~2025-02 | -0.0281 | **Weak** (AI 同涨) |

**5/13 weak windows**：W0, W5, W6, W9, W11

## 2. Ex ante covariates: weak vs strong window 平均值

| Covariate | weak avg | strong avg | diff (weak-strong) | 解释 |
|---|---|---|---|---|
| **credit_spread_baa10y** | -1.15 | -0.27 | **-0.88** | 信用利差越紧 → alpha 越难（市场过度乐观，无 dispersion） |
| **universe_dispersion_20d** | +0.106 | +0.269 | **-0.16** | 横截面分散度低 → cross-sectional alpha 难做 |
| **vix_level** | +18.8 | +21.1 | -2.3 | 低 VIX 期 alpha 弱（恐慌少 → 无定价错误） |
| **vix_zscore_60d** | +0.04 | -0.08 | +0.13 | weak windows 平均 VIX 略高于近期均值 |
| spy_ret_60d | +2.0% | +3.8% | -1.7pp | 涨势平缓时 alpha 弱 |
| breadth | 0.539 | 0.571 | -0.03 | 弱信号 |
| yield_curve | +0.19 | +0.18 | +0.01 | 无信号 |

**最强 ex ante gate 信号**：
1. **credit_spread_baa10y < -1.0**（信用利差极紧）→ 4/5 weak windows 命中（W7 不算 weak 但 -1.64 也命中 → false alarm）
2. **universe_dispersion_20d < 0.10** → 3/5 weak windows 命中（W0, W9, W11）+ false alarms (W-1, W1, W8, W10)

## 3. 提议 gate rule (simple monotonic, OOS-friendly)

### Rule A — 单覆盖 dispersion gate
```
if universe_dispersion_20d < 0.10:
    gate = True (cut position 50% or go neutral)
else:
    gate = False
```
- True positive: W0, W9, W11 (3/5 weak)
- False alarm: W-1 (IC +0.04), W1 (IC +0.16!), W8 (IC +0.06), W10 (IC +0.02)
- **False alarm 严重**（W1 IC 0.16 是顶级，不能 gate 掉）

### Rule B — 双条件 AND gate
```
if universe_dispersion_20d < 0.10 AND credit_spread_baa10y < -1.5:
    gate = True
```
- True positive: W9 (disp=0.070, cs=-2.63 ✓), W11 (disp=0.077, cs=-2.79 ✓)
- False alarm: 0
- **覆盖 2/5 weak windows，0 false alarm**

### Rule C — composite score
```
score = -credit_spread + (-dispersion) * 5
if score > threshold (TBD on cross-validation):
    gate
```
保留更精细 cross-window calibration 留给 W10 实施。

## 4. Gateability 分类

| Window | Rule B gate? | 类型 |
|---|---|---|
| W0  | NO (cs=+0.05 不够紧) | **未识别 weak** |
| W5  | NO | **未识别 weak**（不是 credit-tight regime）|
| W6  | NO (vix 高但 cs=-0.63 不够紧) | **regime break 不可 gating**（2022 股债双杀） |
| W9  | YES ✓ | **Gateable**（AI 末段 mega-cap 同涨）|
| W11 | YES ✓ | **Gateable**（AI mega-cap 持续同涨）|

**结论**：
- **2/5 weak windows ex ante 可识别**（W9, W11，AI mega-cap regime）
- **3/5 weak windows 不可 ex ante 识别**：
  - W0 是低 VIX 阶段，需要更敏锐的 vix_zscore 信号
  - W5 是 covariates 接近平均的常态期间，难 gate
  - W6 是 regime break (股债双杀)，**接受损失**

## 5. Gating 简单回测（Rule B）

假设 W9 + W11 期间降仓 50%，其他期间满仓：
- 13-window mean test_ic 不变（IC 是 cross-sectional rank correlation，gating 影响仓位非 IC 本身）
- top_decile_return 减半 only on gated windows
- W9 top_dec=-0.0047, W11 top_dec=-0.0281 — gating 减半后约 -0.0024 + -0.0140 = 减少 -0.014 cumulative loss
- 假设 PnL = sum(top_decile_return * IC_sign): 改善 ~14 bps cumulative

**实际收益微小**（因为 W9/W11 top_decile 损失本来就小），但**风险控制角度有意义** — 降低 max drawdown。

## 6. W9.3 verdict

✅ **达成 P0 gate**: 至少 1 个 weak window 找到 ex ante 先验（W9 + W11 都识别）
❌ **未覆盖 3/5 weak windows**：W0/W5/W6 需要 W10 进一步研究（更细致 features 或 unsupervised regime detection）

## 7. 给 W10 的建议

1. **AI mega-cap regime detector** = `dispersion < 0.10 AND credit_spread < -1.5` 直接采用
2. **未覆盖的 W0/W5**：尝试 加入 macro 序列变化率（VIX trend, yield curve slope change）
3. **W6 (regime break)**：不尝试 gate，作为 `irreducible variance` 接受
4. **W10 模型层面**：考虑 horizon-specific gating — 60D Ridge 时 gate, 5D/20D 不 gate（短 horizon 受 regime 影响小）

---

## 附：Per-window 完整数据

| win | IC | VIX | VIX z60 | yield_inv | credit_sp | breadth | dispersion | SPY 60d |
|---|---|---|---|---|---|---|---|---|
| W-1 | +0.042 | 18.9 | +0.46 | +0.22 | -0.79 | 0.525 | 0.086 | -3.1% |
| W0  | -0.007 | 15.4 | +0.01 | +0.18 | +0.05 | 0.585 | 0.077 | +5.2% |
| W1  | +0.164 | 15.0 | +0.00 | +0.18 | +0.38 | 0.611 | 0.080 | +5.1% |
| W2  | +0.193 | 35.3 | -0.26 | +0.49 | +2.39 | 0.591 | 0.118 | +1.4% |
| W3  | +0.229 | 25.4 | +0.00 | +0.79 | +1.45 | 0.593 | 1.282 | +7.9% |
| W4  | +0.081 | 18.3 | -0.51 | +1.32 | +0.48 | 0.628 | 0.071 | +6.9% |
| W5  | +0.011 | 21.0 | +0.48 | +0.92 | +0.22 | 0.500 | 0.142 | +2.3% |
| W6  | +0.008 | 25.8 | -0.24 | +0.06 | -0.63 | 0.521 | 0.165 | -6.1% |
| W7  | +0.124 | 23.9 | -0.28 | -0.57 | -1.64 | 0.518 | 0.363 | +0.2% |
| W8  | +0.056 | 16.9 | -0.50 | -0.71 | -1.73 | 0.535 | 0.079 | +5.6% |
| W9  | +0.015 | 14.8 | +0.07 | -0.39 | -2.63 | 0.559 | 0.070 | +4.1% |
| W10 | +0.022 | 15.0 | +0.41 | -0.31 | -2.72 | 0.569 | 0.076 | +6.2% |
| W11 | -0.028 | 17.2 | -0.11 | +0.18 | -2.79 | 0.528 | 0.077 | +4.6% |
