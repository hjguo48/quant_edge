# QuantEdge 项目全景报告 — 供外部 AI 深度分析

**生成日期:** 2026-04-17
**生成者:** Claude Opus 4.6 (项目 AI 协调者)
**目的:** 提交给 GPT Pro 进行独立深度分析，识别盲点和改进方向

---

## 一、项目背景

QuantEdge 是一个**研究驱动的机构级美股量化生产系统**。单人开发，AI 辅助（Claude 协调 + Codex 执行）。目标是构建一个从 Alpha 发现到实盘部署的完整量化系统。

### 核心理念
- 先证明 Alpha 存在 → 再证明能赚钱 → 最后产品化
- 统计严谨：PIT discipline, purge/embargo, DSR/SPA 检验
- 研究定型后再工程化，避免返工

### 技术栈
- Python 3.11, TimescaleDB, Redis, MLflow, Airflow, Celery
- ML: scikit-learn (Ridge, ElasticNet), XGBoost, LightGBM
- Frontend: React + TypeScript + ECharts (已有 dashboard)
- 部署: Docker Compose on WSL2 (计划迁移阿里云 ECS)

---

## 二、数据层现状

### 已有数据源

| 数据源 | 表名 | Rows | Tickers | 日期范围 | 提供商 | 状态 |
|--------|------|------|---------|---------|--------|------|
| 日频价格 OHLCV | stock_prices | 1.71M | 711 | 2015→2026 | Polygon Massive | OK |
| 基本面 (EAV) | fundamentals_pit | 550K | 693 | 2015→2026 | FMP + Polygon | OK |
| 盈余 actual/estimated | earnings_estimates | 28K | 677 | 2015→2026 | FMP | OK |
| 空头仓位 | short_interest | 122K | 696 | 2018→2026 | Polygon | OK |
| 内部交易 | insider_trades | 526K | 722 | 2015→2026 | FMP | OK |
| SEC 文件 metadata | sec_filings | 972K | 643 | 2009→2026 | FMP | OK |
| 宏观指标 | macro_series_pit | 22K | 6 series | VIX/10Y/2Y/FFR/Credit | FRED | OK |
| Sector ETF 价格 | stock_prices (ETF) | 35K | 14 ETFs | 2016→2026 | Polygon | OK |
| 分析师预估 | analyst_estimates | 5K | 632 | forward-only | FMP | 弱: 无历史快照 |
| 新闻情感 | news_sentiment | 2K | 86 | 2024+ | Polygon | 不可用于回测 |

### 已知数据问题
1. analyst_estimates 只有 forward-looking 预估（FMP API 限制），回测中 ~75% NaN
2. news_sentiment 覆盖极差（86 tickers, 2024+ only）
3. SPY 从 2016-04 开始（Polygon 10 年限制），不是 2015
4. universe_membership 只有当前成员，无历史成分变动
5. stock_prices 不是真正的版本化 PIT（单行/ticker/date，无多版本）
6. feature_store (DB) 和 parquet (研究) 特征数不一致（176 vs 238）

---

## 三、特征工程现状

### 特征分类

| 类别 | 数量 | 主要内容 |
|------|------|---------|
| Technical | 37 | ret_5d/10d/20d/60d, vol, ATR, RSI, MACD, BB, residual_momentum, idio_vol, stock_beta_252, reversal |
| Fundamental | 17 | PE/PB/PS, ROE/ROA, asset_growth, revenue_growth, accruals, debt_to_equity |
| Macro | 10 | VIX, yield_spread, credit_spread, sp500_breadth |
| Alternative | 38 | earnings_surprise/recency, insider_intensity/cluster/abnormal, short_interest_sector_rel/squeeze, SEC filing recency/burst, PEAD setup |
| Composite | 26 | value_mom, quality_value, macro_risk, high_vix_x_beta, curve_inverted_x_growth, sector_pressure |
| Sector Rotation | 5 | sector_rel_ret, sector_volume_surge, sector_pressure, stock_vs_sector |
| **总计 base** | **133** | |
| + is_missing | 133 | |
| **总候选** | **~266** | (去重后约 238-240) |

### IC Screening 历史

| 版本 | 候选 | Labels | Passed | Ridge Retained | 说明 |
|------|------|--------|--------|---------------|------|
| v3 (Phase E) | 184 | 5D (错误) | 13 | 15 | sign_cons>=0.6 |
| SA-v2 | 184 | 5D (错误) | 15 | 25 | sign_cons>=0.5 |
| V4 | 210 | 5D (错误) | 17 | 20 | +macro interaction |
| V5 | 210 | 5D (错误) | 18 | 21 | +breadth fix |
| V5 60D true | 210 | **60D (正确)** | **76** | 35 | 大量 fundamental 涌现 |
| **V6 (运行中)** | **~240** | **60D** | ? | ? | +SEC/event/ETF 新特征 |

### 关键发现
1. **IC screening 一直用错了 labels** — 默认 HORIZON_DAYS=5，所有"60D screening"实际用 5D labels
2. 用 60D labels 后 76 个特征通过（vs 5D 的 18 个），fundamental/value 类大量涌现
3. 但 35 个 Ridge retained 反而不如 19 个（IC 0.046 vs 0.062）— 过多弱特征稀释强信号
4. analyst_coverage 去掉后 IC 反升（is_missing 编码了 data-vendor bias）
5. Macro interaction 特征（curve_inverted_x_growth, high_vix_x_beta）是最有效的新增

---

## 四、模型层现状

### 当前只用 Ridge Regression

| Horizon | IC | t-stat | p (1-sided) | Positive | Features | Model | 状态 |
|---------|-----|--------|-------------|----------|----------|-------|------|
| **60D** | **0.062** | **2.17** | **0.025** | **10/13** | 19 (no-analyst) | Ridge | **G3 6/6 PASS** |
| 5D | 0.015 | 2.03 | 0.033 | 10/13 | 21 (60D 特征) | Ridge | 未用专属特征 |
| 1D | 0.017 | 1.10 | 0.146 | 6/13 | 21 (60D 特征) | Ridge | 未用专属特征 |
| 20D | 0.024 | 1.41 | 0.095 | 6/11 | 21 (60D 特征) | Ridge | 未用专属特征 |

### 未测试的模型
- ElasticNet (L1+L2)
- XGBoost — 60D purge 下 IC 暴跌 80%，但 1D/5D 短 purge 未测
- LightGBM — 同上
- Stacking / Ensemble — 未做

### G3 Gate 6 项检验 (60D V5 no-analyst, 13 windows)

| Gate | 结果 | 数值 |
|------|------|------|
| Bootstrap CI > 0 | PASS | |
| Cost-adjusted excess > 5% | PASS | |
| DSR significant | PASS | p≈0 |
| IC t-test (one-sided) | PASS | p=0.025 |
| OOS IC > 0.03 | PASS | IC=0.062 |
| SPA vs zero-IC null | PASS | p≈0 |

---

## 五、经济可行性 — 最大未解决问题

### G3 Gate proxy vs 实际 portfolio backtest

| 来源 | Gross Excess | Cost Drag | Net Excess | 方法 |
|------|-------------|-----------|-----------|------|
| G3 Gate proxy | ~35% | ~5% | ~30% | 理论 top-decile excess 估算 |
| Post-cost script | 13.7% | 23.2% | **-9.6%** | equal-weight top-decile 模拟 |
| **Production-style** | ? | ? | ? | **未做** (score_weighted + turnover controls) |

**两个数差了 44 个百分点。** G3 太乐观（理论估算），post-cost script 太悲观（没用生产级组合构建）。真实 net excess 未知。

### 弱窗口问题

| 窗口 | Test 期 | 60D IC | 诊断 |
|------|---------|--------|------|
| W5 | 2021-09→2022-02 | -0.116 | Regime break — val IC +0.145 但 test 反转 |
| W6 | 2022-03→2022-08 | -0.061 | Val IC 已负 — 可 gating |
| W10 | 2024-03→2024-08 | +0.031 | 去掉 analyst 后翻正 |
| W11 | 2024-09→2025-02 | -0.049 | Val IC 已负 — 可 gating |

---

## 六、当前执行计划 (双轨制)

### 部署轨 (本周)
- 锁定 V5 no-analyst (19f, 60D Ridge) 为生产 baseline — ✅ 已完成
- 确认 DAG + 周五灰度正常运行 — ⏳

### 研究轨 Step 1: 数据底座 + 全 Horizon 优化

| 子步骤 | 任务 | 状态 |
|--------|------|------|
| 1A | 修复 universe_membership | ✅ |
| 1B | 60D true IC screening | ✅ (76 passed) |
| 1C | SEC filing metadata features (8 个) | ✅ 已实现 |
| 1D | Repackaged event features (15 个) | ✅ 已实现 |
| 1E | ETF sector rotation proxy (5 个) | ✅ 已实现 |
| 1F | 冻结数据信封 | ⏳ |
| 1G | IC screening (60D/20D/10D/5D/1D 各一次) | 🔄 V6 60D 运行中 (15/29) |
| 1H | Walk-forward 13w × 5 horizons | ⏳ |
| 1I | G3 Gate × 5 horizons | ⏳ |
| 1J-1N | Model sweep × 5 horizons (Ridge/ElasticNet/XGB/LGB) | ⏳ |
| 1O | 最优配置锁定 | ⏳ |
| 1P | Production-style post-cost backtest × 5 horizons | ⏳ |
| 1Q | Cross-horizon 对比 | ⏳ |

**Step 1 Gate:** 至少一个 horizon annualized net excess > 5%

### Step 2: Cross-Horizon 融合
- 60D core + 短 horizon overlay
- Validation-IC gating
- Post-cost backtest

### Step 3: 数据扩展
- 3A: FMP analyst grades (免费) → 验证 revision signal
- 3B: 如果有效 → Zacks ZEEH ($2-8K/年) 或 I/B/E/S
- 3C: 期权 IV, Russell 1000, NLP

### Step 4: 工程化 + 产品化 (研究定型后)
- 回测系统 (净值曲线, Sharpe, 最大回撤)
- feature_store parity
- MLOps (auto-retrain, drift monitoring)
- 前端更新

---

## 七、关键发现与教训

1. Purge/embargo 将 IC 从 0.091 打到 0.038 — 58% 是 label leakage
2. 树模型在 60D purge 下 IC 暴跌 80% — 但 1D/5D 短 purge 未测试
3. analyst_coverage 去掉后 IC 反升 — is_missing 编码了 data-vendor bias
4. IC screening 一直用 5D labels — 60D labels 结果截然不同
5. G3 proxy (+35%) 和 portfolio backtest (-9.6%) 差 44 个百分点 — 不可互换
6. IC 是必要非充分条件 — 0.062 是研究 win, monetization 未验证
7. 5D 在 W10/W11 为正 (60D 为负) — 互补效应存在
8. Macro interaction 最有效 — 13 个新特征中只有 2 个通过
9. 不同 horizon 需要不同特征和模型
10. 数据先行 — 数据源扩充后特征/模型需要重跑
11. Ridge retained 特征数应控制在 20-25 以内 — 35 个反而更差
12. 多 Horizon 不是信号加权平均，而是决策层融合

---

## 八、需要外部分析的核心问题

1. **经济可行性**: IC=0.062 能否在扣除真实交易成本后盈利? G3 proxy 和 post-cost script 差异巨大，哪个更可信?

2. **特征数量 vs 质量**: 60D labels 筛出 76 个特征，但 35 个 Ridge 不如 19 个。最优特征数是多少? 如何选?

3. **模型选择**: Ridge 在 60D 下最优，但 1D/5D 短 purge 下树模型是否更好? 应该怎么设计实验?

4. **Horizon 融合**: 简单加权融合让 IC 下降。正确的融合方式是什么? 决策层融合 vs 信号层融合?

5. **数据 ROI**: 现有 ~240 特征中只有 20 个有效。增加更多数据源 (analyst revisions, options IV) 的边际 IC 贡献预期是多少?

6. **Plan 合理性**: 双轨制 (部署 + 研究) 是否合理? Step 顺序是否最优?

7. **弱窗口**: W5 (regime break) 本质上不可预测? 还是有方法改善? Validation-IC gating 是否是正确方向?

8. **竞争力**: IC=0.062 在量化行业处于什么水平? 距离"可部署策略"还差什么?

---

## 九、代码结构

```
src/
├── data/          # 数据层 (7 数据源适配器 + PIT 查询)
├── features/      # 特征工程 (133 base features, 5 modules)
├── labels/        # 多 Horizon 标签 (1D/5D/10D/20D/60D)
├── models/        # Ridge + XGBoost + LightGBM + ElasticNet
├── backtest/      # Walk-forward + Almgren-Chriss + 执行模拟
├── portfolio/     # equal_weight + vol_weighted + black_litterman + score_weighted
├── stats/         # IC t-test + Bootstrap + DSR + SPA
├── risk/          # 四层风控
├── api/           # FastAPI backend
└── universe/      # 股票池管理 + active universe resolver

scripts/           # IC screening, walk-forward, G3 Gate, post-cost comparison
dags/              # Airflow DAG (daily data + weekly signal)
frontend/          # React + TypeScript + ECharts dashboard
```

### 关键 Artifacts

| Artifact | Path |
|----------|------|
| Production bundle | data/models/fusion_model_bundle_60d.json |
| 60D walk-forward | data/reports/walkforward_comparison_60d_v5_no_analyst.json |
| 60D G3 Gate | data/reports/g3_gate_v5_no_analyst.json |
| Post-cost comparison | data/reports/post_cost_comparison.json |
| IC screening (5D labels) | data/features/ic_screening_report_v5_60d.csv |
| IC screening (60D labels) | data/features/ic_screening_report_v5_60d_true.csv |
| V6 feature parquet | data/features/all_features_v6.parquet (构建中) |

---

## 十、当前运行任务

- V6 IC screening: 15/29 batches (52%), ~240 候选特征, 60D labels, 预计 ~5:00 完成
- 部署轨: V5 no-analyst bundle 已更新, 待确认 DAG
