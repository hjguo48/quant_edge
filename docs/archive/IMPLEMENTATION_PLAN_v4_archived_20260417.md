# QuantEdge 实施计划 v4.1

**更新日期:** 2026-04-16
**当前状态:** Step 1 — 数据底座完善 + 经济真相测试

---

## 1. 项目定位

QuantEdge 是一个研究驱动的机构级美股量化系统。核心原则:
- **先证明能赚钱，再做产品** — Alpha 验证 → 经济可行性 → 产品化
- **统计严谨** — PIT discipline, purge/embargo, DSR/SPA 检验
- **分层演进** — 数据/特征/模型/组合/执行 五层独立优化
- **研究定型后再工程化** — 不在研究未定型时做生产部署，避免返工

---

## 2. 当前 Baseline

| 指标 | 值 |
|------|-----|
| 配置 | 60D Ridge, 19 features (no-analyst), 13 windows |
| IC | 0.062 |
| t-stat | 2.17 (p=0.025, one-sided) |
| ICIR | 0.553 |
| Positive windows | 10/13 |
| G3 Gate | 6/6 PASS (SPA 修复为 vs zero-IC null) |
| 经济可行性 | **未验证** — G3 proxy +35% vs post-cost script -9.6%, 需 production-style backtest |

### Multi-horizon 现状（全部使用 60D 优化的特征，未做 horizon-specific 优化）

| Horizon | IC | t-stat | Positive | 特征 | 模型 | 状态 |
|---------|-----|--------|----------|------|------|------|
| 60D | 0.062 | 2.17 | 10/13 | 19f (no-analyst) | Ridge | Baseline |
| 5D | 0.015 | 2.03 | 10/13 | 21f (60D 特征) | Ridge only | 待专属优化 |
| 1D | 0.017 | 1.10 | 6/13 | 21f (60D 特征) | Ridge only | 待专属优化 |
| 20D | 0.024 | 1.41 | 6/11 | 21f (60D 特征) | Ridge only | 弱, 不继续 |

---

## 3. 数据层

| 数据源 | Rows | Tickers | 日期范围 | 状态 |
|--------|------|---------|---------|------|
| stock_prices | 1.71M | 711 | 2015→2026 | ✅ |
| fundamentals_pit | 550K | 693 | 2015→2026 | ✅ |
| earnings_estimates | 28K | 677 | 2015→2026 | ✅ |
| short_interest | 122K | 696 | 2018→2026 | ✅ |
| insider_trades | 526K | 722 | 2015→2026 | ✅ |
| sec_filings | 972K | 643 | 2009→2026 | ✅ 数据有, 特征待建 |
| macro_series_pit | 22K | — | VIX/10Y/2Y/FFR/Credit | ✅ |
| analyst_estimates | 5K | 632 | forward-only | ⚠️ 排除出研究核心, 无历史快照 |
| news_sentiment | 2K | 86 | 2024+ | ❌ 不可用于回测 |
| SPY | 2.5K | 1 | 2016-04→2026 | ⚠️ Polygon 10yr limit |

### 特征
- 105 base + 105 is_missing = **210 候选**
- 5 类: Technical(37), Fundamental(17), Macro(10), Alternative(15), Composite(26)
- 当前 60D retained: 19 features

### 已知数据问题
- universe_membership: 仅 2,438 rows / 610 tickers / 2026 年, live 路径不一致 — **Step 1 修复**
- feature_store vs parquet 不一致 (176 vs 210 features) — **Step 5 修复**
- fundamentals_pit: consensus_eps vs eps_consensus 命名重复 — 代码已处理, 低优先级
- stock_prices: 非版本化 PIT, 但当前不影响研究 — 推迟
- corporate_actions: 部分 reverse split 缺失 — 推迟

---

## 4. 关键发现与教训（永久保留）

1. **Purge/embargo 揭示假信号** — 60D purge 将 IC 从 0.091 打到 0.038, 58% 是 label leakage
2. **树模型在 60D purge 下崩溃** — XGB/LGB IC 暴跌 80%, 但 1D/5D 短 purge 未测试
3. **analyst_coverage 是有毒特征** — 去掉后 IC 反升 (0.056→0.062), is_missing 编码了 data-vendor bias
4. **IC screening 用错了 labels** — 默认 HORIZON_DAYS=5, 所有"60D screening"实际用 5D labels
5. **G3 proxy ≠ 实际收益** — proxy +35% vs portfolio backtest -9.6%, 差 44 个百分点, 不可互换
6. **IC 是必要非充分条件** — 0.062 是研究 win, monetization 未验证
7. **5D 有弱窗口互补性** — 5D 在 W10/W11 为正 (60D 为负)
8. **Macro interaction 最有效** — 13 个新特征中只有 curve_inverted_x_growth 和 high_vix_x_beta 通过
9. **不同 horizon 需要不同特征和模型** — 用 60D 特征跑 1D/5D 是不公平比较
10. **数据先行** — 数据源扩充后特征/模型需要重跑, 应先完善数据再统一优化

---

## 5. Active Roadmap

> **双轨制**: 部署轨和研究轨并行。不因研究未完成而阻塞部署，不因部署紧迫而跳过研究。

### 部署轨 (本周完成)

| 步骤 | 任务 | 状态 |
|------|------|------|
| D1 | 锁定 V5 no-analyst (19f, 60D Ridge) 为生产 baseline | ✅ bundle 已更新 |
| D2 | 确认 DAG 可正常运行周五信号生成 | ⏳ |
| D3 | 周五灰度正常出信号 | ⏳ |

**部署版本**: V5 no-analyst, IC=0.062, t=2.17, G3 6/6 PASS
**后续**: 研究轨产出更优版本后, 替换 bundle 即可, 无需改 DAG

---

### 研究轨 Step 1: 数据底座完善 + 全 Horizon 优化

| 子步骤 | 任务 | 状态 |
|--------|------|------|
| 1A | 修复 universe_membership (live 路径一致性) | ⏳ |
| 1B | 60D true IC screening (用 60D labels) | ✅ 完成 (76 passed, 35 retained) |
| 1C | 新特征组 1: SEC filing metadata features (8 个) | ⏳ |
| 1D | 新特征组 2: Repackaged event features — insider/short/earnings 重设计 (15 个) | ⏳ |
| 1E | 新特征组 3: ETF sector rotation proxy (5 个) — 需拉取 sector ETF 价格 | ⏳ |
| 1F | **冻结研究数据信封** | ⏳ |
| 1G | 全量 IC screening (60D/20D/10D/5D/1D labels 各跑一次) | ⏳ |
| 1H | Walk-forward 13w (60D/20D/10D/5D/1D 各跑一次, Ridge) | ⏳ |
| 1I | G3 Gate (60D/20D/10D/5D/1D 各跑一次, IC 门槛按 horizon 调整) | ⏳ |
| 1J | 60D model sweep: Ridge/ElasticNet/XGB/LGB (purge=60d) | ⏳ |
| 1K | 20D model sweep: Ridge/ElasticNet/XGB/LGB (purge=20d) | ⏳ |
| 1L | 10D model sweep: Ridge/ElasticNet/XGB/LGB (purge=10d) | ⏳ |
| 1M | 5D model sweep: Ridge/ElasticNet/XGB/LGB (purge=5d) | ⏳ |
| 1N | 1D model sweep: Ridge/ElasticNet/XGB/LGB (purge=1d) | ⏳ |
| 1O | 最优配置锁定 (每个 horizon 的最佳 features × model 组合) | ⏳ |
| 1P | Production-style post-cost backtest (五个 horizon 各一次, score_weighted + turnover controls) | ⏳ |
| 1Q | Cross-horizon post-cost 对比, 决定最终策略组合 | ⏳ |

**1C: SEC filing metadata features (从 sec_filings 表, 不含 NLP):**
- days_since_last_8k, days_since_last_10q, days_since_last_10k
- recent_8k_count_5d / 20d / 60d
- has_recent_8k_5d / 20d
- recent_filing_burst_20d

**1D: Repackaged event features (重新设计为 decayed/abnormal/interaction):**

Insider 系 (从 insider_trades):
- insider_buy_intensity_20d: sum(buy_value × role_weight × exp(-days/20)) / market_cap
- insider_net_intensity_60d: (decayed_buy - decayed_sell) / market_cap
- insider_cluster_buy_30d_w: sum(exp(-days/30)) over distinct buyers, role-weighted
- insider_abnormal_buy_90d: current intensity / trailing_2y_median
- insider_role_skew_30d: (CEO/CFO buy intensity) - (director sell intensity)

Short interest 系 (从 short_interest):
- short_interest_sector_rel: zscore(days_to_cover within sector/date)
- short_interest_change_20d: (dtc_t - dtc_t-20) / abs(dtc_t-20)
- short_interest_abnormal_1y: (dtc - 1y_median) / 1y_std
- short_squeeze_setup: short_sector_rel × max(0, ret_5d) × volume_surge
- crowding_unwind_risk: short_sector_rel × min(0, ret_20d)

Earnings 系 (从 earnings_estimates):
- earnings_surprise_recency_20d: surprise × exp(-days/20)
- earnings_beat_recency_30d: beat_streak × exp(-days/30)
- surprise_flip_qoq: latest_surprise - prior_surprise
- surprise_vs_history: latest_surprise - avg_4q
- pead_setup: surprise_recency × max(0, ret_5d)

**1E: ETF sector rotation proxy:**
需拉取 sector ETF 价格: SPY, QQQ, IWM, XLK, XLF, XLE, XLV, XLI, XLP, XLY, XLU, XLB, XLRE
- sector_rel_ret_5d: ret_5d(sector_etf) - ret_5d(SPY)
- sector_volume_surge: volume(sector_etf) / adv20
- sector_pressure: z(sector_rel_ret) × z(sector_volume_surge)
- stock_vs_sector_20d: ret_20d(stock) - ret_20d(sector_etf)
- sector_pressure_x_divergence: sector_pressure × stock_vs_sector_20d

**1F-extra: 1D 专属特征 (冻结前补充, 需要分钟级数据或日内 OHLC):**
- intraday_range: (high - low) / close
- close_to_high: (close - low) / (high - low)
- opening_gap_reversal: -sign(open - prev_close) × (close - open) / prev_close
- overnight_return: (open - prev_close) / prev_close
- volume_first_hour_ratio: (如果有分钟数据) 首小时成交量 / 全天成交量
- 注: 以上均可从日频 OHLCV 计算 (除 volume_first_hour_ratio 需分钟数据, 可选)

**冻结范围 (1F):**
- 数据表: stock_prices, fundamentals_pit, earnings_estimates, short_interest, insider_trades, sec_filings, macro_series_pit, ETF prices
- 排除: analyst_estimates (forward-only), news_sentiment (覆盖不足)
- 特征定义: ~240 候选 (现 210 + SEC 8 + repackaged 15 + ETF 5 + 1D专属 ~5, 去重后)
- Universe: stock_prices 中所有 non-SPY tickers
- Labels: 1D/5D/10D/20D/60D forward excess return (vs SPY)
- 成本: Almgren-Chriss default parameters

**IC 目标 (base case):** 60D IC 0.066-0.072

**G3 Gate horizon-specific 门槛:**

| Gate | 60D | 20D | 10D | 5D | 1D |
|------|-----|-----|-----|-----|-----|
| OOS IC > threshold | > 0.03 | > 0.02 | > 0.015 | > 0.01 | > 0.005 |
| IC t-test p < 0.05 | 同 | 同 | 同 | 同 | 同 |
| Bootstrap CI > 0 | 同 | 同 | 同 | 同 | 同 |
| DSR significant | 同 | 同 | 同 | 同 | 同 |
| Cost-adjusted excess > 5% | 同 | 成本略高 | 成本更高 | 成本更高 | 成本最高 |
| SPA vs null | 同 | 同 | 同 | 同 | 同 |

**注意:** 不同 horizon 的 raw IC 不可直接比较。1D IC=0.02 和 60D IC=0.06 可能经济价值相当。最终比较以 post-cost net excess 为准。

**Step 1 Gate:**
- GO: 至少一个 horizon annualized net excess > 5% → Step 2
- CONDITIONAL: 最优 horizon 0~5% → 优化 portfolio construction
- NO-GO: 全部 < 0% → 停止 alpha 优化, 先解决变现层

**Ridge retained 特征数控制:** 基于 V5 35f vs 19f 的教训, Ridge 模型保留特征数应控制在 **20-25 个以内**。更多特征会稀释强信号。树模型不受此限制。

### Step 2: Cross-Horizon 融合

- 60D core + 5D/1D overlay
- Validation-IC gating: val IC < 0 时降低暴露
- Regime-aware 动态调整, 不是固定权重
- Production-style post-cost backtest

**Gate:**
- GO: 融合 post-cost > 单 horizon 最优 → Step 4
- NO-GO: 融合无增益 → 锁定最优单 horizon, 进 Step 4

### Step 3: 扩展数据 + 标的池（按需, 每个新增必须改善 post-cost）

**3A: FMP Analyst Grades (免费, 优先)**
- 数据: FMP `/stable/grades` — 评级升降级历史 (2012+, ~2500 条/ticker)
- 特征: upgrade_count_20d, downgrade_count_20d, net_revision_20d, revision_momentum_60d, days_since_last_upgrade/downgrade, analyst_attention_burst, revision_consensus_shift (~8 个)
- 目的: 低成本验证 analyst revision signal 的 IC 贡献
- Gate: 如果 IC 提升 > +0.003 → 考虑升级到完整 I/B/E/S

**3B: 完整 Analyst Revision History (付费, 视 3A 结果)**
- 如果 FMP grades IC 贡献显著 → 购买 Zacks ZEEH ($2-8K/年) 或 Refinitiv I/B/E/S ($10-50K/年)
- 月度/日频 EPS/Revenue consensus 修改历史
- 预期 IC 从 FMP grades 的 +0.002~0.006 提升到 +0.006~0.020
- WRDS 不可用 (仅限学术非商业用途)

**3C: 其他数据扩展**
- 期权 IV (Polygon, BS 反推)
- Russell 1000 标的池扩展 (500→1000)
- NLP 新闻情感 (FinBERT + 历史语料)
- 扩展后重跑受影响的 screening + walk-forward

**Gate:**
- GO: post-cost improvement → Step 4
- NO-GO: 无经济提升 → 直接 Step 4

### Step 4: 工程化 + 产品化（Step 1-3 研究定型后才开始）

- feature_store / parquet parity 修复
- Production-style 回测系统 (净值曲线, Sharpe, 最大回撤, 月度归因)
- Live pipeline 更新 + fusion bundle 更新
- 灰度切换到最终 baseline
- Auto-retrain DAG + drift monitoring (PSI)
- Champion/Challenger 自动切换
- 前端展示更新
- DAG 完善 (日频/周频信号管道, 新数据源增量更新)

---

## 6. 技术架构

```
src/
├── data/          # 数据源适配器 + PIT 查询 + 数据质量
├── features/      # technical(37) + fundamental(17) + macro(10) + alternative(15) + composite(26)
├── labels/        # 1D/5D/10D/20D/60D forward excess return
├── models/        # Ridge + XGBoost + LightGBM + ElasticNet
├── backtest/      # Walk-forward engine + Almgren-Chriss + 执行模拟
├── portfolio/     # equal_weight + vol_weighted + black_litterman + score_weighted
├── stats/         # IC t-test + Bootstrap + DSR + SPA
├── risk/          # 四层风控
├── api/           # FastAPI backend (Step 5)
└── tasks/         # Celery 异步任务 (Step 5)

scripts/
├── run_ic_screening.py            # IC 筛选 (--horizon 控制 label horizon)
├── run_walkforward_comparison.py  # Walk-forward 验证 (13 windows)
├── run_g3_gate.py                 # G3 Gate 6 项检验
├── run_post_cost_comparison.py    # Post-cost 对比
├── run_quintile_stats.py          # Quintile 统计
└── backfill_earnings_and_news.py  # 数据回填 (--incremental)
```

### 关键 Artifacts

| Artifact | Path |
|----------|------|
| Production bundle | data/models/fusion_model_bundle_60d.json |
| 60D walk-forward (no-analyst) | data/reports/walkforward_comparison_60d_v5_no_analyst.json |
| 60D G3 Gate | data/reports/g3_gate_v5_no_analyst.json |
| Post-cost comparison | data/reports/post_cost_comparison.json |
| IC screening (5D labels) | data/features/ic_screening_report_v5_60d.csv |
| IC screening (60D labels) | data/features/ic_screening_report_v5_60d_true.csv |

---

## 7. 部署状态

| 组件 | 状态 | 依赖 |
|------|------|------|
| TimescaleDB + Redis + MLflow | ✅ 运行中 | — |
| Airflow DAG | ✅ 基本可用 | Step 5 完善 |
| 灰度系统 | ✅ 运行中 | Step 5 切换 |
| 前端 Dashboard | ✅ W22 完成 | Step 5 更新 |
| 回测系统 (净值曲线) | ⏳ | Step 5 |
| feature_store parity | ⏳ | Step 5 |
| MLOps (auto-retrain, drift) | ⏳ | Step 5 |

---

## 8. 历史归档

Phase 1/2 周度记录, S1 Phase A-E 实验日志, S2 Stage 1 Batch 记录:
- `docs/archive/IMPLEMENTATION_PLAN_v3_archived_20260416.md`
