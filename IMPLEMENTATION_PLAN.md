# QuantEdge 实施计划 v5.1

**更新日期:** 2026-04-17
**方法论来源:** plan.txt (规范化研究协议) + GPT Pro (4 层决策 + Massive/FMP 深度利用)
**核心理念:**
- V5 no-analyst 保持 live baseline (IC=0.062, 6/6 PASS)
- 四层决策系统: 60D core + 20D adjustment + 5D event/micro overlay + 1D execution throttle
- 不扩付费 vendor，补免费公开数据 (FINRA + FTD 立即加，13F 下阶段)
- 吃满 Massive + FMP 现有订阅的边际价值
- V7 必须 post-cost > V5 才能替换

---

## 1. 项目定位

QuantEdge 是研究驱动的机构级美股量化系统。核心原则:
- **先证能赚钱，再做产品** — Alpha → 经济可行性 → 产品化
- **IC 是必要非充分条件** — 最终以 post-cost net excess 为准
- **Family → Model → Portfolio** 严格实验顺序
- **研究定型后才工程化** — 避免返工

---

## 2. 四层决策系统

| 层 | Horizon | 职责 | 主特征 | 主模型 | 产出 |
|----|---------|------|--------|--------|------|
| **A** | **60D** | Core alpha — 中期主排序 | quality / profitability / value / residual momentum / macro interaction / analyst proxy | Ridge / ElasticNet | core ranking |
| **B** | **20D** | Revision & Crowding Adjustment | analyst proxy / price target drift / earnings drift / short-interest dynamics / sector diffusion | Ridge / ElasticNet + shallow tree challenger | ranking adjustment |
| **C** | **5D** | Event & Microstructure Overlay | gap / fill / opening range / late-day pressure / minute vol / trade imbalance proxy / filing burst / earnings recency | Ridge baseline + shallow LightGBM/XGBoost + ranking | overlay score |
| **D** | **1D** | Execution / Risk Throttle (非主 alpha) | VIX state / opening shock / intraday vol / liquidity stress / event-day tag | rule-based + small classifier | normal/cautious/defensive 状态 |

**关键**：1D 不追求主 alpha，是风控/执行层。60D/20D 是主盈利线，5D 是事件增强层。

---

## 3. 当前 Baseline (V5 no-analyst)

| 指标 | 值 |
|------|-----|
| 配置 | 60D Ridge, 19 features (no-analyst), 13 windows |
| IC | 0.062 |
| t-stat | 2.17 (p=0.025, one-sided) |
| G3 Gate | 6/6 PASS |
| 经济可行性 | **production-style backtest 未做, 真实 net excess 未知** |

**V5 = live baseline。所有 V7 改进必须在此基础上增量提升并通过 post-cost truth table。**

---

## 4. 数据层现状与扩展

### 已有订阅（吃满边际价值）

**Massive Stocks Developer (Polygon)**:
- 10 年 day aggregates ✅ 已用
- 10 年 minute aggregates ❌ **未用**
- Second aggregates ❌ 未用
- Trades (tick) ❌ 未用（flat files）
- Snapshots / websocket
- Corporate actions / reference

**FMP Premium**:
- fundamentals ✅
- earnings actual/estimate ✅
- insider_trades ✅
- SEC filings metadata ✅
- historical stock grades ❌ **未用**
- price target summary/consensus ❌ **未用**
- ratings-historical ❌ **未用**
- earnings calendar ❌ **未用**

### 立刻补的免费公开数据

| 优先级 | 数据源 | 覆盖 | 主要用途 |
|--------|--------|------|---------|
| 1 | **FINRA Daily Short Sale Volume** | 2018-08+ | short flow proxy / crowding (1D/5D/20D) |
| 2 | **SEC FTD (Fails-to-Deliver)** | 2004+ | 结算压力 / ftd_shock (5D/20D/60D) |
| 3 | **SEC 13F Holdings** | 2013-07+ | ownership/network (20D/60D, 下阶段) |

### 暂缓的付费数据

**不买**：LSEG I/B/E/S、OptionMetrics IvyDB、S&P Securities Finance、NYSE Daily TAQ。等 Massive+FMP+FINRA/FTD 全部吃透后再评估。

### 已知缺陷
- analyst_estimates (FMP) forward-only, 无历史快照
- news_sentiment 覆盖不足, 不可回测
- stock_prices 非版本化 PIT
- feature_store vs parquet 特征数不一致

---

## 5. Family Registry (9 类固定)

1. `price_momentum`
2. `liquidity_microstructure`
3. `fundamental_quality`
4. `analyst_expectations_proxy`
5. `shorting_crowding`
6. `macro_regime`
7. `event_earnings_sec`
8. `sector_network_diffusion`
9. `execution_risk`

**每个 horizon 必做 2 个实验**：
- `only_one_family` — 单独上某 family 的独立增量
- `leave_one_family_out` — 去掉某 family 损失多少

---

## 6. 关键发现与教训（永久保留）

1. Purge/embargo 揭示 58% IC 是 label leakage
2. 树模型在 60D purge 下崩溃 80%；1D/5D 短 purge 可能不同
3. analyst_coverage 去掉后 IC 反升 — is_missing 编码 data-vendor bias
4. IC screening 默认 HORIZON_DAYS=5 — 必须每个 horizon 独立
5. G3 proxy (+35%) ≠ portfolio backtest (-9.6%) — 需 production-style truth table
6. 不同 horizon 需要不同特征 + 不同模型
7. V6 加 28 特征后 IC 反而下降 — 更多特征 ≠ 更好，需 family ablation
8. Ridge retained 控制 20-25 个，避免弱特征稀释
9. 多 Horizon 融合不是信号加权，是决策层融合
10. Massive minute/trades 数据之前被完全忽略
11. 1D 应定位为执行/风控层，不追主 alpha
12. 统一评判必须在 post-cost truth table

---

## 7. 12 周执行排期

### 部署轨（并行，不中断）
- V5 no-analyst bundle live
- weekly_signal_pipeline DAG 每周五自动跑
- 灰度持续运行，作为 V7 对比基线

---

### Week 1: 协议冻结 + 当前真相统一

**目标**：消除"多版本真相"，建立单一 champion 来源。

**任务**：
- 建 `configs/research/current_state.yaml` (唯一当前 champion 来源)
- 建 `configs/research/horizon_registry.yaml` (1D/5D/20D/60D 的 label/purge/window/cost)
- 建 `configs/research/family_registry.yaml` (9 个 family)
- 建 `configs/research/champion_registry.yaml`
- 生成 `data/reports/research_manifest_YYYYMMDD.json`

**Gate**: 任何人只看一个文件就知道当前 champion

---

### Week 2: 数据真值与特征一致性审计 [✅ DONE 2026-04-17 — audit + 2.5 修复全部完成]

**目标**：研究数据底座清干净，再扩新特征。

**任务**：
- [✅ 2026-04-17 commit 07330e7] `feature_store` (176) vs `parquet` (238) parity audit
- [✅ 2026-04-17 commit 07330e7] `stock_prices` split/adjustment/PIT 规则审计
- [✅ 2026-04-17 commit 07330e7] benchmark 与 SPY 起始日审计
- [✅ 2026-04-17 commit 07330e7] active universe 规则固定 (audit 产出, universe 历史回填在 P3)
- [ ] 对齐 Massive day aggregates 与 stock_prices (Week 3 做)

**产出**：
- [✅] `data/reports/feature_parity_audit_20260417.json`
- [✅] `data/reports/price_truth_audit_20260417.json`
- [✅] `data/reports/universe_audit_20260417.json`
- [✅] `data/reports/benchmark_audit_20260417.json`

**Gate**: 训练与服务特征定义统一，日频价格真值可回放 — **✅ 通过** (Week 2.5 P0-P3 全部完成)

---

### Week 2.5: 审计问题修复 (Gate 前必做)

**目标**: Week 2 审计暴露的 4 类结构性问题必须修复才能通过 Gate 进 Week 3。

**P0 止血** [✅ 2026-04-17 commit db3586c]
- Live bundle fail-closed 护栏: BundleValidator + schema check + SHA256 指纹封印
- 3 个 live 入口 (run_live_pipeline / run_greyscale_live / DAG validate_bundle_schema) 均 fail-closed
- 结果: bundle 特征缺失时直接 raise, 不再用 NaN 顶上

**P1 数据统一** [✅ 2026-04-17 commit b2e2d28]
- parquet ↔ feature_store single source of truth: `scripts/build_feature_matrix.py` + `prepare_feature_export_frame()`
- 13 个 live 缺失特征补齐 (bundle_missing_feature_count=0)
- DAG update_features_cache 改用统一 exporter
- ProcessPoolExecutor spawn context 修复 DB 连接串扰
- 结果: 50 样本 parquet/store 值 1e-8 内一致, validate_bundle_schema 通过

**P1.5 Audit output contract 测试** [✅ 2026-04-17 commit a5ecbe8]
- 4 个 audit script 的最小 schema 测试 (tests/test_scripts/test_audit_contracts.py)
- 测试调用 build_report() + mock 外部依赖, 不连 DB
- 4/4 pass, JSON 字段漂移会立即失败

**P2 价格真值 + 标签重建** [✅ 2026-04-17 commit 3317a37]
- pit_violations: 2435 → 0
- zero_volume: 407 → 0
- split_anomalies: 20747 → 0 (17 tickers 重拉 Polygon + 123 条由 corporate_actions 解释)
- 1D/5D/10D/20D/60D labels 在修正价格上重建, 加 is_valid_excess + invalid_reason
- pre-SPY excess_return 全部 NULL, 10D null 原因可解释
- V5 60D IC sanity: 0.0623 → 0.0594 (退化 4.6%, 通过 10% tolerance)

**P3 Universe PIT 回填** [✅ 2026-04-17 commit d9dd839]
- Scheme A 完整回填: FMP 503 current + 1279 历史 change events → 706 PIT interval rows
- universe_membership 覆盖 2016-01-01 ~ 2026-04-01 (3144 行, 711 tickers)
- governed universe 下 live=610, research=610, diff=0
- V5 60D IC 不变 (0% degradation)

**Week 2.5 Gate (Week 3 启动前必须全部 pass)** — **✅ 6/6 通过**:
1. ✅ Live 19/19 required features 全部可见 (P1)
2. ✅ parquet/feature_store 同 cutoff 下字段/行/时间戳一致 (P1)
3. ✅ 1D/5D/10D/20D 标签已去 pre-SPY 污染, null excess_return 可解释 (P2)
4. ✅ knowledge_time T+1 违反在训练窗口内清零 (P2, pit_violations=0)
5. ✅ Universe 历史 PIT 回填完成 (P3, Scheme A, 2016-01-01 ~ 2026-04-01)
6. ✅ V5 champion sanity 复跑, 60D IC 不退化 >10% (P2 4.6% + P3 0%)

**→ Week 2 Gate 通过, 可 merge 到 main 并进入 Week 3**

---

### Week 3: Massive Minute Aggregates 入库 [🔄 IN PROGRESS — Week 3.0 smoke + A-plus gate DONE, 待 3.1 全量]

**目标**：5D/1D 获得专属数据层。

**已完成子任务** (branch: feature/s2-v5.1-week3-minute-aggs, PR #2 draft):
- [✅ commit e1b6fbc] Week 3.0 smoke — Polygon minute ingest + stock_minute_aggs hypertable + 3 intraday features, 10 ticker × 5 日 = 19550 行
- [✅ commit 7646295] Week 3.0.5 B-lite 三向对账诊断 — 24 样本, 归因 polygon_daily_vs_minute (vendor 差异, 本地 0bp)
- [✅ commit e33d905] Week 3.0.6 A-plus gate + C-partial 血缘 + minute 内部一致性 — smoke pass=true, 53 warning 落 price_reconciliation_events

**待做子任务**:
- [ ] Week 3.1 全量回填 2019-01 ~ 2026-04 governed universe
- [ ] Week 3.2 首批 9 个 intraday 特征 (补 6 个: open_30m_ret / last_30m_ret / realized_vol_1d / volume_curve_surprise / close_to_vwap / transactions_count_zscore)
- [ ] Week 3.3 Gate 验证 (覆盖率 >95% / minute↔day A-plus / 特征质量三件套)

**任务**：
- 新建 `src/data/polygon_minute.py`
- 新表 `stock_minute_aggs`
- 优先回填 2019+ active universe 的 minute aggregates
- 建 `src/features/intraday.py` (分钟级特征)
- 同步生成 overnight / intraday labels (`labels_intraday`)

**首批新特征**：
- `gap_pct`, `overnight_ret`, `intraday_ret`
- `open_30m_ret`, `last_30m_ret`
- `realized_vol_1d`, `volume_curve_surprise`
- `close_to_vwap`, `transactions_count_zscore`

**新脚本**：
- `scripts/run_intraday_feature_build.py`
- `scripts/run_intraday_label_build.py`

**Gate**: 2019+ active universe minute 覆盖率 > 95%

---

### Week 4: Massive Trades 定向抽样

**目标**：不全量落库 trades，只拿最值钱的 trade-level 信息。

**任务**：
- 定向抽取 trades:
  - top 200 liquidity 股票
  - earnings / SEC / 大 gap 事件窗口
  - W5/W6/W11 弱窗口样本
- 新表 `stock_trades_sampled`
- 构建 tick-rule trade imbalance proxy
- 构建 trade size skew / late-day aggressiveness

**首批新特征**：
- `trade_imbalance_proxy`
- `large_trade_ratio`
- `late_day_aggressiveness`
- `offhours_trade_ratio`

**Gate**: 拿到一套足够用于 5D / 弱窗口诊断的 trade family

---

### Week 5: 免费公开数据 + FMP 新端点

**目标**：补 shorting/crowding 盲区 + 启动 analyst proxy。

**任务 A: FINRA Daily Short Sale Volume**:
- `src/data/finra_short_sale.py`
- 新表 `short_sale_volume_daily`
- 特征: `short_sale_ratio_1d/5d`, `short_sale_accel`, `abnormal_off_exchange_shorting`

**任务 B: SEC FTD**:
- `src/data/sec_ftd.py`
- 新表 `ftd_pit`
- 特征: `ftd_to_float`, `ftd_persistence`, `ftd_shock`

**任务 C: FMP 新端点**:
- `src/data/sources/fmp_grades.py` (historical stock grades)
- `src/data/sources/fmp_price_target.py` (summary + consensus)
- `src/data/sources/fmp_ratings.py` (ratings-historical)
- `src/data/sources/fmp_earnings_calendar.py`

**首批新特征** (analyst proxy):
- `net_grade_change_5d/20d/60d`
- `upgrade_count`, `downgrade_count`
- `consensus_upside`, `target_price_drift`
- `target_dispersion_proxy`, `coverage_change_proxy`
- `financial_health_trend`

**Gate**: analyst_proxy family 至少 8-12 个可回测特征, 缺失率与 lag rule 清楚

---

### Week 6: Family 归一 + 覆盖率诊断

**目标**：每个 horizon 拥有自己的家族配置。

**任务**：
- 所有新特征写入 family registry
- 做 family coverage report
- 做 missingness economic meaning 审计
- 对 `is_missing` 去伪存真，删除 vendor bias 列
- 产出每个 horizon 的候选 family 清单

**产出**：
- `data/reports/family_coverage_report.json`
- `data/reports/missingness_audit.json`

**Gate**: 每个特征唯一 family, 保留的 is_missing 都能解释经济含义

---

### Week 7: Per-horizon IC Screening + Family Ablation

**目标**：1D/5D/20D 不再共用 60D 特征集。

**任务**：
- 分别跑 1D / 5D / 20D / 60D IC screening
- 跑 `only_one_family` 消融
- 跑 `leave_one_family_out` 消融
- 分 horizon 产出 retained 候选集

**复用/新增**：
- `scripts/run_ic_screening.py` (复用, --horizon)
- `scripts/run_family_ablation.py` (**新建**)

**Gate**:
- 每个 horizon 有自己的 top features / top families
- 明确回答"最值钱的前三个 family 是什么"

---

### Week 8: Model Sweep (baseline → challenger → ranking)

**目标**：模型选择服务 horizon，不一视同仁。

**任务**：
- 全 horizon 跑 Ridge / ElasticNet
- 5D / 20D 跑 shallow LightGBM / XGBoost
- 60D 只让树模型做 challenger
- 加入 top-decile weighted loss / pairwise ranker prototype
- 比较回归目标 vs 排序目标

**新脚本**：
- `scripts/run_ranker_comparison.py`

**Gate**:
- 至少 2 个 horizon 出现"模型提升真实非噪声"证据
- 排序目标胜出则进 Week 9 主线

---

### Week 9: Walk-forward + 弱窗口 + Regime 诊断

**目标**：新增 family/模型是否穿过 OOS 稳定性。

**任务**：
- 13-window walk-forward × 4 horizons
- 专门复查 W5/W6/W10/W11
- Regime analysis 重跑
- 区分"不可预测 regime break" vs "可 gating 窗口"

**Gate**:
- 至少一个 horizon 新版本 OOS IC 显著优于旧 baseline
- W6/W11 可 gating 窗口有明确先验特征

---

### Week 10: Production-style Post-cost Truth Table

**目标**：结束"研究赢了但交易没赢"的模糊阶段。

**任务** — 所有 horizon/模型进入同一 truth table，比较：
- `equal_weight_top_decile`
- `score_weighted`
- `score_weighted + buffer`
- `hold-and-trim`
- `sector/beta-neutral constrained optimizer`

优化 turnover budget, 统一输出 gross/cost_drag/net/DD/participation。

**Gate**:
- 至少一个 horizon production-style 成本后年化净超额 > 5%
- 未达则不得进 fusion / deployment 提级

---

### Week 11: 多层决策融合 + Risk Throttle

**目标**：决策层融合，不是信号平均。

**任务**：
- 固定单 horizon champions
- 结构化融合:
  - 60D core ranking (Layer A)
  - 20D ranking adjustment (Layer B)
  - 5D event overlay (Layer C)
  - 1D execution throttle (Layer D)
- 引入 `normal / cautious / defensive` 三档状态
- validation-IC gating + market-state gating 合并

**Gate**:
- Fusion 净表现优于单一 60D champion
- 或至少显著改善回撤/弱窗口
- 简单平均若胜出 → 架构设计有问题, 回退

---

### Week 12: 灰度/Shadow/Live Validation + 下一阶段决策

**目标**：研究结果收束为可执行部署方案。

**任务**：
- 更新 champion bundle
- Shadow mode / greyscale 验证
- Live IC consistency 检查
- 回滚规则 + drift rules 写入配置
- 评估是否需要进入下一阶段采购

**Gate**:
- 单一 champion + 单一 rollback 逻辑
- "下一阶段该不该买新数据"明确判断

---

## 8. 付费采购决策 (12 周后评估)

**只有同时满足 2 条才考虑扩数据预算：**
1. Massive+FMP+FINRA+FTD 跑完仍无法让 production-style net excess 拉正
2. 20D/60D 已经很强，但明显缺 analyst revision/options/borrow
3. 5D minute/trade proxy 已有价值，但受 quote/spread/BBO 缺失限制
4. 研究纪律和 post-cost pipeline 已成熟

**采购顺序**：
1. LSEG I/B/E/S（$10-50K/年）
2. OptionMetrics IvyDB（$5-20K/年）
3. S&P Securities Finance
4. NYSE Daily TAQ（仅 1D 成为核心时）

---

## 9. 明确暂缓（现在不做）

- ❌ 全量 Massive trades lake 建设
- ❌ 全 depth microstructure / quote imbalance 研究
- ❌ Transcript NLP 大工程
- ❌ LSTM / Transformer 主线化
- ❌ 简单 IC-weighted fusion 作为正式主线
- ❌ 再买昂贵数据订阅

---

## 10. 运行节奏

**每日**:
- Massive day/minute 增量同步
- FMP insider/SEC/grades/price target 增量同步
- Data quality checks

**每周三**: 特征覆盖率/缺失率检查 + 弱窗口监控

**每周五收盘 / 周六**: 统一跑 signal generation + walk-forward + live validation + 周报

**每周日**: 研究分支合并 + champion review + retrain/rollback 决策

---

## 11. 成功标准

### 研究层
- 每个 horizon 有专属 top families
- 至少 2 个 horizon OOS IC 明显优于旧 baseline
- 5D 不再依赖 60D 特征集

### 经济层
- 至少一个 horizon production-style 成本后年化净超额 > 5%
- 相比 equal-weight post-cost，净值曲线显著改善
- Turnover 与 capacity 不失控

### 部署层
- 单一 champion 真相
- 单一 rollback rule
- Shadow / greyscale 连续稳定
- Live IC consistency 不恶化

---

## 12. 技术架构

```
src/
├── data/
│   ├── sources/                # FMP (+ grades/price_target/ratings), Polygon, FRED
│   ├── polygon_minute.py       # (Week 3 新) minute aggregates
│   ├── finra_short_sale.py     # (Week 5 新)
│   └── sec_ftd.py              # (Week 5 新)
├── features/
│   ├── technical.py
│   ├── fundamental.py
│   ├── alternative.py          # (已扩展)
│   ├── sector_rotation.py
│   └── intraday.py             # (Week 3 新) 分钟级特征
├── labels/                     # 1D/5D/10D/20D/60D + intraday
├── models/                     # Ridge + ElasticNet + XGBoost + LightGBM + ranker
├── backtest/
├── portfolio/
├── stats/
├── risk/
└── universe/active.py

configs/research/               # (新)
├── current_state.yaml
├── horizon_registry.yaml
├── family_registry.yaml
└── champion_registry.yaml

scripts/
├── run_ic_screening.py
├── run_family_ablation.py      # (Week 7 新)
├── run_walkforward_comparison.py
├── run_g3_gate.py
├── run_intraday_feature_build.py  # (Week 3 新)
├── run_intraday_label_build.py
├── run_ranker_comparison.py    # (Week 8 新)
├── run_portfolio_comparison.py
├── run_portfolio_optimization_comparison.py
├── run_turnover_optimization.py
├── run_regime_analysis.py
├── run_horizon_fusion.py
└── run_post_cost_truth_table.py   # (Week 10 新)

dags/
├── dag_daily_data.py
└── dag_weekly_signal.py

frontend/                       # Phase 5 更新
```

### 关键 Artifacts

| Artifact | Path |
|----------|------|
| Live bundle | data/models/fusion_model_bundle_60d.json (V5 no-analyst) |
| V5 60D walk-forward | data/reports/walkforward_comparison_60d_v5_no_analyst.json |
| V5 60D G3 Gate | data/reports/g3_gate_v5_no_analyst.json |
| V7 per-horizon IC screening | data/features/ic_screening_report_v7_<horizon>.csv (Week 7) |
| Family ablation | data/reports/family_ablation_<horizon>.json (Week 7) |
| Post-cost truth table | data/reports/post_cost_truth_table_<date>.json (Week 10) |

---

## 13. 部署状态

| 组件 | 状态 |
|------|------|
| TimescaleDB + Redis + MLflow | ✅ 运行 |
| Airflow DAG | ✅ weekly_signal 4/17 恢复 |
| 灰度系统 | ✅ week_05 已出 |
| V5 no-analyst bundle | ✅ 锁定 |
| Minute aggregates | ⏳ Week 3 |
| FINRA + FTD | ⏳ Week 5 |
| 回测系统 (净值曲线) | ⏳ Phase 5 (Week 12+) |

---

## 14. 历史归档

- v3 归档: `docs/archive/IMPLEMENTATION_PLAN_v3_archived_20260416.md`
- v4.1 归档: `docs/archive/IMPLEMENTATION_PLAN_v4_archived_20260417.md`
- v5.0 归档: `docs/archive/IMPLEMENTATION_PLAN_v5_0_archived_20260417.md`
- plan.txt: `docs/plan.txt` (v5.0 方法论来源)
- GPT Pro 方案: `docs/QuantEdge_optimized_plan_FMP_Massive_2026-04-17.md` (v5.1 架构来源)

---

## 15. 关键判断总结

**GPT Pro 的核心见解**：
- 四层决策架构 (A/B/C/D)
- Massive minute/trades 被严重低估
- FMP 新端点未用 (grades/price target/ratings)
- 1D 应为执行层，不追主 alpha

**我们之前的核心见解（与 GPT Pro 互补）**：
- Family ablation 方法论
- 双轨: V5 live 不中断
- 免费公开数据 (FINRA/FTD/13F) 应补

**最终共识**：
- **不扩付费 vendor**（GPT Pro 对）
- **补免费公开数据**（我们对）
- **吃满现有 Massive + FMP 订阅**（GPT Pro 强调）
- **12 周 week-by-week 执行**（GPT Pro 排期）
- **FINRA 优先 > FTD > 13F**（GPT Pro 判断）

这个 v5.1 是研究和工程层双方 AI 充分讨论后的最终方案。
