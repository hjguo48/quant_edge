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

### Week 3: Massive Minute Aggregates 入库 [✅ DONE 2026-04-21 — merged to main via PR #2 `5e09410`]

**目标**：5D/1D 获得专属数据层。

**已完成子任务** (branch: feature/s2-v5.1-week3-minute-aggs → main):
- [✅ e1b6fbc] 3.0 smoke — Polygon minute ingest + stock_minute_aggs hypertable + 3 intraday features (10 ticker × 5 日)
- [✅ 7646295] 3.0.5 B-lite 三向对账诊断 — 24 样本, 归因 polygon_daily_vs_minute vendor 差异
- [✅ e33d905] 3.0.6 A-plus gate + C-partial 血缘 + minute 内部一致性
- [✅ 0180812] 3.0.7 P1/P2 hotfix — t=16:00 post-close 修正, close bp 13.63→6.87
- [✅ 63ec8e2 + 后续] 3.1 Polygon flat files 客户端 + migration 005 (compression + state table) + backfill runner
- [✅ 2026-04-21] **3.1 全量回填 2016-04-20 → 2026-04-17** — 2513 sessions 100% 覆盖, **553 M minute rows**
- [✅ c220dfb] 3.2 补 6 个 intraday 特征 (共 9 个: gap_pct / overnight_ret / intraday_ret / open_30m_ret / last_30m_ret / realized_vol_1d / volume_curve_surprise / close_to_vwap / transactions_count_zscore)
- [✅ bd3decb] 3.A dag_daily_data minute_incremental TaskGroup (ENABLE_MINUTE_INCREMENTAL=false 默认, Step A 完成)
- [✅ c50430c] 3.3.1 历史 intraday 特征构建 — **11.5 M rows**, 10 年 × 9 特征
- [✅ c96d8b2] 3.3.2 Gate 验证三件套全绿 (覆盖率 100% / A-plus 0 blocker / missing<20% outlier<1%)
- [✅ 1029650] 3.3.3 V5 sanity PASS — 60D IC=0.0594, **0% 退化**
- [✅ 5e09410] PR #2 merged to main (2026-04-21)

**Gate**: 2019+ active universe minute 覆盖率 > 95% — **实际 2016-04+ 全覆盖 100%, 超额达成** ✅

**Week 3 期间 bonus 修复** (未在原 plan 范围但实际做了):
- Week 2.5-P3 universe_membership 真实 PIT bug (FMP 数据不对称, 改 Wikipedia fallback) — commit 5b0fb43
- 2016-04-18 单日 Polygon 10-year rolling cutoff → skipped_holiday
- Codex review 19 轮 / 40+ P1&P2 修复 / +64 tests

**Week 3 累计统计**:
- Minute 数据: 553 M rows (stock_minute_aggs, TimescaleDB hypertable + compression)
- Intraday 特征: 11.5 M rows (feature_store, 9 features × 10 年)
- Universe: 3150 rows (universe_membership, 2016-01 ~ 2026-04, 真实 PIT)
- 代码: 20+ commits, 40+ bug fix, +64 tests
- 耗时: 3 天 (2026-04-18 ~ 2026-04-21)

---

### Week 3 DAG 集成分层 (Step A/B/C, 严格分阶段不混淆)

**原则**: 数据层连续性 ≠ 研究生产化. V5 live baseline 不被新 minute 链路干扰 (双轨).

#### Step A — 现在做 (Week 3 当下, 3.1 并行): minute_incremental 最小扩展
在 `dag_daily_data.py` 加 TaskGroup:
```
minute_incremental:
  ├─ resolve_minute_dates_to_sync
  ├─ sync_polygon_minute_incremental  (Polygon flat file, T-1/T)
  ├─ validate_minute_internal_quality  (gap/overlap/monotonic)
  ├─ validate_minute_day_reconciliation_aplus  (OHL<10bp blocker, close/vol warning)
  └─ publish_minute_watermark  (写 minute_backfill_state)
```

- 挂法: 放在 day 数据同步之后、最终 data quality 结果之前
- **trigger_rule=ALL_DONE**: minute 链路失败不 block V5 weekly signal
- **不挂默认成功路径** (feature flag 或 downstream_on_failure=none)
- 只做原始 minute 数据增量同步 + QC, 不跑 feature/label build

#### Step B — 3.3 Gate 通过后启用默认调度
条件:
- minute 覆盖率达到 >95% pass 条件
- 连续 5 个交易日 A-plus gate blocker 全绿
- 三件套阈值稳定 (missing < X%, outlier rate 稳定, lag 规则明确)

此时把 `minute_incremental` 挂进 dag_daily_data 默认成功路径 (移除 feature flag).

#### Step C — Week 7 后评估 intraday feature/label DAG 化
条件:
- per-horizon IC screening 证明 intraday family 对 1D/5D 有独立增量
- family ablation 显示 minute-derived 特征可独立存在

此时才考虑把 `run_intraday_feature_build.py` + `run_intraday_label_build.py` 挂进 DAG. 在此之前保持脚本态.

**禁止事项** (避免 scope creep):
- 不要把 Week 3.1 历史回填挂进 dag_daily_data (保持独立批处理脚本)
- 不要让 minute 链路失败 block weekly_signal_pipeline
- 不要在 Step A 阶段就挂 intraday feature/label build 到默认成功路径
- 不要在 IC screening (Week 7) 前工程化 minute family

---

### Week 4: Massive Trades 定向抽样 [✅ DONE 2026-04-24 — Gate FAIL 但有核心发现]

**目标 (完成情况)**: 不全量落库 trades, 定向抽 S1 弱窗口 (W5/W10/W11), 测 trade-level microstructure 特征是否有 alpha.

**最终 Gate 结果**: FAIL (|IC| + t-stat 联合判据未过), 但 **1 个特征 `offhours_trade_ratio` 在 n_windows=3 下 t=2.11 |IC|=0.018 过阈值**, 作为单特征是 statistically significant.

#### 工程交付 (28 commits, ~10,000 LOC, 80+ tests)

**基础设施** (Task 1-10 全部完成):
- `alembic/versions/007_add_stock_trades_sampled.py` — trade hypertable + state 表
- `src/data/polygon_trades.py` — REST `/v3/trades` client (retry + pagination + stable_sequence_fallback hash)
- `src/data/polygon_trades_flat.py` — **Polygon flat files client 新增** (streaming + per-ticker memory isolation)
- `src/data/event_calendar.py` — earnings/gap/weak_window/top_liquidity 事件聚合
- `src/universe/top_liquidity.py` — PIT ADV rank wrapper
- `src/features/trade_microstructure.py` — 5 trade features + PIT split
- `src/features/trade_microstructure_minute_proxy.py` — **3 minute 代理特征** (Week 3 minute_aggs 复用)
- `scripts/build_trades_sample_universe.py` / `run_trades_sampling.py` (含 streaming mode) / `build_trade_microstructure_features.py` / `build_trade_microstructure_flat_features.py` / `build_trade_microstructure_minute_proxy_features.py` / `run_week4_gate_verification.py` / `preflight_trades_estimator.py` / `calibrate_trade_vs_minute_features.py`
- `configs/research/week4_trades_sampling.yaml` — 完整 Week4 config

**Bug 修复 (跨项目受益)**:
- 🔥 `universe_membership` DELETE predicate bug — monthly sync 误删历史 anchor, 影响**整个 Phase 1 所有历史回测** (commit `100d2c8` merged to main)
- Flat-files trades memory leak: 7.5 GB → 1.7 GB (per-ticker isolation)
- `iter_body_chunks` retry scope bug (大文件下载中途断连不重试)
- `off_exchange_volume_ratio` parser bug (`trf_id=0` 被当非空 → 全 1.0)

#### 特征产出 (data 保留在本地 parquet, gitignored)

**Tier 1 — 可用 (进 V5 bundle 候选, default-off)**:
- 🌟 **`offhours_trade_ratio` (flat-files trade-level)**: W5+W10+W11 pooled IC=+0.027, n_windows=3 下 t=2.11 sign=100% 一致 — 统计显著
- **`late_day_aggressiveness` (minute-proxy)**: t=4.56 11/11 sign 全正 |IC|=0.014, Week 7 per-horizon 组合候选
- **`trade_imbalance_proxy` (minute-proxy)**: t=3.01 10/11 sign 全负 |IC|=0.013, short signal 候选

**Tier 3 — 条件化 Phase 2 (regime-gated)**:
- `large_trade_ratio`: W5=-0.031 vs W10=+0.039 符号反转, 跨 regime 不稳. 未来加 `regime_flag` 做 conditional feature 可解锁 (~0.05 IC)

**数据 artifacts** (data/features/, gitignored; 重建脚本已 commit):
- `trade_microstructure_flat_W5.parquet` — 18,400 rows (W5 121 + W10 127 + W11 120 sessions)
- `trade_microstructure_minute_proxy.parquet` — 68,900 rows (2019-09 → 2025-02, 1378 sessions)
- `forward_excess_return_5d_labels.parquet` — 845,939 labels

**Gate 报告** (data/reports/week4/):
- gate_summary_minute_proxy.json + 5 个 flat-files variants (n_windows=3/5/6/11 × W5 / W5W10W11)
- calibration_trade_vs_minute.csv — 10-sample ratio 分析
- preflight_estimate.json — 预算估算

#### 研究发现 (即使 Gate FAIL 也有独立价值)

1. **盘外活动 (offhours) 是最稳定的微结构信号** — 跨制度 (W5 熊市 + W10 牛市 + W11 大选后) 同号, magnitude 一致
2. **Large trade signal 跨 regime 符号反转** — 下跌期 (W5) 负相关, 上涨期 (W10) 正相关. Phase 2 conditional feature 方向
3. **Minute-proxy 和 trade-level 捕不同维度** — 50% calibration samples 符号不一致, 说明两者独立
4. **Gate 3 的 n_windows 敏感** — 11 windows 噪声大, 3 windows (每 S1 弱窗口一个 slice) 能暴露真实统计显著性

#### 关键 plan 偏离 (诚实记录)

- 原 plan scope: "3 weak windows + earnings + gap 全 Gate PASS". 实际: W5+W10+W11 scope + Gate FAIL (仅 1 特征过阈值, 要求 2)
- SEC event window 延 Week 5 (一致按 plan 延期)
- `stock_trades_sampled` DB schema 建成但空 (streaming mode 不入库, Phase 2 productionize 时启用)
- `trade_microstructure` family 注册但 default-off, 需显式 opt-in

#### 3 个候选特征 (修正 framing, 按 Codex 审核) — **"promising candidates", 非 Gate-adopted**

| 特征 | 状态 | Gate 判据 |
|---|---|---|
| `offhours_trade_ratio` (flat) | 3-window 下 |IC|=0.018 > 0.015 AND t=2.11 > 2.0 AND sign 2/2 ✅ | **但 sign_consistent_windows_min=7 硬阈值使其在 3-window 下结构性 fail**. 单特征显著, 未达 Gate 2/5 要求 |
| `late_day_aggressiveness` (minute-proxy) | t=4.56, 11/11 sign | **|IC|=0.0140 < 0.015 硬阈值** → Gate miss |
| `trade_imbalance_proxy` (minute-proxy) | t=3.01, 10/11 sign | **|IC|=0.0134 < 0.015 硬阈值** → Gate miss |

**诚实评估**: 3 特征**均未通过 Gate 3 当前阈值**. 属 "promising candidates", Week 7 per-horizon screening / Phase 2 feature selection 时可**组合 / opt-in** 测试, **不直接入 V5 bundle**.

#### Codex review findings (高/中 优先级 post-merge TODOs)

- **High (framing)**: 上文已修正, "Tier 1 usable" → "promising candidates below Gate threshold"
- **Medium (gate scope)**: Coverage Gate 使用全 `trades_sampling_state` 表, 不随 features parquet scope 变化. 子 scope 评估 Coverage Gate 失真. **TODO**: 改 Coverage scope 按 evaluated artifact 过滤日期/ticker, 或加 `--state-scope` CLI flag.
- **Medium (streaming resume)**: `scripts/run_trades_sampling.py` streaming-mode checkpoint 不验 `run_config_hash`. 改 config 后 `--resume` 可能混入不兼容 rows. **TODO**: 和 Task 8 builder 一致地 hash-aware (mismatch → 全量 recompute).

两个 medium 都非 blocker, 作为 Phase 2 productionize 前修复.

---

### Week 5: 免费公开数据 + FMP 新端点 **[✅ DONE 2026-04-25 PR #4 + PR #5 merged]**

**目标**：补 shorting/crowding 盲区 + 启动 analyst proxy。

**已交付 (Tranche A, PR #4 + PR #5, 23 commits, ~+8000 lines)**:
- ✅ Task 0-8: base delivery (PR #4, commit `1a61464`)
- ✅ Post-merge fixes (PR #5, commit `5c91e75`) — 2 轮 Codex 严格 review 修 9 处 bug (FMP endpoint schema drift / FINRA market column / test fixture TRUNCATE / PIT alignment / pagination safety / firm-day dedupe)

**Gate verification 最终结果** (2026-04-25, sample 30 tickers × 20 dates, 2025-10 → 2026-04):
- **Missing rate: 12/14 PASS**
  - ✅ shorting 3/4: short_sale_ratio_1d/5d/accel (0.5-0.8% missing)
  - ✅ analyst 9/10: net_grade_change × 3 horizons (0%), upgrade/downgrade_count (0%), consensus_upside (4.5%), target_dispersion (30%), coverage_change (11.8%), financial_health_trend (2%)
  - ❌ abnormal_off_exchange_shorting 100%: **FINRA CDN 对 ADF/BNY 文件返 403 AccessDenied**, Polygon Massive + FMP Premium 均无替代, 需付费 FINRA Industry Data Services
  - ❌ target_price_drift 62%: **真实数据稀疏** — mid/低覆盖 ticker 60d 内 <5 个不同 analyst 发报日, 非 bug. Week 7 IC screening 再决定调参或弃用
- **Lag rule: PASS** ✅
- **Coverage: 4/5** (earnings_calendar 24% — FMP 402 blocks pre-2022 + threshold 95% 过严, Tranche A 14 特征不依赖)
- **Source integrity: 3/4 populate 100%** (grades / price_target / ratings), earnings 0.14% 因大部分 future events 无 eps_actual

**数据规模** (2026-04-25 merged 状态):
- short_sale_volume_daily: 13.46M rows (2021+, 仅 CNMS, ADF/BNY 被 FINRA 封禁)
- grades_events: 129K / 502 tickers / 2012+
- ratings_events: 980K / 503 tickers / 2018+ 日频
- price_target_events: 32K (499 consensus + 31.5K per-analyst via /stable/price-target-news)
- earnings_calendar: 4.6K (2022+, SP500 仅)

**Follow-up (non-blocking, 独立 PR 处理)**:
- Gate source_integrity tests 非 hermetic (populated dev DB 有 2 failure), 需 schema-per-run 隔离
- Earnings coverage threshold 过严, 需调低 (FMP 季度频率合理)
- Gate 性能: missing_rate gate --sample-ratio 参数已加, 避免全量 5.8M 循环

**Tranche B (SEC FTD, 未做)**: 视 Phase E (Week 7 IC screening) 信号决定, 若 Tranche A 12 特征信号弱则做 B 补 `ftd_shock` 替代 abnormal_off_exchange_shorting

**Tranche B (未做, 视后续 Gate 决定)**:
- SEC FTD — `src/data/sec_ftd.py`, 新表 `ftd_pit`, 特征 `ftd_to_float` / `ftd_persistence` / `ftd_shock`

**14 特征 default OFF**: Week 5 新特征需通过 Gate 验证 + S1 walk-forward 证明独立 IC 贡献后, 才 flip default enable。当前默认 OFF 保护 S1 v5 既有 38 特征集。

---

### Week 6: Family 归一 + 覆盖率诊断 **[✅ DONE 2026-04-25 PR #6 merged]**

**目标**：每个 horizon 拥有自己的家族配置。

**已交付 (PR #6, 2 commits, +3500 lines)**:
- ✅ Family coverage report — `scripts/run_family_coverage_report.py` (sample 20 tickers × 12 months)
- ✅ Missingness audit — `scripts/run_missingness_audit.py` (161 features 分类)
- ✅ Horizon families config — `configs/research/horizon_families.yaml` (1D/5D/20D/60D × 包含/排除 families)
- ✅ 共用 helper `scripts/_week6_family_utils.py` (348 LOC)
- ✅ Codex strict review 修 3 critical bugs:
  - C1: trade_microstructure 5 features (intraday flag-off) 重分类为 `sample_disabled_pipeline` 而非 vendor_bias
  - C2: high_vix_x_beta composite 加显式 input-validity 检查 (vix_z + stock_beta_252)
  - C3: summary 拆分为 `vendor_bias`/`data_source_block`/`sample_disabled_pipeline` 三个独立 keys

**Audit 最终结果** (sample 20 tickers × 12 months):
- keep: 142 features (88%)
- convert_to_imputed: 12 features (sparse but real)
- data_source_block: 1 (`abnormal_off_exchange_shorting` — FINRA ADF 封禁)
- sample_disabled_pipeline: 5 (`trade_microstructure` family, intraday data 路径关闭)
- vendor_bias: 0 (genuinely zero, 之前 review 找的 7 个都是 false positive)

**产出**:
- `data/reports/family_coverage_report.json` (15 families × monthly coverage)
- `data/reports/missingness_audit.json` (summary-only, 29 lines)
- `data/reports/missingness_audit_full.json` (per-feature 详细, gitignored)

**Gate**: ✅ 每特征唯一 family (15 families × 161 features), 所有 keep 的 is_missing 都有 family-level rationale 解释

---

### Week 7: Per-horizon IC Screening + Family Ablation

**目标**：1D/5D/20D 不再共用 60D 特征集。

**任务**：
- 分别跑 1D / 5D / 20D / 60D IC screening
- 跑 `only_one_family` 消融
- 跑 `leave_one_family_out` 消融
- 分 horizon 产出 retained 候选集

**预处理 TODO (Week 7 开前需处理)**:
- [ ] `scripts/run_ic_screening.py` 加 `--allow-missing-intraday` CLI flag, `build_or_load_feature_cache` 传透; 默认 False (研究 fail-closed), 显式 opt-in tolerant. 否则默认 2016-03~2025-06 范围会因 minute coverage gap 直接 raise IntradayHistoryError.

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

### Week 12: 灰度/Shadow/Live Validation + 下一阶段决策 + Production Hardening

**目标**：研究结果收束为可执行部署方案。

**任务**：
- 更新 champion bundle
- Shadow mode / greyscale 验证
- Live IC consistency 检查
- 回滚规则 + drift rules 写入配置
- 评估是否需要进入下一阶段采购

**Production Hardening TODO (延后项, 不 block Week 3-11):**
- [ ] Week 3 Codex review P2 遗留:
  - `aggregate_minute_to_daily` 不查 09:30/15:59 端点, partial-session 日 gap_pct/overnight_ret/intraday_ret 可能用偏移 anchor
  - `reference_prices` 空 DataFrame 列访问 KeyError (daily 先于 minute ingest 边角场景)
  - health_check 日内 timing (pre-open 假 unhealthy, 低价值)
- [ ] `validate_minute_day_reconciliation_aplus` 只持久化 warning_events, blocker events 只 raise 不入 price_reconciliation_events → Gate 2 查 blocker 计数掩盖问题 (仅 Step B 启用后影响)
- [ ] (未来 review 发现的其他 edge cases 续加)

**Gate**:
- 单一 champion + 单一 rollback 逻辑
- "下一阶段该不该买新数据"明确判断
- Production hardening TODO 清零 (或明确接受长期 known issue)

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
