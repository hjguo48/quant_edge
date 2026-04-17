# QuantEdge 实施计划 v5.0

**更新日期:** 2026-04-17
**方法论:** 基于 plan.txt (规范化研究协议) 重构
**核心理念:**
- V5 no-analyst 保持 live baseline (IC=0.062, 6/6 PASS)
- V7 按 plan.txt 方法论重做 (免费数据 + family ablation + 严格门控)
- V7 必须 post-cost > V5 才能替换

---

## 1. 项目定位

QuantEdge 是研究驱动的机构级美股量化系统。核心原则:
- **先证能赚钱，再做产品** — Alpha → 经济可行性 → 产品化
- **IC 是必要非充分条件** — 最终以 post-cost net excess 为准
- **Family → Model → Portfolio** 严格顺序实验
- **研究定型后才工程化** — 避免返工

---

## 2. 当前 Baseline (V5 no-analyst)

| 指标 | 值 |
|------|-----|
| 配置 | 60D Ridge, 19 features (no-analyst), 13 windows |
| IC | 0.062 |
| t-stat | 2.17 (p=0.025, one-sided) |
| G3 Gate | 6/6 PASS |
| Post-cost (proxy vs script) | +35% vs -9.6% (未解决) |
| 经济可行性 | **production-style backtest 未做, 真实 net excess 未知** |

**V5 被锁定为 live baseline，所有 V7 改进必须在此基础上增量提升。**

---

## 3. 数据层现状

### 已有 (研究数据信封 V5)
- stock_prices (711 tickers + 14 ETFs)
- fundamentals_pit (693 tickers)
- earnings_estimates (677 tickers)
- short_interest (696 tickers)
- insider_trades (722 tickers)
- sec_filings (643 tickers, metadata only)
- macro_series_pit (VIX/10Y/2Y/FFR/Credit)

### 待加 (Phase 1 免费数据)
- FINRA Daily Short Sale Volume
- SEC FTD (Fails-to-Deliver)
- SEC 13F Holdings
- Chicago Fed NFCI/ANFCI
- FRED STLFSI4, ICE BofA OAS
- FMP analyst grades (已付费 API)

### 已知问题
- analyst_estimates: forward-only, ~75% 回测为 NaN
- news_sentiment: 86 tickers / 2024+, 不可用于回测
- SPY: 2016-04+ (Polygon 10yr limit)
- feature_store (176) vs parquet (238) 不一致 — Phase 0 修复
- **V6 parquet 有 IC 回归问题，已放弃**

---

## 4. 关键发现与教训（永久保留）

1. Purge/embargo 揭示 58% IC 是 label leakage
2. 树模型在 60D purge 下崩溃 80%，1D/5D 短 purge 可能不同
3. analyst_coverage 去掉后 IC 反升 (0.056→0.062)
4. IC screening 默认 HORIZON_DAYS=5 — 必须每个 horizon 独立
5. G3 proxy (+35%) ≠ portfolio backtest (-9.6%) — 需 production-style truth table
6. **不同 horizon 需要不同特征 + 不同模型**
7. V6 加 28 特征后 IC 反而下降 — **更多特征 ≠ 更好，需要 family ablation**
8. Ridge retained 应控制 20-25 个，避免弱特征稀释
9. 多 Horizon 融合不是信号加权，是决策层融合
10. WRDS / I/B/E/S alumni 不可用，商业用户需直接购买

---

## 5. 双轨执行计划

### 部署轨 (立即，本周)

| 步骤 | 任务 | 状态 |
|------|------|------|
| D1 | V5 no-analyst bundle 锁定 | ✅ 完成 |
| D2 | 排查 weekly_signal_pipeline DAG 4/11 失败原因 | 🔄 Codex 处理中 |
| D3 | 手动触发一次 weekly signal 确认 OK | ⏳ |
| D4 | 今晚周五灰度正常出信号 | ⏳ |
| D5 | 灰度稳定跑 1-2 周，作为 V7 的对比基线 | ⏳ |

**不改 bundle，不动 live 系统，直到 V7 在 post-cost truth table 上明确胜出。**

---

### 研究轨 (V7 重构，1-2 周)

#### Phase 0: 研究协议冻结 (2-3 天)

**0A** 建立 4 个 YAML 配置文件:
- `configs/research/horizon_registry.yaml` — 1D/5D/10D/20D/60D 的 label/purge/window/cost
- `configs/research/family_registry.yaml` — 9 个标准 family (price_momentum, liquidity_microstructure, fundamental_quality, analyst_expectations, options_implied, shorting_crowding, macro_regime, network_industry, event_text)
- `configs/research/current_champion.yaml` — horizon/model/features/metrics/deploy_status
- `data/reports/research_manifest_<date>.json` — 实验可回放

**0B** Feature parity audit:
- 解决 feature_store (176) vs parquet (238) 不一致
- 输出 `data/reports/feature_parity_audit_<date>.json`
- 逐列确认 name/dtype/null_rate/lag/PIT rule/source table

**0C** 新建 `scripts/run_family_ablation.py`:
- 支持 only-one-family 和 leave-one-family-out
- 输出 `data/reports/family_ablation_<horizon>_<date>.json`

**0D** 冻结 post-cost truth table 协议:
- 4 个组合层: equal_weight_top_decile / score_weighted / score_weighted+buffer / sector-beta-neutral
- 统一输出: gross_excess / cost_drag / net_excess / turnover / max_drawdown / ADV_participation
- 所有 horizon 都在同一张表对决

**Gate:** 4 个 YAML + 3 个脚本 + parity audit 完成

---

#### Phase 1: 免费数据扩展 (1 周)

**1A** FINRA Short Sale Volume:
- `src/data/finra_short_sale.py`
- 新表: `short_sale_volume_daily`
- 新特征: short_sale_ratio_1d/5d, short_sale_accel_5d, short_sale_ratio_sector_rel

**1B** SEC FTD (Fails-to-Deliver):
- `src/data/sec_ftd.py`
- 新表: `ftd_pit`
- 新特征: ftd_to_float, ftd_shock_1m, ftd_persistence

**1C** SEC 13F Holdings:
- `src/data/sec_13f.py`
- 新表: `holdings_13f_pit`
- 新特征: inst_ownership_pct, ownership_change_qoq, common_owner_overlap

**1D** 宏观扩展:
- 加入 NFCI, ANFCI, STLFSI4, ICE BofA HY OAS
- 只做 level/zscore + macro × stock interaction
- 不做 raw macro 截面排序

**1E** FMP analyst grades (免费，已有 API):
- 已在 Step 3A 计划中
- 拉历史评级升降级数据
- 构建 upgrade_count, net_revision, days_since_last_upgrade 等特征

**Gate:** 所有数据源有 PIT rule + lag rule + coverage report + null report

---

#### Phase 2: Horizon-specific 特征 (3-5 天)

**2A** 1D/5D 特征（事件/流动性为主）:
- `src/features/microstructure.py` (新)
- close_to_close + overnight + intraday 三套 labels
- gap_pct, gap_fill_score, close_open_return, volume_shock, amihud_proxy
- short_sale_ratio_1d/5d, ftd_shock
- 5D event-drift family: earnings_surprise_recency, pead_setup, filing_burst

**2B** 20D/60D 特征（基本面/质量/拥挤为主）:
- 质量 family: gross_profitability, operating_profitability, accrual_quality, investment, asset_growth, quality_composite
- 拥挤 family: days_to_cover_proxy, short_interest_delta, ftd_persistence, ownership_crowding, financial_conditions × quality

**2C** 所有 family 两版：base + with_missing_flag
- is_missing 留下必须能解释"缺失本身有经济含义"
- 不能把 vendor coverage bias 当 alpha

**Gate:** 每个 horizon 有独立特征集，不再共用

---

#### Phase 3: Family → Model → Portfolio 严格实验 (3-5 天)

**3A** Per-horizon IC screening:
- 4 个 horizon 独立跑 (1D/5D/20D/60D)，V7 parquet
- 输出 `top_features_<horizon>.csv`
- **串行跑，避免 OOM**

**3B** Family ablation:
- only-one-family + leave-one-family-out
- 明确"最值钱的 3 个 family 是什么"

**3C** Walk-forward per horizon:
- 每个 horizon 只用自己的 feature set
- 输出 `walkforward_<horizon>.json`

**3D** Model sweep per horizon:
- 60D: Ridge/ElasticNet 主线，树模型 challenger
- 20D: Ridge + shallow tree
- 1D/5D: 独立测试 tree 是否有边际价值

**3E** G3 Gate per horizon:
- 每个 horizon 独立 gate
- IC 门槛按 horizon 调整

**Gate:** 至少一个 horizon 成本后年化超额 > 5%

---

#### Phase 4: Post-cost Truth Table (2-3 天)

**4A** `run_portfolio_comparison.py`:
- equal_weight_top_decile vs score_weighted
- 诊断差异来源 (turnover / 集中度 / cost model)

**4B** `run_portfolio_optimization_comparison.py`:
- score_weighted + buffer
- sector/beta-neutral constrained optimizer
- hold-and-trim

**4C** `run_turnover_optimization.py`:
- 找 IC retained / cost drag 最优折中
- 每个 horizon 给出 turnover budget

**4D** `run_regime_analysis.py`:
- 弱窗口 W5/W6/W11 在加 stress/crowding 特征后是否可 gating
- 输出 normal/cautious/defensive 三档 gating 规则

**Gate:** 完整 truth table，每个 horizon 的 net excess 可信

---

#### Phase 5: 单 horizon 成立后才融合 (2-3 天)

**5A** 选出每个 horizon 的 champion
**5B** 决策层融合 (不是信号平均):
- 60D = core ranking
- 20D = ranking adjustment
- 5D = event overlay
- 1D = execution timing / risk throttle

**5C** `run_horizon_fusion.py`:
- 评估标准: net truth table，不是裸 IC

**Gate:** V7 fusion post-cost > V5 no-analyst post-cost

---

### 部署升级轨 (V7 PASS 后)

- 更新 fusion_model_bundle
- 灰度切换 V7
- V5 保留 1-2 周作为 fallback
- 观察 live 一致性

---

## 6. 付费数据采购轨 (Step 3 完成后，按 ROI)

| 优先级 | 数据源 | 费用/年 | 什么时候买 |
|--------|--------|---------|----------|
| 1 | LSEG I/B/E/S Detail History | $10-50K | 免费 FMP grades 证明有效后 |
| 2 | OptionMetrics IvyDB | $5-20K | 5D/20D 需要 forward vol signal 时 |
| 3 | S&P Securities Finance | 议价 | 需要借券费率/使用率时 |
| 4 | NYSE Daily TAQ | $5-20K | 1D 成为核心策略时 |

---

## 7. 现在不要做的 (plan.txt 强调)

- ❌ 扩大 generic news_sentiment (覆盖不足)
- ❌ 10-K/10-Q 全文 NLP 大工程
- ❌ 把复杂模型当主线
- ❌ 简单 signal averaging / IC-weighted fusion

---

## 8. 技术架构

```
src/
├── data/                # 数据源 (+ FINRA/FTD/13F 待加)
├── features/            # 特征 (+ microstructure/quality family 待加)
├── labels/              # 1D/5D/10D/20D/60D forward excess return
├── models/              # Ridge + ElasticNet + XGBoost + LightGBM
├── backtest/            # Walk-forward + Almgren-Chriss + 执行模拟
├── portfolio/           # equal_weight + vol_weighted + BL + score_weighted
├── stats/               # IC t-test + Bootstrap + DSR + SPA
├── risk/                # 四层风控
└── universe/            # active universe resolver

scripts/
├── run_ic_screening.py            # IC 筛选
├── run_family_ablation.py         # (新) family 消融分析
├── run_walkforward_comparison.py
├── run_g3_gate.py
├── run_portfolio_comparison.py
├── run_portfolio_optimization_comparison.py
├── run_turnover_optimization.py
├── run_regime_analysis.py
├── run_horizon_fusion.py
└── run_post_cost_truth_table.py   # (新) 统一 post-cost 对比

configs/research/
├── horizon_registry.yaml          # (新)
├── family_registry.yaml           # (新)
└── current_champion.yaml          # (新)

dags/                              # daily data + weekly signal
frontend/                          # React dashboard
```

### 关键 Artifacts

| Artifact | Path |
|----------|------|
| Live bundle | data/models/fusion_model_bundle_60d.json (V5 no-analyst) |
| V5 60D walk-forward | data/reports/walkforward_comparison_60d_v5_no_analyst.json |
| V5 60D G3 Gate | data/reports/g3_gate_v5_no_analyst.json |
| V7 IC screening | data/features/ic_screening_report_v7_<horizon>.csv (待做) |
| Family ablation | data/reports/family_ablation_<horizon>.json (待做) |
| Post-cost truth table | data/reports/post_cost_truth_table_<date>.json (待做) |

---

## 9. 部署状态

| 组件 | 状态 |
|------|------|
| TimescaleDB + Redis + MLflow | ✅ Docker 运行中 |
| Airflow DAG | ⚠️ weekly_signal 4/11 失败, 修复中 |
| 灰度系统 | ⚠️ 停在 4/14 |
| V5 no-analyst bundle | ✅ 锁定 |
| Docker → 阿里云 ECS 迁移 | ⏳ Phase 5 |
| 回测系统 (净值曲线) | ⏳ Phase 5 |

---

## 10. 历史归档

- v3 归档: `docs/archive/IMPLEMENTATION_PLAN_v3_archived_20260416.md`
- v4.1 归档: `docs/archive/IMPLEMENTATION_PLAN_v4_archived_20260417.md`
- plan.txt 参考: `docs/plan.txt` (本次重构来源)
