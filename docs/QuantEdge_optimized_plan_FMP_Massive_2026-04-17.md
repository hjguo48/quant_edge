# QuantEdge 详细研究方案与 12 周排期（基于 FMP Premium + Massive Stocks Developer）

**版本**：2026-04-17 重排版  
**适用对象**：QuantEdge 当前主仓库与现有订阅条件  
**目标**：优先提升模型预测能力，并把研究指标更稳地转化为 production-style 成本后净收益。

---

## 一、结论先行：方案需要优化，而且优化方向已经变化

基于你当前的订阅组合，原来的计划需要做 5 个关键调整：

1. **短端研究优先级上调，但不是把 1D 直接变成主线。**  
   Massive Stocks Developer 已经给到 **10 年历史的 day/minute aggregates、second aggregates、trades、snapshot、websocket、flat files**。这意味着你现在已经具备做更严肃的 **5D 短端事件/微观结构研究** 的条件。  
   但是因为仍然**没有 quotes / NBBO**，1D 仍然不适合作为主 alpha，而更适合作为 **execution timing / risk throttle / 风险开关层**。

2. **20D/60D 仍然是主盈利线，5D 是第一增强线。**  
   60D 现在仍是最强、最干净的证据来源；20D 最可能成为“IC 与净收益最平衡”的第二主线；5D 因为 Massive 的分钟/trades 数据，现在值得升级为 **事件吸收与拥挤修正层**。  
   所以总架构应改成：
   - `60D = core alpha`
   - `20D = revision/crowding adjustment`
   - `5D = event + microstructure overlay`
   - `1D = execution / risk throttle`

3. **FMP Premium 的最佳用法不是再拉更多基础财务表，而是把“分析师代理信号”和“事件层”吃满。**  
   你已经有 fundamentals、earnings_estimates、insider_trades、sec_filings。接下来最该补的是：
   - historical stock grades
   - price target summary / consensus
   - ratings-historical（作为财务健康/质量趋势，不是 analyst rating 本体）
   - corporate calendars / earnings calendars
   这些是你当前最接近 “analyst revision proxy” 的低成本增强方式。

4. **不要立刻全量回填 Massive trades。**  
   Massive 的 trades flat files 非常大，适合做“定向抽取”，不适合第一阶段就对全市场全历史全量落库。第一阶段应以 **minute aggregates 为主、trades 为辅**：
   - minute aggregates：优先 2019+ 全 active universe
   - trades：只做 top liquidity 子集 + 事件窗口 + 弱窗口诊断样本

5. **短期内不急着再买新数据，先把现有订阅的边际价值吃满。**  
   当前最缺的外部层依然是：
   - 真正的 analyst estimate snapshots / line-item revisions
   - 单股票 options IV/skew/term structure
   - borrow fee / utilization / lendable supply
   - quotes / NBBO
   但在没把 Massive + FMP 的潜力吃透前，不建议立刻扩预算。

---

## 二、基于当前订阅，你已经可以做什么、还缺什么

## 2.1 Massive Stocks Developer：现在就能直接强化的能力

### 已可用能力
- 10 年历史 day aggregates
- 10 年历史 minute aggregates
- second aggregates
- trades
- flat files / 无限下载
- snapshots / websocket
- corporate actions / reference

### 这些能力最适合拿来做什么
- **1D/5D 短端标签重构**：拆开 overnight / intraday
- **微观结构 proxy**：gap、opening range、closing pressure、volume shock、realized vol、VWAP 偏离、minute-level Amihud proxy
- **事件吸收特征**：财报日后 1-5 日的 intraday drift、gap-fill、late-day continuation
- **执行层**：交易时点、缓冲区、last-30m volatility、开盘/收盘风险标签

### 当前仍然做不到的部分
- 无 quotes / NBBO，因此做不了真正的 bid-ask spread、BBO depth、quote imbalance、microprice
- 无 Financials & Ratios（Developer 不含），所以日频基本面仍以 FMP 为主

## 2.2 FMP Premium：最值得重新利用的能力

### 已经在库的强项
- fundamentals
- earnings actual / estimate
- insider trades
- SEC filings metadata
- 基本的 analyst estimates（但历史快照弱）

### 现在最该补用的 endpoint / 数据族
- historical stock grades
- stock grades summary / latest grades
- price target summary
- price target consensus
- ratings-historical（财务健康评分历史）
- earnings calendar / transcript availability（仅当你的 key 真的可用）

### FMP 在你当前体系中的最佳角色
- **不是主价格源**（Massive 更强）
- **不是短端微观结构源**（Massive 更强）
- **而是中频基本面、事件、分析师代理信号源**

## 2.3 现阶段仍然缺失的关键层

### 仍然缺、且会限制你接近顶级量化上限的部分
1. 真正历史快照化的 analyst estimates / revisions  
2. 单股票 options IV / skew / term structure  
3. borrow fee / utilization / lendable supply  
4. historical quotes / NBBO  
5. 更干净的历史 universe membership / benchmark  

### 因此对“顶级量化实力”的现实定义应改成
在**当前订阅和当前系统边界内**，优先把 QuantEdge 做成：
- 接近顶级的 **中频横截面研究纪律**
- 接近顶级的 **60D/20D 预测与组合转化能力**
- 中上水平的 **5D 事件与微观结构覆盖**
- 受限但可用的 **1D 执行与风控层**

---

## 三、从今天起的总目标重写

### 总目标
在不新增高价数据订阅的前提下，利用 **FMP Premium + Massive Stocks Developer + 现有仓库脚本体系**，在 12 周内完成：

1. 把当前研究流程从“多版本现实”统一成单一真相。
2. 把特征研究改造成 **horizon × family** 的正式矩阵。
3. 显著强化 5D 与 20D/60D 的预测能力。
4. 让模型比较不再只看 IC，而是统一在 production-style post-cost truth table 上决胜。
5. 输出一个可灰度、可 shadow、可回滚的多层决策系统。

### 12 周后的理想状态
- 60D：仍是主 alpha，但更稳、更能穿透组合层
- 20D：形成独立 champion，且能显著补足 60D 的弱窗口
- 5D：不再是 60D 缩短版，而是事件/微观结构专属层
- 1D：形成执行/risk throttle 原型，不再强求主 alpha
- 至少一个 horizon 在 production-style post-cost 下年化净超额 > 5%

---

## 四、总架构：四层系统而不是一个“大一统模型”

### Layer A：60D Core Alpha
职责：提供中期主排序、组合核心方向  
主特征：quality / profitability / value / residual momentum / macro interaction / analyst proxy  
主模型：Ridge / ElasticNet  
产出：core ranking

### Layer B：20D Revision & Crowding Adjustment
职责：对 60D 排序进行中频修正  
主特征：analyst proxy / price target drift / earnings drift / short-interest dynamics / sector diffusion  
主模型：Ridge / ElasticNet + shallow tree challenger  
产出：ranking adjustment

### Layer C：5D Event & Microstructure Overlay
职责：捕捉事件后漂移、短端拥挤、微观结构修正  
主特征：gap / fill / opening range / late-day pressure / minute volatility / trade imbalance proxy / filing burst / earnings recency  
主模型：Ridge baseline + shallow LightGBM/XGBoost + ranking prototype  
产出：overlay score

### Layer D：1D Execution / Risk Throttle
职责：决定是否减仓、缓执行、跳过高风险交易日  
主特征：VIX state / opening shock / intraday realized vol / liquidity stress / event-day tag  
主模型：rule-based + small classifier/ranker  
产出：execution timing / throttle state

---

## 五、研究流程重构：必须从“按模型”改成“按 horizon × family”

## 5.1 Family Registry（建议固定为 9 类）
1. `price_momentum`
2. `liquidity_microstructure`
3. `fundamental_quality`
4. `analyst_expectations_proxy`
5. `shorting_crowding`
6. `macro_regime`
7. `event_earnings_sec`
8. `sector_network_diffusion`
9. `execution_risk`

## 5.2 每个 horizon 必做 2 个实验
1. `only_one_family`：只上单一 family，看独立增量  
2. `leave_one_family_out`：去掉一个 family，看损失  

## 5.3 统一评判顺序
1. family ablation  
2. model sweep  
3. walk-forward  
4. regime / weak-window analysis  
5. post-cost truth table  
6. G3 gate  
7. 才允许进入 fusion / live

---

## 六、按 horizon 的预测增强策略

## 6.1 60D：维持主线，但更强调“强特征少而精”

### 目标
- 保住当前最强 OOS IC
- 降低弱窗口损伤
- 显著提升 post-cost 可实现性

### 重点增强 family
- fundamental_quality
- analyst_expectations_proxy
- macro_regime interaction
- event_earnings_sec（中期漂移）

### 新特征优先级
1. `quality_composite`
2. `gross_profitability`
3. `operating_profitability`
4. `accrual_quality`
5. `historical_grade_trend_60d`
6. `price_target_drift_60d`
7. `coverage_change_proxy`
8. `rating_momentum_60d`
9. `high_vix_x_beta`
10. `credit_spread_x_distress`

### 模型策略
- 主线：Ridge / ElasticNet
- challenger：shallow GBDT 只吃 interaction residual
- retained 特征目标：12–25 个

## 6.2 20D：重点打造第二主线

### 目标
- 做出独立于 60D 的 champion
- 提升对 W10/W11 这类窗口的适应能力

### 重点增强 family
- analyst_expectations_proxy
- shorting_crowding
- sector_network_diffusion
- event_earnings_sec

### 新特征优先级
1. `net_grade_change_20d`
2. `upgrade_minus_downgrade_20d`
3. `price_target_revision_20d`
4. `consensus_upside_zscore`
5. `target_dispersion_proxy`
6. `earnings_surprise_recency`
7. `pead_setup_20d`
8. `short_interest_accel_20d`
9. `sector_rel_ret_20d`
10. `stock_vs_sector_20d`

### 模型策略
- Ridge / ElasticNet baseline
- LightGBM / XGBoost shallow tree challenger
- 尝试 top-decile weighted loss / pairwise ranking prototype

## 6.3 5D：正式升级为事件与微观结构线

### 目标
- 不再复用 60D 特征集
- 让短端信号来自 Massive minute/trades + FMP events

### 重点增强 family
- liquidity_microstructure
- event_earnings_sec
- shorting_crowding
- execution_risk

### 新特征优先级
1. `gap_pct`
2. `gap_fill_score_1d_5d`
3. `open_30m_ret`
4. `last_30m_ret`
5. `intraday_reversal_score`
6. `realized_vol_1d_5d`
7. `volume_curve_surprise`
8. `close_to_vwap`
9. `trade_imbalance_proxy`
10. `event_cluster_score`
11. `filing_burst_5d`
12. `earnings_recency_5d`
13. `short_sale_ratio_accel`
14. `minute_amihud_proxy`

### 模型策略
- Ridge baseline
- shallow LightGBM / XGBoost
- ranking prototype 比纯回归更优先

## 6.4 1D：保持辅助层定位

### 目标
- 不追求主 alpha
- 主要用于执行、避坑、控 turnover、控 regime risk

### 核心信号
- overnight / intraday split
- opening shock
- realized intraday vol
- close auction proxy
- event-day risk tag
- VIX regime

### 模型策略
- 规则 + 轻量模型
- 产出：`normal / cautious / defensive` 执行状态

---

## 七、数据实施优先级（按 ROI）

## P0：立刻做，不新增订阅

### Massive
1. day aggregates 作为日频价格真值核验层  
2. minute aggregates（2019+，active universe）  
3. trades（只做定向抽样）  
4. corporate actions 与 reference 补齐价格/成分修复

### FMP
1. historical stock grades  
2. price target summary / consensus  
3. ratings-historical  
4. earnings calendar / report dates  
5. 继续强化 insider / SEC features

## P1：仅在 P0 吃透后再决定
1. I/B/E/S 或 Zacks 类 analyst history  
2. 单股票 options IV / skew / term structure  
3. borrow fee / utilization / lendable supply  
4. quotes / NBBO  

---

## 八、12 周详细排期

## Week 1：统一当前真相，冻结研究协议

### 目标
消除“当前冠军不唯一”“研究协议分散”“artifact 与报告不一致”问题。

### 任务
- 建 `CURRENT_STATE.yaml`
- 建 `CHAMPION_REGISTRY.yaml`
- 建 `horizon_registry.yaml`
- 建 `family_registry.yaml`
- 建 `research_manifest_<date>.json`
- 对齐以下结论来源：
  - 最新项目总览
  - 当前 data/reports artifacts
  - 当前 live / shadow 配置

### 复用脚本
- `run_registry_setup.py`
- `diagnose_model_consistency.py`

### 产出物
- `configs/research/current_state.yaml`
- `configs/research/horizon_registry.yaml`
- `configs/research/family_registry.yaml`
- `data/reports/research_manifest_YYYYMMDD.json`

### Gate
- 任何人只看一个文件就知道当前 champion 是谁
- 以后不再允许“报告说 A，artifact 说 B”

---

## Week 2：数据真值与特征一致性审计

### 目标
先把研究数据底座清干净，再扩新特征。

### 任务
- 做 `feature_store vs parquet` parity audit
- 审计 `stock_prices` 的 split / adjustment / PIT 规则
- 审计 benchmark 与 SPY 起始日问题
- 固定 active universe / universe_membership 规则
- 明确 Massive day aggregates 与当前 stock_prices 的差异

### 复用脚本
- `check_db_status.py`
- `check_date_range.py`
- `backfill_price_gap.py`
- `fix_split_adjusted_prices.py`
- `backfill_sp500_history.py`

### 产出物
- `data/reports/feature_parity_audit.json`
- `data/reports/price_truth_audit.json`
- `data/reports/universe_audit.json`

### Gate
- 训练与服务特征定义统一
- 日频价格真值层可回放

---

## Week 3：Massive minute aggregates 正式入库（先不碰全量 trades）

### 目标
让 5D/1D 终于拥有专属数据层。

### 任务
- 新建 minute 数据表 / parquet pipeline
- 优先回填 **2019+ active universe** 的 minute aggregates
- 建 minute-derived feature pipeline
- 同步生成 overnight / intraday labels

### 建议新表
- `stock_minute_aggs`
- `labels_intraday`

### 首批新特征
- `gap_pct`
- `overnight_ret`
- `intraday_ret`
- `open_30m_ret`
- `last_30m_ret`
- `realized_vol_1d`
- `volume_curve_surprise`
- `close_to_vwap`
- `transactions_count_zscore`

### 建议新脚本
- `run_intraday_feature_build.py`
- `run_intraday_label_build.py`

### Gate
- 2019+ active universe 的 minute 覆盖率 > 95%
- 1D/5D 新标签成功生成

---

## Week 4：定向接入 Massive trades，构建低成本 trade-proxy family

### 目标
不用全量落库 trades，也先拿到最值钱的 trade-level 信息。

### 任务
- 只对以下范围抽 trades：
  - top 200 liquidity 股票
  - earnings / SEC filing / 大 gap 事件窗口
  - W5/W6/W11 等弱窗口样本
- 构建 tick-rule trade imbalance proxy
- 构建 trade size skew / late-day aggressiveness proxy
- 用 Massive condition codes 做过滤

### 建议新表
- `stock_trades_sampled`

### 首批新特征
- `trade_imbalance_proxy`
- `large_trade_ratio`
- `late_day_aggressiveness`
- `offhours_trade_ratio`

### Gate
- 不做全量 trade lake
- 能拿到一套足够用于 5D/弱窗口诊断的 trade family

---

## Week 5：FMP analyst proxy / event proxy 扩容

### 目标
在不买 I/B/E/S 的情况下，先把 FMP 的分析师代理层做起来。

### 任务
- 接 historical stock grades
- 接 price target summary
- 接 price target consensus
- 接 ratings-historical（作为财务健康/质量趋势）
- 接 earnings calendar / report dates（如果当前库还不完整）

### 建议新表
- `analyst_grades_history_pit`
- `price_target_summary_pit`
- `price_target_consensus_pit`
- `ratings_history_pit`

### 首批新特征
- `net_grade_change_5d/20d/60d`
- `upgrade_count`
- `downgrade_count`
- `consensus_upside`
- `target_price_drift`
- `target_dispersion_proxy`
- `coverage_change_proxy`
- `financial_health_trend`

### Gate
- analyst_proxy family 至少形成 8–12 个可回测特征
- 缺失率与发布时间规则明确

---

## Week 6：特征家族归一与覆盖率诊断

### 目标
形成“每个 horizon 拥有自己的家族配置”的研究基础。

### 任务
- 所有新特征写入 family registry
- 做 family coverage report
- 做 missingness economic meaning 审计
- 对 `is_missing` 做去伪存真，删除 vendor bias 列
- 产出每个 horizon 的候选 family 清单

### 建议产出
- `data/reports/family_coverage_report.json`
- `data/reports/missingness_audit.json`

### Gate
- 每个特征都有唯一 family
- 任何保留的 `is_missing` 都能解释其经济含义

---

## Week 7：按 horizon 重跑 IC screening 与 family ablation

### 目标
不再让 1D/5D/20D 使用 60D 的 proxy 结论。

### 任务
- 分别跑 1D / 5D / 20D / 60D screening
- 跑 `only_one_family`
- 跑 `leave_one_family_out`
- 分 horizon 产出 retained 候选集

### 复用脚本
- `run_ic_screening.py`

### 建议新增脚本
- `run_family_ablation.py`

### Gate
- 每个 horizon 都有自己的 top features / top families
- 你能明确回答“现在最值钱的前三个 family 是什么”

---

## Week 8：模型层重排：先 baseline，再 challenger，再 ranking

### 目标
让模型选择真正服务于 horizon，而不是一视同仁。

### 任务
- 全 horizon 跑 Ridge / ElasticNet
- 5D / 20D 跑 shallow LightGBM / XGBoost
- 60D 只允许树模型做 challenger，不抢主线
- 加入 top-decile weighted loss 或 pairwise ranker prototype
- 比较回归目标 vs 排序目标

### 复用脚本
- `run_tree_comparison.py`
- `run_lstm_comparison.py`（只作记录性对比，不作为主线）

### 建议新增脚本
- `run_ranker_comparison.py`

### Gate
- 至少 2 个 horizon 出现“模型提升是真实而非噪声”的证据
- 排序目标若胜出，则进入 Week 9 主线

---

## Week 9：walk-forward、弱窗口与 regime 诊断

### 目标
确认新增 family 和模型是否真的穿过 OOS 稳定性。

### 任务
- 13-window walk-forward × 4 horizons
- 专门复查 W5/W6/W10/W11
- 重新跑 regime analysis
- 区分“不可预测的 regime break”与“可以提前 gating 的窗口”

### 复用脚本
- `run_extended_walkforward.py`
- `run_walkforward_comparison.py`
- `run_regime_analysis.py`
- `diagnose_w6_w8.py`

### Gate
- 至少一个 horizon 的新增版本在 OOS IC 上显著优于旧 baseline
- W6 / W11 之类可 gating 的窗口，有明确先验特征或 validation 线索

---

## Week 10：production-style post-cost truth table 统一评估

### 目标
结束“研究赢了但交易没赢”的模糊阶段。

### 任务
- 所有 horizon / 模型统一放入同一 truth table
- 比较：
  - `equal_weight_top_decile`
  - `score_weighted`
  - `score_weighted + buffer`
  - `hold-and-trim`
  - `sector/beta-neutral constrained optimizer`
- 优化 turnover budget
- 统一输出 gross / cost drag / net / DD / participation

### 复用脚本
- `run_portfolio_comparison.py`
- `run_portfolio_optimization_comparison.py`
- `run_turnover_optimization.py`

### Gate
- 至少一个 horizon 在 production-style 成本后年化净超额 > 5%
- 若没有，则不得进入 fusion / deployment 提级

---

## Week 11：多层决策融合与 risk throttle

### 目标
做“决策层融合”，不是简单 signal averaging。

### 任务
- 固定单 horizon champions
- 做以下结构化融合：
  - `60D core ranking`
  - `20D ranking adjustment`
  - `5D event overlay`
  - `1D execution throttle`
- 引入 `normal / cautious / defensive` 三档状态
- 把 validation-IC gating 与 market-state gating 合并

### 复用脚本
- `run_horizon_fusion.py`
- `run_ic_weighted_fusion.py`（仅作对照，不作主线）
- `run_signal_fusion_experiment.py`

### Gate
- fusion 的净表现优于单一 60D champion，或者至少显著改善回撤/弱窗口
- 若简单平均仍优于结构融合，说明架构设计有问题，需回退

---

## Week 12：灰度 / shadow / live validation 与下一阶段决策

### 目标
把研究结果收束成可执行部署方案与下一阶段采购条件。

### 任务
- 更新 champion bundle
- shadow mode / greyscale 验证
- live IC consistency 检查
- 回滚规则与 drift rules 写入配置
- 评估是否需要进入下一阶段采购（I/B/E/S / options / borrow / quotes）

### 复用脚本
- `run_g3_gate.py`
- `run_shadow_mode.py`
- `run_greyscale_live.py`
- `run_greyscale_monitor.py`
- `run_live_ic_validation.py`
- `run_live_pipeline.py`

### Gate
- 形成单一冠军 + 单一回滚逻辑
- 形成“下一阶段该不该买新数据”的明确判断

---

## 九、每周固定运行节奏（建议）

## 每日
- Massive day / minute 增量同步
- FMP insider / SEC / grades / price target 增量同步
- data quality checks

## 每周三
- 本周特征覆盖率 / 缺失率检查
- 弱窗口监控更新

## 每周五收盘后 / 周六
- 统一跑 signal generation
- walk-forward / live validation 增量更新
- 生成周报

## 每周日
- 研究分支合并 / champion review
- 是否触发 retrain / rollback 决策

---

## 十、成功标准（你真正应该盯的门槛）

### 研究层
- 每个 horizon 都有专属 top families
- 至少 2 个 horizon 的 OOS IC 明显高于当前旧 baseline
- 5D 不再依赖 60D 特征集

### 经济层
- 至少一个 horizon production-style 成本后年化净超额 > 5%
- 相比当前最悲观的 equal-weight post-cost，净值曲线显著改善
- turnover 与 capacity 不再失控

### 部署层
- 单一 champion 真相
- 单一 rollback rule
- shadow / greyscale 连续运行稳定
- live IC consistency 不恶化

---

## 十一、明确暂缓的事项

在这 12 周里，以下事项不作为主线：

1. 全量 Massive trades lake 建设  
2. full-depth microstructure / quote imbalance 研究  
3. transcript NLP 大工程（除非你确认 FMP Premium 实际可用）  
4. 更深层的 LSTM / Transformer 主线化  
5. 再买昂贵数据订阅  
6. 简单 IC-weighted fusion 作为正式主线  

---

## 十二、12 周后如何决定是否升级数据订阅

只有当以下条件至少满足 2 条时，才建议扩数据预算：

1. 你已经证明 Massive + FMP 的新增 family 仍无法把 production-style net excess 拉正  
2. 20D/60D 已经很强，但明显缺 analyst revision / options / borrow 这三层  
3. 5D 的 minute/trade proxy 已有价值，但仍被 quote / spread / BBO 缺失明显限制  
4. 研究纪律和 post-cost pipeline 已经成熟，不会“买了更贵数据却继续被流程问题卡住”  

若满足，再按顺序升级：
1. I/B/E/S / Zacks 类 analyst history  
2. 单股票 options IV/skew/term structure  
3. borrow fee / utilization / lendable supply  
4. quotes / NBBO  

---

## 十三、最后一句话

你现在最优的路线，不是继续抽象地追“更多 alpha 因子”，而是把现有订阅的边际价值吃满，形成：

**Massive 驱动的 5D/1D 微观结构层 + FMP 驱动的 20D/60D 分析师代理与事件层 + 统一的 post-cost portfolio truth table。**

只要这 12 周执行到位，QuantEdge 会从“研究成功但 monetization 未闭环”，推进到“具备接近顶级中频量化研究实力、并开始接近可部署净收益能力”的新阶段。
