# QuantEdge — 美股量化研发实施说明书 (v3.0)

**作者：** Manus AI
**日期：** 2026年3月26日
**版本：** v3.0 — 机构级研发规范与产品化实施标准

## 版本演进说明

本文档是 QuantEdge 项目方案的第三次迭代。v1.0 是一份完整的"量化产品方案书"，经专业量化审查后指出其在 Alpha 研究定义、回测规范、组合优化稳健性、风控体系和模型治理方面存在系统性不足。v2.0 针对上述 8 项反馈进行了全面重构，将方案从"产品概念说明"升级为"量化研发实施说明书"。

v3.0 基于对 v2.0 的第二轮深度审查反馈，进一步解决了以下 10 项核心问题：数据底座仍停留在 MVP 阶段与治理框架不匹配、研究定义过于单一方案锁死、统计检验需引入 Deflated Sharpe Ratio 等更严格的方法、交易成本模型需细化为非线性市场冲击模型、缺少朴素权重基线对比、MLOps 架构存在过度工程化风险需分期实施、缺乏面向终端用户的可解释性层、合规部分需升格为产品边界设计约束、参考文献需补充学术级来源、以及文档本身的 QA 问题。

---

## 目录

1. 项目概述与研发路径
2. 核心数据底座与演进路径
3. Alpha 研究定义与多维实验矩阵
4. 特征工程与数据规范
5. 回测规范与实验制度
6. 高级统计检验与防过拟合框架
7. 微观结构与交易成本建模
8. 稳健组合优化与仓位管理
9. 四层系统化风控体系
10. MLOps 分期实施策略
11. 用户解释层与信任构建
12. 数据库设计与可回放能力
13. 系统架构总览
14. 技术栈总览
15. 项目实施路线图
16. 美国 SEC 合规与产品边界设计
17. 参考文献

---

## 1. 项目概述与研发路径

### 1.1 项目定位

QuantEdge 不仅仅是一个面向终端用户的 Web 平台，其核心是一个具备机构级研发规范的**量化生产系统**。本项目方案摒弃了"先做产品，再做策略"的传统软件开发模式，转而采用**"研究中心化"**的量化实施路径 — 先证明策略有效，再做产品封装。

> **核心原则：** 一个量化系统的价值不取决于它的 UI 有多漂亮，而取决于它的 Alpha 是否真实存在、是否可复现、是否能在扣除交易成本后持续盈利。

### 1.2 三阶段研发路径

项目开发严格遵循以下三个阶段，每个阶段都有明确的交付物和准入/准出标准：

| 阶段 | 名称 | 周期 | 核心交付物 | 准出标准 |
|------|------|------|----------|----------|
| Phase 1 | 证明 Alpha | 8-10 周 | 回测报告 + DSR/SPA 统计检验结果 | 样本外 IC > 0.03, DSR p-value < 0.05, 成本后年化超额 > 5% |
| Phase 2 | 生产架构与模型治理 | 6-8 周 | 分阶段 MLOps 管道 + 风控引擎 | Shadow Mode 运行 4 周无异常 |
| Phase 3 | 产品化与用户系统 | 8-10 周 | Web 平台 + 用户 Dashboard + 解释层 | 端到端联调通过 + 合规审查 |

这个顺序与传统软件项目不同，但更符合真实量化项目的成功路径：**先证明策略，再做产品，而不是先把产品全画出来** [1]。

---

## 2. 核心数据底座与演进路径

量化研究中最常被忽视的真理是：再先进的深度学习模型和风控架构，如果建立在存在幸存者偏差、缺乏 Point-in-Time (PIT) 属性的数据之上，最终的实盘结果必然是灾难性的。v2.0 版本坦诚承认了对 Yahoo Finance 和 Alpha Vantage 等免费数据源的依赖，v3.0 在此基础上制定了明确的三阶段数据源升级路径，使数据底座的演进与项目的商业验证阶段严格对齐。

### 2.1 数据源三阶段演进策略

项目在不同生命周期面临的资金约束和数据需求不同，必须采取务实的数据采购策略 [2]。

**第一阶段：原型验证期 (MVP) — 验证基础 Alpha。** 在项目初期（前 3 个月），重点是跑通端到端的数据管道和模型验证流程。此时采用低成本数据源，接受一定的数据瑕疵，但必须在代码层面预留处理 PIT 数据的接口。日频量价数据采用 Polygon.io 或 EODHD API，相比完全免费的 Yahoo Finance，它们提供更稳定的接口、更准确的复权价格和更完整的历史数据。宏观数据采用美联储 FRED API（免费），获取高质量的无风险利率、信用利差和宏观经济指标。基本面数据采用 Financial Modeling Prep (FMP) API，获取基础的财务报表数据，但需注意其 PIT 属性可能不完美。

**第二阶段：准生产期 — 引入机构级特征。** 当基础策略在样本外测试中展现出初步潜力（如 IC > 0.03）并准备进行小规模 Shadow Mode 运行（第 4-6 个月）时，必须升级数据源以消除隐性偏差。核心升级是引入 Zacks Data 或 FactSet 的基础包，获取真正的 **Point-in-Time 基本面数据** [3]。这是消除"未来函数"的唯一方法，确保系统在回测 2020 年 Q1 时，使用的绝对是当时市场实际可见的未修正财报数据。同时采购标普 500 历史成分股变动数据（Constituent History），以彻底消除幸存者偏差（Survivorship Bias）。

**第三阶段：机构级生产期 — 全面专业化。** 当平台准备面向真实用户收费或管理较大规模资金时，数据底座必须达到机构级标准。终极方案是接入 Refinitiv/LSEG 或 Bloomberg B-PIPE，获取毫秒级的高质量行情、精确到分钟的公司行为（Corporate Actions）调整历史，以及最权威的分析师预期（Estimates）数据 [4]。

| 阶段 | 数据源 | 月成本估算 | PIT 支持 | 幸存者偏差处理 |
|------|------|----------|--------|----------|
| Tier 1 MVP | Polygon.io + FRED + FMP | $100-200 | 部分（需手动处理） | 不完整 |
| Tier 2 准生产 | + Zacks/FactSet 基础包 | $500-2,000 | 完整 | 完整（历史成分股） |
| Tier 3 机构级 | + Refinitiv/Bloomberg | $10,000+ | 完整 + 审计级 | 完整 + Corporate Actions |

### 2.2 数据库层面的 PIT 治理

无论使用何种数据源，QuantEdge 的底层数据库架构强制执行**双时间戳隔离机制**。`event_time` 记录事件在现实世界中发生的理论时间（如财报截至日期），`knowledge_time` 记录该数据点首次被系统或市场获取的精确时间戳（如财报实际发布到 SEC EDGAR 系统的时间）。在任何回测查询中，SQL 语句必须强制包含 `WHERE knowledge_time <= backtest_current_time` 的约束，从物理层面上杜绝数据窥探。

### 2.3 数据质量的诚实评估

v3.0 明确承认一个事实：**当前方案的治理框架已经走到"准生产级"，但数据层仍停留在 MVP/低成本阶段。** 这种错位对于原型验证阶段是可以接受的，但如果对外宣称"真实可帮助用户长期决策"，则数据一致性、PIT 基本面、历史成分股、退市数据和公司行为完整性，都可能成为系统的致命短板。因此，数据源的升级不是"锦上添花"，而是从 Phase 1 到 Phase 2 的**硬性准入条件**：只有当 Tier 1 数据上的策略通过了 DSR 检验，才有资格升级到 Tier 2 数据进行更严格的验证。

---

## 3. Alpha 研究定义与多维实验矩阵

### 3.1 核心研究定义 (v2.0 主线)

在引入任何复杂的机器学习模型之前，必须首先对量化研究的边界进行严格定义。以下五个定义构成了整个量化研究的地基 [1]。

| 定义项 | 具体方案 | 设计理由 |
|--------|----------|----------|
| 股票池 (Universe) | S&P 500 成分股，叠加 ADV > $50M 流动性过滤 | 确保可交易性，降低冲击成本；避免全市场噪声 |
| 预测标签 (Label) | 未来 5 个交易日的相对基准超额收益 (5D Forward Excess Return) | 横截面排序问题，而非绝对收益预测；基准为 SPY |
| 问题类型 | Cross-sectional Ranking（横截面排序） | 预测"哪只股票比其他股票表现更好"，而非"某只股票涨多少" |
| 再平衡频率 | 周度 (Weekly)，周五收盘信号，周一开盘执行 | 平衡 Alpha 捕获频率与换手率成本 |
| 信号到仓位 | Top Decile Long-Only，经稳健优化后的权重 | 做多预测排名前 10% 的股票，权重由多种方案对比决定 |

**股票池每月重构机制：** 股票池并非静态的。每月初，系统根据最新的 S&P 500 成分股列表和过去 20 个交易日的日均成交额重新构建 Universe。历史上每个月的 Universe 成员资格被完整记录在 `universe_membership_history` 表中，以支持回测的可回放性。

### 3.2 多维实验矩阵 (v3.0 新增)

v2.0 将核心问题单一地定义为 "5D Forward Excess Return + Weekly Rebalance"，虽然逻辑自洽，但存在过度拟合特定时间周期的风险。如果 5D 这条线不稳，整个方案会被绑死。v3.0 建立了一个多维度的**实验矩阵（Experiment Matrix）**，以验证 Alpha 信号的稳健性，并为未来的策略扩展保留空间 [5]。

**预测目标 (Horizon) 矩阵。** 模型不应只针对单一的时间窗口进行训练，而应同时预测多个时间维度的相对超额收益：

| Horizon | 捕获的 Alpha 来源 | 交易成本敏感度 | 角色定位 |
|---------|----------------|------------|---------|
| 短期 (1D/2D) | 微观市场结构失衡、短期动量反转 | 极高 | 执行层辅助信号 |
| 中期 (5D/10D) | 财报后漂移 (PEAD)、分析师评级调整、中期因子动量 | 中等 | 核心 Alpha 捕获窗口（v2.0 主线） |
| 长期 (20D/60D) | 宏观基本面变化、长期价值回归 | 低 | 容量扩展备选 |

**信号落地与仓位转换矩阵。** 从连续的预测分数转化为实际的投资组合权重，系统设计了从朴素到复杂的递进测试路径：

1. **Rank-based Equal Weighting（朴素排序等权）：** 最透明的基线方案。直接将模型预测排名前 10% 的股票等权重买入。这用于纯粹评估"模型选股能力"本身，不受任何优化器黑盒的干扰。
2. **Volatility-scaled Rank Weighting（波动率倒数加权）：** 在排序的基础上，根据个股的 60 日历史波动率的倒数进行加权（Risk Parity 的简化版）。这能有效防止高波动率股票在组合中占据不成比例的风险敞口。
3. **Black-Litterman 稳健优化：** 引入完整的协方差矩阵和多重约束。只有当这种复杂优化器在扣除交易成本后的表现**统计显著地优于**前两种朴素方案时，才会被采纳为最终的生产方案。

通过这种矩阵式实验设计，系统可以清晰地拆解超额收益的来源：多少归功于特征工程？多少归功于模型非线性？多少归功于组合优化器？避免了"一锅炖"导致的归因困难。

### 3.3 三层基线对比模型体系

为了验证复杂模型的真实增益，系统建立了严格的三层模型对比体系。只有当高层模型在统计上显著优于基线模型时，才会被纳入生产环境。

**第一层：Baseline（基线模型）。** 使用 Ridge 回归和传统的动量/价值多因子排序打分。这确立了系统能够获取的最低 Alpha 下限。如果连线性模型都无法从特征中提取出有意义的信号，那么问题出在特征和标签设计上，而不是模型不够复杂。

**第二层：Tree 模型（挑战者 A）。** 采用 XGBoost 或 LightGBM，引入非线性特征交互。在大量实证研究中，树模型通常是金融预测中性价比最高的模型层 — 它能捕获特征间的非线性关系，同时训练速度快、可解释性相对较好 [6]。

**第三层：Deep Learning（挑战者 B）。** 采用 LSTM 或 Temporal Fusion Transformer (TFT) 处理时间序列特征。通过与前两层的严格对比（如 IC、RankIC 的提升度和统计显著性检验），评估深度学习在处理非平稳金融数据时是否真的物有所值。金融时间序列的非平稳性使得 LSTM 极易过拟合，因此必须用数据说话，而非盲目迷信复杂网络。

### 3.4 模型评估指标体系

模型评估不使用传统机器学习中的 MSE 或准确率，而是采用量化研究专用的信号质量指标：

| 指标 | 定义 | 合格阈值 | 含义 |
|------|------|----------|------|
| IC (Information Coefficient) | 预测值与实际收益的 Pearson 相关系数 | > 0.03 | 信号具有预测能力 |
| RankIC | 预测排名与实际收益排名的 Spearman 相关系数 | > 0.05 | 排序能力（对排序问题更重要） |
| IC IR (IC Information Ratio) | IC 均值 / IC 标准差 | > 0.5 | 信号稳定性 |
| Hit Rate | 预测方向正确的比例 | > 52% | 方向性准确度 |
| Top Decile Excess Return | 前 10% 组合相对基准的超额收益 | > 5% 年化 | 策略实际盈利能力 |
| Turnover | 每期调仓比例 | < 30% 周度 | 交易成本可控性 |
| 成本后 Alpha | 扣除交易成本和滑点后的超额收益 | > 0 | 策略在真实世界中是否盈利 |

---

## 4. 特征工程与数据规范

### 4.1 特征分类体系

特征工程是量化模型中比网络结构更重要的环节。系统将特征划分为四大类，共计约 60-80 个候选特征：

| 特征类别 | 代表特征 | 计算频率 | 数据源 |
|----------|----------|----------|--------|
| 价格动量 | 5D/10D/20D/60D 收益率、动量排名、52 周高低比 | 日频 | 行情数据 |
| 波动率 | 20D 已实现波动率、ATR、Garman-Klass 波动率、波动率排名 | 日频 | 行情数据 |
| 成交量 | 成交量变化率、OBV、VWAP 偏离度、Amihud 非流动性指标 | 日频 | 行情数据 |
| 技术指标 | RSI(14)、MACD、布林带宽度、ADX、Stochastic Oscillator | 日频 | 行情数据 |
| 基本面 (PIT) | P/E、P/B、ROE、营收增长率、自由现金流收益率 | 季频 | 财报数据 |
| 宏观环境 | VIX、10Y 国债收益率、信用利差、联邦基金利率 | 日频 | FRED |

### 4.2 数据质量保障流程

所有特征在进入模型之前，必须通过严格的数据质量检查：

**缺失值处理：** 对于行情数据，单日缺失率超过 5% 触发告警；对于基本面数据，采用前向填充（Forward Fill），但严格标记填充标志位，确保模型知道哪些数据是真实观测值、哪些是填充值。

**极端值处理：** 对每个特征计算横截面 Z-score，绝对值超过 5 的数据点被 Winsorize（缩尾处理）至 5 倍标准差边界。这既保留了极端信息的方向性，又防止了单个异常值对模型训练的过度影响。

**标准化：** 所有特征在横截面维度上进行 Rank 标准化（转换为 0-1 之间的百分位排名），消除量纲差异和极端值影响。Rank 标准化相比 Z-score 标准化，对金融数据中常见的厚尾分布更加稳健。

---

## 5. 回测规范与实验制度

回测不仅是一个功能模块，更是决定量化策略生死的"研究制度"。QuantEdge 摒弃了简单的静态切分，实施机构级的回测规范，以最大限度地消除过拟合和各类偏差 [7]。

### 5.1 消除回测偏差的核心机制

**幸存者偏差 (Survivorship Bias) 处理。** 回测数据必须包含历史退市、被收购或破产的股票数据。如果仅在当前存在的 S&P 500 股票上进行回测，历史收益将被严重高估。系统通过维护 `universe_membership_history` 表，确保每个历史时间点使用的是当时的成分股列表 [8]。

**Point-in-Time (PIT) 数据原则。** 特别是在使用基本面数据（如财报指标）时，系统严格使用当时市场实际可见的数据，而非事后修正（Restated）的数据。例如，某公司 2024 年 Q1 财报于 2024 年 4 月 25 日发布，那么在 4 月 24 日之前的回测中，该季度数据不可使用。数据库中通过 `knowledge_time` 字段实现这一隔离。

**公司行为 (Corporate Actions) 完整处理。** 回测引擎内置对以下事件的自动调整机制：

| 公司行为 | 处理方式 | 影响范围 |
|----------|----------|----------|
| 股票拆分 (Splits) | 按拆分比例调整历史价格和成交量 | 价格序列连续性 |
| 特殊分红 (Special Dividends) | 调整除权价格 | 收益率计算准确性 |
| 代码变更 (Ticker Changes) | 维护 old_ticker 到 new_ticker 映射 | 数据连续性 |
| 退市/被收购 | 保留历史数据，标记退市日期和原因 | 幸存者偏差消除 |

### 5.2 Walk-Forward 滚动窗口验证

放弃单次的 Train/Test 切分。采用滚动窗口来评估模型在不同市场环境（Regime）下的稳定性 [9]。

**窗口设计：** 训练窗口固定为 3 年（约 750 个交易日），验证窗口为 6 个月（约 125 个交易日），测试窗口为 6 个月。窗口每 6 个月向前滚动一次。

**示例时间线：**

| 窗口编号 | 训练期 | 验证期 | 测试期 |
|----------|--------|--------|--------|
| Window 1 | 2018.01 - 2020.12 | 2021.01 - 2021.06 | 2021.07 - 2021.12 |
| Window 2 | 2018.07 - 2021.06 | 2021.07 - 2021.12 | 2022.01 - 2022.06 |
| Window 3 | 2019.01 - 2021.12 | 2022.01 - 2022.06 | 2022.07 - 2022.12 |
| Window 4 | 2019.07 - 2022.06 | 2022.07 - 2022.12 | 2023.01 - 2023.06 |
| Window 5 | 2020.01 - 2022.12 | 2023.01 - 2023.06 | 2023.07 - 2023.12 |

每个窗口独立训练模型并在样本外进行测试，最终汇总所有窗口的 IC、RankIC 和收益曲线，评估模型的跨周期稳健性。如果模型仅在某一个窗口表现优异而在其他窗口失效，则说明该模型缺乏泛化能力。

### 5.3 Benchmark 基准体系

所有回测结果必须与以下基准进行对比：

| 基准 | 用途 |
|------|------|
| SPY (S&P 500 ETF) | 市场基准，评估是否跑赢大盘 |
| Equal-Weight S&P 500 | 消除市值加权偏差的基准 |
| Sector-Neutral Baseline | 评估选股能力是否独立于行业配置 |

---

## 6. 高级统计检验与防过拟合框架

在金融时间序列预测中，当你测试了成百上千个特征和模型组合后，总能找到一个在历史数据上表现完美的策略。这种**多重测试偏差（Multiple Testing Bias）**是导致量化策略实盘失效的头号杀手。v2.0 引入了 IC t-test、Bootstrap 和 Bonferroni 校正，v3.0 进一步升级为学术界最前沿的统计检验框架 [10]。

### 6.1 降阶夏普比率 (Deflated Sharpe Ratio, DSR)

传统的夏普比率检验（如 t-test）假设你只测试了一个策略。Bailey 和 Lopez de Prado (2014) 提出的 Deflated Sharpe Ratio (DSR) 从数学上修正了多重测试带来的偏差 [11]。相比传统的 Bonferroni/Sidak 校正，DSR 的优势在于它不仅考虑了测试次数，还同时建模了选择偏差和夏普比率估计本身的不确定性，包括收益率分布的偏度和峰度。

**实施步骤：**

**Step 1 — 记录所有试验。** 系统通过 MLflow 强制记录研究员运行过的每一次模型训练和回测结果（包括失败的实验）。这是 DSR 的数据基础：没有完整的试验记录，就无法计算多重测试的惩罚力度。

**Step 2 — 聚类独立试验数 (N)。** 由于很多实验是高度相关的（如仅微调了学习率），直接用总试验次数会导致过度惩罚。系统使用最优聚类数（ONC）算法对所有回测的收益率序列进行聚类，估算出真正独立的试验次数 N [11]。

**Step 3 — 计算期望最大夏普比率。** 基于 False Strategy Theorem (FST)，利用独立试验数 N 和夏普比率的横截面方差，计算出在纯随机情况下，N 次试验能"碰巧"得到的最高夏普比率阈值 E[max(SR)]。这个阈值随 N 的增大而单调递增 — 试验越多，随机产生高夏普的概率越大 [12]。

**Step 4 — DSR 显著性判定。** 只有当候选策略的实际夏普比率显著高于上述阈值（经偏度和峰度修正后，p-value < 0.05），该策略才被认为是真实的 Alpha，而非数据挖掘的产物。

### 6.2 Hansen 的卓越预测能力检验 (SPA Test)

除了 DSR，系统还集成了 Hansen (2005) 提出的 Superior Predictive Ability (SPA) Test [13]。相比于 White's Reality Check，SPA 检验通过对测试统计量进行学生化（Studentization）处理，使其对包含大量"劣质模型"的测试集更加稳健。当我们在 Baseline（Ridge 回归）、Tree（XGBoost）和 Deep Learning（LSTM）三个模型家族中选择时，使用 SPA Test 严格检验复杂模型是否在统计意义上提供了超越简单基线的增量预测能力。如果 SPA Test 的 p-value 不显著，则无论复杂模型在点估计上看起来多好，系统都将采用更简单的 Baseline 模型 — 因为那个"更好"的表现很可能只是噪声。

### 6.3 统计检验合格标准汇总

| 检验方法 | 目的 | 通过标准 | 学术来源 |
|----------|------|----------|----------|
| IC t-test | 检验平均 IC 是否显著异于零 | p-value < 0.05 | 基础统计检验 |
| Bootstrap 置信区间 | 评估年化收益的置信区间 | 95% CI 下限 > 0 | 非参数检验 |
| Deflated Sharpe Ratio | 校正多重测试后的夏普比率显著性 | DSR p-value < 0.05 | Bailey & Lopez de Prado (2014) |
| Hansen SPA Test | 检验复杂模型是否显著优于基线 | SPA p-value < 0.05 | Hansen (2005) |

---

## 7. 微观结构与交易成本建模

将成本简单设定为"万分之五手续费"是业余回测的标志。真实的交易成本（Implementation Shortfall）包含了显性佣金、买卖价差（Bid-Ask Spread）和市场冲击成本（Market Impact）。v2.0 引入了基于波动率和 ADV 的简单冲击模型，v3.0 进一步升级为基于 Almgren-Chriss 框架的非线性成本模型 [14]。

### 7.1 非线性市场冲击模型

根据经典的微观结构理论，市场冲击并非与订单规模呈线性关系，而是呈现平方根法则（Square Root Law）。系统在回测中采用以下成本函数：

> **临时冲击 (Temporary Impact)：** g(v) = η × σ × (v / ADV)^β

> **永久冲击 (Permanent Impact)：** h(v) = γ × σ × (v / ADV)

其中 σ 为该股票近期的日波动率（高波动率股票的冲击成本更大），v / ADV 为订单量占该股票日均成交量的比例，β 为冲击弹性系数（学术界和业界实证研究通常将其设定在 0.5 到 0.6 之间，即平方根关系），η 为流动性提供者的风险厌恶系数（需通过高频交易数据进行校准）。永久冲击与订单规模呈线性关系，反映了大额订单对市场价格发现过程的持久影响。

### 7.2 真实的执行价格假设

回测中最常见的陷阱是假设能在"收盘价"成交。QuantEdge 对执行价格进行了严格的约束：

**信号生成：** 周五收盘后（T 日），基于截至周五收盘的数据生成下周的预测信号。

**执行假设：** 订单在周一（T+1 日）执行，默认采用周一的开盘价 (Open Price) 或周一的 VWAP (成交量加权平均价) 作为成交价格基准。

**极端行情处理：** 如果周一开盘出现跳空（Gap）且偏离周五收盘价超过 2%，或者周一的成交量异常萎缩（低于 20 日均量的 30%），系统将动态调高该笔交易的滑点惩罚，模拟真实世界中流动性枯竭时的执行困境。

### 7.3 总成本分解

| 成本组成 | 估算方法 | 典型量级 (大盘股) |
|----------|----------|----------------|
| 佣金 (Commission) | 固定费率，约 $0.005/股 | 1-2 bps |
| 买卖价差 (Spread) | 与流动性成反比，取 Bid-Ask 中点 | 2-5 bps |
| 临时冲击 (Temporary Impact) | Almgren-Chriss 平方根模型 | 5-20 bps |
| 永久冲击 (Permanent Impact) | 线性模型 | 2-10 bps |
| 时机成本 (Delay Cost) | 信号延迟 + 隔夜跳空 | 变动 |

---

## 8. 稳健组合优化与仓位管理

将预测信号转化为实际交易仓位是量化投资中最脆弱的一环。传统的马科维茨（MPT）均值-方差优化对预期收益率输入极其敏感，模型预测的微小噪声会被优化器无限放大 [15]。QuantEdge 采用**稳健组合优化（Robust Portfolio Optimization）**技术来解决这一问题。

### 8.1 协方差矩阵的稳健估计

当股票数量增加时，样本协方差矩阵的估计误差会急剧上升。系统采用 **Ledoit-Wolf Shrinkage（收缩估计）** 技术，将样本协方差矩阵向结构化目标（如恒定相关性矩阵）进行收缩，从而获得更稳定、条件数更优的协方差估计 [16]。

> **Shrinkage 公式：** Σ_shrunk = δ × F + (1 - δ) × S

其中 S 为样本协方差矩阵，F 为结构化目标矩阵，δ 为最优收缩强度（由 Ledoit-Wolf 公式自动确定）。

### 8.2 预期收益率的贝叶斯修正 (Black-Litterman)

为缓解 MPT 对输入信号的敏感性，系统引入 **Black-Litterman 模型** [17]。该模型将市场均衡隐含收益率（作为先验分布）与 AI 模型输出的预测信号（作为投资者观点）进行贝叶斯融合。置信度高的预测信号将被赋予更大权重，而低置信度信号将被拉回市场均衡水平，从而生成更加平滑和稳健的目标权重。

### 8.3 带有现实约束的优化求解

优化目标函数从单纯的"最大化夏普比率"升级为包含多重惩罚项和约束条件的凸优化问题，使用 CVXPY 求解：

| 约束/惩罚 | 具体设定 | 设计理由 |
|----------|----------|----------|
| 换手率惩罚 | 目标函数加入 λ × abs(w_new - w_old) | 抑制高频无效交易 |
| 个股仓位上限 | w_i <= min(10%, 5 × ADV_i / PortfolioSize) | 固定比例 + 流动性双重约束 |
| 行业暴露 | 单一 GICS 行业权重偏离基准不超过 +/- 10% | 避免行业集中风险 |
| Beta 暴露 | 组合 Beta 控制在 0.8 - 1.2 之间 | 控制市场风险暴露 |
| 做多约束 | w_i >= 0, sum(w_i) = 1 | Long-only 全仓投资 |
| 最低持仓数 | 组合至少包含 20 只股票 | 确保分散化 |

### 8.4 朴素基线对比 (v3.0 新增)

v3.0 强调：Black-Litterman 稳健优化器只有在统计显著地优于朴素方案时才被采纳。系统在每次回测中同时运行三种仓位方案（等权、波动率倒数加权、BL 优化），并通过 SPA Test 检验优化器是否提供了真实的增量价值。如果优化器的增量不显著，则生产环境将采用更简单、更透明的波动率倒数加权方案 — 因为简单方案的可解释性和稳定性在很多情况下比"理论最优"更有实际价值。

---

## 9. 四层系统化风控体系

风控不应仅仅是"跌了 8% 就止损"的简单规则，而应贯穿于量化系统的整个生命周期。QuantEdge 构建了从数据到执行的**四层风控框架**。

### Layer 1: 研究前风控 (Pre-research Risk)

关注数据质量，是整个系统的第一道防线。每日数据拉取后，自动检查缺失率（单日缺失股票数 / Universe 总数），缺失率超过 5% 触发黄色告警，超过 15% 触发红色告警并暂停后续管道。对每个特征的横截面分布进行 Kolmogorov-Smirnov 检验，与过去 60 日的历史分布进行比较，显著偏离（p-value < 0.01）触发特征级告警。监控 API 响应时间、返回数据量和错误率，数据源连续 3 次失败自动切换到备用源。

### Layer 2: 信号风控 (Signal Risk)

监控模型输出的健康度，防止"模型静默失效"。定期绘制 Calibration Plot，检查模型预测的概率是否与实际频率一致，校准偏差超过阈值时触发模型复审。持续跟踪过去 20 期的滚动 IC，当滚动 IC 连续 4 期低于历史均值的 50% 时，系统自动将该模型降权或切换到 Baseline 模型。当 Champion 模型的滚动表现持续低于 Challenger 模型时，系统触发自动切换流程（需人工确认）。

### Layer 3: 组合风控 (Portfolio Risk)

传统的风控核心，但比简单的止损规则更加系统化。

| 风控维度 | 具体规则 | 触发动作 |
|----------|----------|----------|
| 个股集中度 | 单股权重 > 10% | 强制减仓至 10% |
| 行业集中度 | 单行业偏离基准 > 15% | 强制再平衡 |
| 组合 Beta | Beta > 1.3 或 < 0.7 | 调整至目标区间 |
| 持仓相关性 | 任意两股票 60D 相关性 > 0.85 | 保留信号更强的一只 |
| 周度换手率 | 单周换手 > 40% | 截断调仓至 40% |
| CVaR (99%) | 组合 99% CVaR 超过 -5% | 整体降仓 20% |
| 压力测试 | 模拟 2020.03 级别下跌 | 评估最大损失，预警 |

### Layer 4: 运行风控 (Operational Risk)

保障生产系统的稳定性和可追溯性。每个 DAG 任务设置超时告警和失败重试机制，关键任务（如信号生成）失败后，系统自动进入"维持现有仓位"模式，不执行任何调仓。当主数据源不可用时，系统按照预设的降级策略运行 — 使用缓存数据或备用数据源，并在 Dashboard 上明确标注"降级模式运行中"。每一次仓位变动都记录完整的决策链：使用了哪个模型版本、哪个特征批次、哪个优化配置、信号生成时间和执行假设。

---

## 10. MLOps 分期实施策略

对于一个初创的量化团队，首日上线全套的 MLOps 架构（Airflow + Feast + MLflow + Celery + DVC）存在严重的过度工程化（Over-engineering）风险。v2.0 版本一次性列出了所有治理组件，v3.0 将其改为与项目生命周期严格对齐的分期实施策略 [18]。

### Stage 1: 轻量级研究架构 (MVP 期, 0-3 月)

目标是最低成本证明 Alpha 的存在，快速迭代特征和模型。

**任务调度：** 放弃 Airflow，使用简单的 Python schedule 库或操作系统的 Cron 配合 Shell 脚本。

**特征管理：** 放弃重量级的 Feast，使用标准的 Pandas DataFrame 配合 Parquet 文件存储，通过严格的代码 Review 和 Git 版本控制来保证训练与推理逻辑的一致性。

**实验追踪：** 保留 MLflow 作为整个 MVP 阶段唯一不可妥协的 MLOps 组件 — 没有实验追踪，就无法进行后续的 DSR 检验，研究过程将陷入混乱。

### Stage 2: 准生产级架构 (规模化期, 4-8 月)

当策略在实盘或影子模式下证明了持续盈利能力，且特征数量超过 100 个、团队规模扩大时，逐步引入治理组件。

**引入 Airflow：** 当数据管道开始出现复杂的依赖关系（如必须先等财报数据清洗完毕，再计算基本面因子，最后合成多因子得分）时，引入 Airflow 进行 DAG 编排和失败重试管理。

**引入 Redis + Celery：** 当模型训练时间过长，开始阻塞 API 响应时，引入异步任务队列，将耗时的回测和训练任务剥离到后台 Worker 节点。

**MLflow 升级：** 从纯实验追踪升级为 Model Registry，实行 Champion/Challenger 灰度发布。

### Stage 3: 机构级架构 (全面治理期, 9 月+)

当系统管理的资金规模显著增加，任何一个微小的 Training-Serving Skew 都可能导致巨额损失时，完成终极进化。

**引入 Feast (Feature Store)：** 彻底分离特征的离线计算（用于训练）和在线服务（用于推理），确保实盘使用的特征计算逻辑与回测完全一致。

**引入 DVC：** 实现数据版本控制，确保不仅代码可追溯，连训练用的历史数据集也能精确复现。

**引入 Prometheus + Grafana：** 实现全链路监控，包括数据管道延迟、模型推理延迟、API 响应时间和系统资源使用率。

---

## 11. 用户解释层与信任构建

对于面向终端用户的产品，黑盒模型是无法建立信任的。如果系统只是简单地给出一个信号分数，一旦发生回撤，用户将立刻流失。QuantEdge 在产品设计上，将可解释性（Explainability）作为核心竞争力而非锦上添花 [19]。

### 11.1 推荐逻辑的可视化拆解

在个股的预测详情页，系统必须回答用户最关心的三个问题：

**"为什么这只股票得分高？"（驱动因子归因）。** 利用 SHAP (SHapley Additive exPlanations) 值，将模型的黑盒预测分数拆解为具体的特征贡献。例如，Dashboard 会清晰显示：AAPL 的高预测得分中，40% 来源于"近期强劲的动量突破"，35% 来源于"超预期的财报营收增长"，但被"较高的估值乘数 (P/E)"抵消了 15%。

**"这个信号和过去相比是增强还是减弱？"（信号时序演变）。** 提供该股票过去 60 天的模型预测分数走势图。用户可以直观地看到系统对该股票的看好程度是在逐步升温，还是刚刚经历了一次突然的跃升。

**"置信区间对应什么风险？"（不确定性量化）。** 模型输出的不仅是一个点估计（如预期超额收益 2.5%），而是一个概率分布（如 95% 置信区间为 [-1.2%, 6.2%]）。区间越宽，代表模型对该预测的把握越低，提示用户该标的可能面临较高的不确定性。

### 11.2 失败案例的坦诚披露

真正的专业性体现在敢于展示系统的局限性。系统特设了"模型表现回溯"模块：

**历史胜率：** 真实展示模型过去 1 年对该股票预测的方向准确率。

**典型失败场景分析：** 总结并展示模型在什么市场环境（Regime）下最容易出错。例如，"本模型在 2022 年的高通胀加息周期中，对成长股的预测误差显著放大，请在类似宏观环境下谨慎参考。" 这种坦诚的风险披露，反而能极大增强高净值用户的长期信任。

---

## 12. 数据库设计与可回放能力

量化系统的数据库设计必须支持"时光倒流"（Time-Travel）。如果系统无法精确重现"2024 年 3 月 1 日那个时刻，系统基于当时可用的数据做出了什么决策"，那么这个系统就是不可调试的。

### 12.1 核心表结构

系统包含 11 张核心数据表，相比 v1.0 新增了 `UNIVERSE_MEMBERSHIP`、`CORPORATE_ACTIONS`、`FEATURE_STORE`、`MODEL_REGISTRY` 和 `AUDIT_LOG` 五张关键表。

**双时间戳隔离：** 所有与时间敏感的数据表都包含两个时间字段 — `event_time`（事件实际发生的时间）和 `knowledge_time`（系统获知该事件的时间）。

**版本关联：** 每条预测记录（PREDICTIONS）必须关联 `model_version_id`（来自 MODEL_REGISTRY）和 `feature_batch_id`（来自 FEATURE_STORE），确保任何历史预测都可以追溯到"是基于哪版数据、哪版模型、哪次特征计算生成的"。

**快照能力：** `universe_membership_history` 记录每个月的股票池成员变化；`backtest_config_snapshot` 以 JSON 格式保存每次回测的完整配置，确保回测结果可完整复现。

### 12.2 核心表设计一览

| 表名 | 核心字段 | 用途 |
|------|----------|------|
| STOCKS | ticker, company_name, sector, industry, ipo_date, delist_date | 股票基础信息（含退市标记） |
| STOCK_PRICES | ticker, trade_date, open, high, low, close, adj_close, volume, knowledge_time | 行情数据（双时间戳） |
| FUNDAMENTALS_PIT | ticker, fiscal_period, metric_name, metric_value, event_time, knowledge_time | PIT 基本面数据 |
| UNIVERSE_MEMBERSHIP | ticker, effective_date, end_date, index_name | 历史成分股成员资格 |
| CORPORATE_ACTIONS | ticker, action_type, ex_date, ratio, old_ticker, new_ticker | 公司行为记录 |
| FEATURE_STORE | ticker, calc_date, feature_name, feature_value, batch_id | 特征数据（含批次号） |
| MODEL_REGISTRY | model_id, model_name, version, train_start, train_end, metrics_json, status | 模型版本管理 |
| PREDICTIONS | ticker, signal_date, model_version_id, feature_batch_id, pred_score, pred_rank | 预测记录（完整关联） |
| PORTFOLIOS | portfolio_id, user_id, strategy_name, config_json | 组合配置 |
| BACKTEST_RESULTS | backtest_id, config_snapshot_json, metrics_json, equity_curve | 回测结果（含完整快照） |
| AUDIT_LOG | timestamp, action, actor, model_version_id, feature_batch_id, details_json | 审计日志 |

### 12.3 存储策略

| 数据类型 | 存储引擎 | 理由 |
|----------|----------|------|
| 行情 + 技术指标 | TimescaleDB (超级表) | 时序数据压缩和查询优化 |
| 特征数据 | TimescaleDB + Feast (Stage 3) | 统一管理，支持 PIT 查询 |
| 用户 + 策略 + 审计 | PostgreSQL | 关系型数据，事务一致性 |
| 热点数据 + 任务队列 | Redis | 低延迟缓存和 Celery Broker |

---

## 13. 系统架构总览

系统采用**六层架构**设计：

| 架构层 | 核心组件 | 职责 |
|--------|----------|------|
| 前端展现层 | React.js + TypeScript + ECharts | 用户交互与数据可视化 |
| 后端 API 层 | FastAPI + Celery | RESTful API 网关 + 异步任务 |
| 量化研究引擎 | 三层模型 + 稳健优化器 + 回测引擎 | Alpha 研究与信号生成 |
| MLOps 治理层 | MLflow (必选) + Airflow/Feast (分期) | 任务编排 + 特征管理 + 模型治理 |
| 风控层 | 四层风控引擎 | 数据/信号/组合/运行全链路风控 |
| 数据存储层 | TimescaleDB + PostgreSQL + Redis | PIT 时序数据 + 关系数据 + 缓存 |

**设计原则：**

**关注点分离 (Separation of Concerns)：** 每一层只负责自己的职责，通过明确定义的接口与其他层通信。研究引擎不关心数据如何存储，前端不关心模型如何训练。

**可重现性 (Reproducibility)：** 从数据版本、特征版本到模型版本，系统的每一个环节都可追溯和复现。

**Fail-Safe Defaults：** 当任何组件失败时，系统默认进入安全模式（维持现有仓位，不执行新交易），而非继续使用可能有问题的数据或模型。

---

## 14. 技术栈总览

| 领域 | 技术选型 | 核心用途 | 选型理由 |
|------|----------|----------|----------|
| 前端框架 | React.js + TypeScript | 单页应用 SPA | 类型安全，组件化开发 |
| 图表可视化 | ECharts + Lightweight Charts | 金融 K 线图 + 数据图表 | 高性能金融图表渲染 |
| 后端框架 | FastAPI (Python) | RESTful API | 异步高性能，自动文档 |
| 异步任务 | Celery + Redis (Stage 2+) | 模型训练/回测异步执行 | 避免阻塞 API 服务 |
| 数据编排 | Apache Airflow (Stage 2+) | 数据管道 DAG 编排 | 可视化 DAG，丰富的连接器 |
| 特征管理 | Feast (Stage 3) | Feature Store | 消除 Training-Serving Skew |
| 模型治理 | MLflow (Stage 1 起必选) | 实验追踪 + 模型注册 | 业界标准，开源免费 |
| 时序数据库 | TimescaleDB | PIT 行情 + 指标存储 | PostgreSQL 兼容，时序优化 |
| 关系数据库 | PostgreSQL | 用户/策略/审计数据 | 成熟稳定，ACID 事务 |
| 缓存 | Redis | 热点数据 + 消息队列 | 低延迟，多用途 |
| ML 框架 | scikit-learn + XGBoost + PyTorch | 三层模型训练 | 覆盖从线性到深度学习 |
| 可解释性 | SHAP | 模型预测归因 | 理论严谨，业界标准 |
| 组合优化 | PyPortfolioOpt + CVXPY | 稳健优化求解 | 支持 BL + Shrinkage + 约束 |
| 统计检验 | 自研 DSR + arch (SPA) | 多重测试校正 | 学术级防过拟合 |
| 容器化 | Docker + Docker Compose | 开发/部署环境一致性 | 环境隔离，快速部署 |
| 版本控制 | Git + DVC (Stage 3) | 代码 + 数据版本管理 | 数据版本化追踪 |

---

## 15. 项目实施路线图

### Phase 1: 证明 Alpha（第 1-10 周）

这是整个项目最关键的阶段。如果 Alpha 不成立，后续的一切产品化工作都没有意义。

| 里程碑 | 周期 | 交付物 |
|--------|------|----------|
| 数据采集与 PIT 清洗 | 第 1-2 周 | 干净的 PIT 行情数据 + 基本面数据 |
| 特征工程与 Parquet 存储 | 第 3-4 周 | 60-80 个候选特征 + 质量检查报告 |
| Baseline 模型训练 | 第 5 周 | Ridge 回归 + 多因子排序基线 |
| XGBoost/LightGBM 训练 | 第 6 周 | Tree 模型 + 与 Baseline 对比 |
| LSTM/TFT 训练 | 第 6-7 周 | 深度模型 + 与 Tree 模型对比 |
| 多 Horizon 实验矩阵 | 第 7-8 周 | 1D/5D/10D/20D 全维度对比 |
| Walk-Forward + DSR/SPA 检验 | 第 8-10 周 | 完整回测报告 + 统计检验结果 |

**Phase 1 准出标准：** 至少一个模型在 3 个以上 Walk-Forward 窗口中的平均 IC > 0.03，DSR p-value < 0.05，成本后年化超额 > 5%。

### Phase 2: 生产架构与模型治理（第 11-18 周）

| 里程碑 | 周期 | 交付物 |
|--------|------|----------|
| MLflow 实验追踪与模型注册 | 第 11-12 周 | 实验追踪 + Champion/Challenger |
| Airflow DAG 编排 | 第 13-14 周 | 数据 + 特征 + 推理 DAG |
| 四层风控引擎开发 | 第 13-14 周 | 风控规则引擎 + 告警系统 |
| 稳健组合优化器 | 第 15-16 周 | BL + Shrinkage + 约束优化 |
| 非线性交易成本模型 | 第 15-16 周 | Almgren-Chriss 回测集成 |
| Shadow Mode 灰度发布 | 第 17-18 周 | 模型并行运行验证 (4 周) |

### Phase 3: 产品化与用户系统（第 19-28 周）

| 里程碑 | 周期 | 交付物 |
|--------|------|----------|
| 前端 Dashboard 开发 | 第 19-21 周 | 市场概览 + 行情中心 |
| AI 预测中心与 SHAP 解释层 | 第 22-23 周 | 预测信号展示 + 因子归因 |
| 投资组合构建器 | 第 22-23 周 | 有效前沿 + 权重调整 |
| 回测配置与报告页面 | 第 24-25 周 | 用户自定义回测 |
| 用户系统与合规审查 | 第 26-27 周 | 注册/登录/订阅 + 合规声明 |
| 系统联调与压力测试 | 第 27-28 周 | 端到端测试 + 性能优化 |

### 团队配置建议

| 角色 | 人数 | 核心职责 |
|------|------|----------|
| 量化研究员 | 1-2 人 | Alpha 研究、特征工程、模型训练、统计检验 |
| 后端工程师 | 1-2 人 | API 开发、Airflow 编排、MLOps 分期实施 |
| 前端工程师 | 1 人 | React Dashboard + SHAP 可视化 |
| 数据工程师 | 1 人 | 数据管道、PIT 治理、数据库 |

---

## 16. 美国 SEC 合规与产品边界设计

如果 QuantEdge 未来面向美国真实用户提供服务，必须严格遵守《1940 年投资顾问法》(Investment Advisers Act of 1940)。v2.0 将合规写成了一个提醒，v3.0 将其升格为**产品边界设计约束** — 合规不是附录里的免责声明，而是从第一天起就必须嵌入产品设计的硬约束 [20]。

### 16.1 投资建议 (Advice) vs. 投资教育 (Education)

SEC 判定"投资建议"的关键标准包括：个性化（Personalization）、具体性（Specificity）和是否基于此收费。QuantEdge 必须将自己严格定位为一个**"量化数据分析与教育工具"**。

**产品设计红线（绝对禁止）：**

- 绝对不能在注册时询问用户的年龄、收入、风险承受能力，然后据此生成"为您量身定制的投资组合" — 这直接触发 fiduciary duty（信托责任）。
- UI 界面和邮件中绝对不能出现 "You should buy"、"Strong Buy"、"Action: Sell" 等指令性词汇。
- 平台绝对不触碰用户的资金，不提供一键下单到券商的 API 接口（除非平台本身取得了 Broker-Dealer 牌照）。

### 16.2 Safe Harbor 避风港设计规范

**合规的 UI 话术：**

将 "Buy Recommendation" 改为 "Model Output: Top Decile" 或 "High Alpha Probability"。将 "Your Portfolio" 改为 "Model Tracking Portfolio" 或 "Paper Trading Simulation"。强调结果是基于算法的客观输出，而非针对任何个人的主观建议。

**强制性免责声明 (Prominent Disclosures)：**

在所有包含模型预测分数的页面底部，必须以显著字体（不可折叠）展示合规声明：

> *"QuantEdge is a data analytics platform, not a registered investment adviser. The algorithmic outputs and model portfolios provided are for informational and educational purposes only and do not constitute personalized investment advice. All trading strategies involve risk of loss. Past performance of any algorithmic model does not guarantee future results."*

**收费模式的合规性：**

平台的订阅费在法律协议中必须被明确定义为"软件使用费"（Software/SaaS Access Fee）或"数据服务费"，而绝对不能被称为"咨询费"（Advisory Fee）。

### 16.3 合规对产品设计的具体影响

| 产品功能 | 合规安全区 | 红线区 |
|----------|----------|--------|
| 信号展示 | "Model Output: Top Decile" | "You should buy AAPL" |
| 组合展示 | "Model Tracking Portfolio" | "Your Portfolio" |
| 历史表现 | "Historical IC/RankIC" | "Guaranteed Returns" |
| 归因分析 | "SHAP Factor Attribution" | "Why you should invest" |
| 收费描述 | "SaaS/Data Service Fee" | "Advisory Fee" |
| 风险评估 | 不收集个人风险偏好 | 基于 risk profile 的个性化推荐 |

---

## 17. 参考文献

以下参考文献按照"学术论文 + 权威教材 + 官方文档"三层结构组织，确保每一项关键技术决策都有可溯源的学术或工程依据。

### 学术论文

[1]: Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*. John Wiley & Sons.

[2]: Dixon, M. F., Halperin, I., & Bilokon, P. (2020). *Machine Learning in Finance: From Theory to Practice*. Springer.

[3]: Harvey, C. R., Liu, Y., & Zhu, H. (2016). "...and the Cross-Section of Expected Returns." *The Review of Financial Studies*, 29(1), 5-68.

[4]: Gu, S., Kelly, B., & Xiu, D. (2020). "Empirical Asset Pricing via Machine Learning." *The Review of Financial Studies*, 33(5), 2223-2273.

[5]: Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*, Chapter 12: "Backtesting through Cross-Validation."

[6]: Elton, E. J., Gruber, M. J., & Blake, C. R. (1996). "Survivorship Bias and Mutual Fund Performance." *The Review of Financial Studies*, 9(4), 1097-1120.

[7]: Arlot, S., & Celisse, A. (2010). "A Survey of Cross-Validation Procedures for Model Selection." *Statistics Surveys*, 4, 40-79.

[8]: Harvey, C. R., & Liu, Y. (2015). "Backtesting." *The Journal of Portfolio Management*, 42(1), 13-28.

[9]: Bailey, D. H., & Lopez de Prado, M. (2014). "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality." *The Journal of Portfolio Management*, 40(5), 94-107.

[10]: Bailey, D. H., & Lopez de Prado, M. (2014). "The Strategy Approval Decision: A Sharpe Ratio Indifference Curve Approach." *Algorithmic Finance*, 3(1-2), 99-109.

[11]: Hansen, P. R. (2005). "A Test for Superior Predictive Ability." *Journal of Business & Economic Statistics*, 23(4), 365-380.

[12]: Almgren, R., & Chriss, N. (2001). "Optimal Execution of Portfolio Transactions." *Journal of Risk*, 3, 5-39.

[13]: Kolm, P. N., Tutuncu, R., & Fabozzi, F. J. (2014). "60 Years of Portfolio Optimization: Practical Challenges and Current Trends." *European Journal of Operational Research*, 234(2), 356-371.

[14]: Ledoit, O., & Wolf, M. (2004). "A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices." *Journal of Multivariate Analysis*, 88(2), 365-411.

[15]: Black, F., & Litterman, R. (1992). "Global Portfolio Optimization." *Financial Analysts Journal*, 48(5), 28-43.

[16]: Lundberg, S. M., & Lee, S. I. (2017). "A Unified Approach to Interpreting Model Predictions." *Advances in Neural Information Processing Systems*, 30.

### 工程实践与官方文档

[17]: FactSet. (2024). "FactSet Fundamentals Point-in-Time." https://www.factset.com/marketplace/catalog/product/factset-fundamentals-point-in-time

[18]: LSEG/Refinitiv. (2024). "Point in Time Fundamentals | Data Analytics." https://www.lseg.com/en/data-analytics/financial-data/company-data/fundamentals-data/point-in-time-fundamentals

[19]: Treveil, A., et al. (2020). *Introducing MLOps: How to Scale Machine Learning in the Enterprise*. O'Reilly Media.

[20]: U.S. Securities and Exchange Commission. (2017). *Robo-Advisers: IM Guidance Update*. Division of Investment Management, No. 2017-02.
