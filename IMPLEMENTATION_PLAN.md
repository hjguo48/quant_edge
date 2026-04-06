# QuantEdge 详细实施计划

**基于 QuantEdge.md v3.0 生成**
**生成日期：** 2026-03-28
**项目当前状态：** Phase 2 — Batch 8 全部完成，灰度 v2 自动运行中 (2026-04-06)

---

## 目录

1. [项目深度分析](#1-项目深度分析)
2. [项目目录结构规划](#2-项目目录结构规划)
3. [Phase 1: 证明 Alpha（第 1-10 周）](#3-phase-1-证明-alpha第-1-10-周)
4. [Phase 2: 生产架构与模型治理（第 11-18 周）](#4-phase-2-生产架构与模型治理第-11-18-周)
5. [Phase 3: 产品化与用户系统（第 19-28 周）](#5-phase-3-产品化与用户系统第-19-28-周)
6. [技术栈与依赖清单](#6-技术栈与依赖清单)
7. [数据库核心表设计](#7-数据库核心表设计)
8. [API 接口设计](#8-api-接口设计)
9. [风险评估与应对](#9-风险评估与应对)
10. [关键决策检查点](#10-关键决策检查点)

---

## 1. 项目深度分析

### 1.1 核心定位

QuantEdge 是一个**研究驱动的机构级美股量化生产系统**。核心理念是"先证明策略有效，再做产品封装"，这与传统 fintech 项目"先画产品再填策略"的路径截然不同。系统价值不取决于 UI，而取决于 Alpha 是否真实存在、是否可复现、是否在扣除交易成本后持续盈利。

### 1.2 五大核心技术挑战

| # | 挑战 | 难度 | 说明 |
|---|------|------|------|
| 1 | **PIT 数据治理** | 高 | 双时间戳隔离是系统基石，引入未来函数则一切回测无效 |
| 2 | **多重测试防过拟合** | 高 | DSR + SPA 检验需要从 Day 1 记录所有实验，事后无法补救 |
| 3 | **非线性交易成本建模** | 中 | Almgren-Chriss 平方根冲击模型需要校准，固定费率会严重高估策略盈利 |
| 4 | **稳健组合优化** | 高 | Ledoit-Wolf Shrinkage + Black-Litterman 工程实现远比理论复杂 |
| 5 | **合规嵌入产品设计** | 中 | SEC 红线是 UI/API 硬约束，不是事后补的免责声明 |

### 1.3 风险矩阵

| 风险 | 概率 | 影响 | 应对策略 |
|------|------|------|----------|
| Alpha 不成立 | 中 | **致命** | Phase 1 硬性准出标准，不通过则 pivot 或终止 |
| 免费数据源质量不足 | 高 | 中 | Tier 1 接受瑕疵但预留 PIT 接口，验证通过后升级 |
| 过度工程化拖慢进度 | 高 | 高 | MLOps 严格分期，MVP 阶段只用 MLflow |
| 深度模型过拟合 | 高 | 中 | 三层基线对比 + DSR 检验硬性卡关 |
| 单人/小团队交付压力 | 高 | 高 | 优先 Phase 1 核心路径，延迟非关键功能 |
| 协方差矩阵病态导致优化器崩溃 | 中 | 中 | Shrinkage 兜底 + 朴素方案保底 |

### 1.4 核心设计原则

1. **研究中心化：** 一切以证明 Alpha 为优先，产品功能服从于研究验证
2. **可重现性：** 数据版本、特征版本、模型版本全链路可追溯
3. **Fail-Safe Defaults：** 任何组件失败时默认进入安全模式，不执行新交易
4. **分期演进：** 数据源、MLOps、基础设施三条线独立分期升级
5. **统计严谨：** 任何策略结论必须通过 DSR/SPA 统计检验才可采信

---

## 2. 项目目录结构规划

```
quantedge/
├── pyproject.toml                  # 项目元数据与依赖管理 (Poetry/uv)
├── docker-compose.yml              # TimescaleDB + PostgreSQL + Redis + MLflow
├── .env.example                    # 环境变量模板 (API keys, DB 连接等)
├── .gitignore
├── alembic.ini                     # Alembic 数据库迁移配置
├── alembic/
│   ├── env.py
│   └── versions/                   # 数据库迁移版本文件
│
├── src/
│   ├── __init__.py
│   ├── config.py                   # 全局配置 (Pydantic Settings)
│   │
│   ├── data/                       # ===== 数据层 =====
│   │   ├── __init__.py
│   │   ├── sources/                # 数据源适配器（策略模式）
│   │   │   ├── __init__.py
│   │   │   ├── base.py             # 数据源抽象基类
│   │   │   ├── polygon.py          # Polygon.io 日频量价数据
│   │   │   ├── fred.py             # FRED 宏观经济数据
│   │   │   └── fmp.py              # FMP 基本面数据
│   │   ├── db/                     # 数据库层
│   │   │   ├── __init__.py
│   │   │   ├── models.py           # SQLAlchemy ORM (11 张核心表)
│   │   │   ├── session.py          # 数据库会话管理
│   │   │   └── pit.py              # PIT 查询封装 (强制 knowledge_time 约束)
│   │   ├── quality.py              # 数据质量检查 (缺失率/KS检验/Winsorize)
│   │   └── corporate_actions.py    # 公司行为处理 (拆分/分红/代码变更)
│   │
│   ├── universe/                   # ===== 股票池管理 =====
│   │   ├── __init__.py
│   │   ├── builder.py              # Universe 月度重构 (成分股 + ADV 过滤)
│   │   └── history.py              # 历史成分股成员资格管理
│   │
│   ├── features/                   # ===== 特征工程 =====
│   │   ├── __init__.py
│   │   ├── registry.py             # 特征注册表 (元数据管理)
│   │   ├── technical.py            # 价格动量 + 波动率 + 成交量 + 技术指标
│   │   ├── fundamental.py          # PIT 基本面特征
│   │   ├── macro.py                # 宏观环境特征
│   │   ├── preprocessing.py        # Z-score/Winsorize/Rank标准化/缺失值处理
│   │   └── pipeline.py             # 特征计算管道 (批量生成 + batch_id)
│   │
│   ├── labels/                     # ===== 标签工程 =====
│   │   ├── __init__.py
│   │   └── forward_returns.py      # 多 Horizon 前瞻超额收益 (1D/2D/5D/10D/20D/60D)
│   │
│   ├── models/                     # ===== 模型层 =====
│   │   ├── __init__.py
│   │   ├── base.py                 # 模型基类接口 (train/predict/evaluate)
│   │   ├── baseline.py             # Layer 1: Ridge 回归 + 多因子排序打分
│   │   ├── tree.py                 # Layer 2: XGBoost / LightGBM
│   │   ├── deep.py                 # Layer 3: LSTM / TFT (PyTorch)
│   │   ├── evaluation.py           # IC/RankIC/ICIR/HitRate/TopDecileReturn
│   │   └── experiment.py           # 实验矩阵管理 (Horizon × Model × Weighting)
│   │
│   ├── backtest/                   # ===== 回测引擎 =====
│   │   ├── __init__.py
│   │   ├── engine.py               # Walk-Forward 滚动窗口回测核心引擎
│   │   ├── cost_model.py           # Almgren-Chriss 非线性交易成本模型
│   │   ├── execution.py            # 执行假设 (T+1 开盘/VWAP, 跳空处理)
│   │   ├── benchmarks.py           # 基准体系 (SPY / Equal-Weight / Sector-Neutral)
│   │   └── report.py              # 回测报告生成 (收益曲线/回撤/月度归因)
│   │
│   ├── stats/                      # ===== 统计检验 =====
│   │   ├── __init__.py
│   │   ├── ic_test.py              # IC t-test (平均 IC 是否显著异于零)
│   │   ├── bootstrap.py            # Bootstrap 置信区间 (年化收益 95% CI)
│   │   ├── dsr.py                  # Deflated Sharpe Ratio (多重测试校正)
│   │   └── spa.py                  # Hansen SPA Test (模型优越性检验)
│   │
│   ├── portfolio/                  # ===== 组合优化 =====
│   │   ├── __init__.py
│   │   ├── equal_weight.py         # 方案 1: 朴素排序等权 (Top Decile)
│   │   ├── vol_weighted.py         # 方案 2: 波动率倒数加权
│   │   ├── black_litterman.py      # 方案 3: BL 稳健优化
│   │   ├── shrinkage.py            # Ledoit-Wolf 协方差矩阵收缩估计
│   │   └── constraints.py          # CVXPY 约束 (仓位/行业/Beta/换手率/最低持仓)
│   │
│   ├── risk/                       # ===== 四层风控引擎 =====
│   │   ├── __init__.py
│   │   ├── data_risk.py            # Layer 1: 数据风控 (缺失率/分布/API降级)
│   │   ├── signal_risk.py          # Layer 2: 信号风控 (滚动IC/校准/模型切换)
│   │   ├── portfolio_risk.py       # Layer 3: 组合风控 (集中度/Beta/CVaR/压力测试)
│   │   └── operational_risk.py     # Layer 4: 运行风控 (超时/Fail-Safe/审计)
│   │
│   └── api/                        # ===== API 层 (Phase 3) =====
│       ├── __init__.py
│       ├── main.py                 # FastAPI 应用入口
│       ├── routers/
│       │   ├── market.py           # 市场概览
│       │   ├── stocks.py           # 个股行情
│       │   ├── predictions.py      # 模型预测 + SHAP 解释
│       │   ├── portfolio.py        # 组合构建器
│       │   ├── backtest.py         # 回测配置与结果
│       │   └── auth.py             # 用户认证
│       ├── schemas/                # Pydantic 请求/响应模型
│       ├── deps.py                 # 依赖注入
│       └── middleware.py           # 合规中间件 (强制免责声明)
│
├── frontend/                       # ===== 前端 (Phase 3) =====
│   ├── package.json
│   ├── tsconfig.json
│   ├── src/
│   │   ├── App.tsx
│   │   ├── pages/
│   │   │   ├── Dashboard.tsx       # 市场概览
│   │   │   ├── StockDetail.tsx     # 个股详情 + K线
│   │   │   ├── Predictions.tsx     # AI 预测中心
│   │   │   ├── Portfolio.tsx       # 组合构建器
│   │   │   ├── Backtest.tsx        # 回测配置与报告
│   │   │   └── Login.tsx           # 用户登录
│   │   ├── components/
│   │   │   ├── charts/             # ECharts + Lightweight Charts 组件
│   │   │   ├── shap/              # SHAP 归因可视化组件
│   │   │   └── compliance/         # 合规声明组件 (不可折叠)
│   │   └── services/               # API 客户端
│   └── public/
│
├── notebooks/                      # ===== 研究 Notebook =====
│   ├── 01_data_eda.ipynb           # 数据探索性分析
│   ├── 02_feature_analysis.ipynb   # 特征 IC 分析
│   ├── 03_model_comparison.ipynb   # 模型层对比
│   ├── 04_backtest_report.ipynb    # 回测报告可视化
│   └── 05_stat_tests.ipynb         # 统计检验结果
│
├── tests/                          # ===== 测试 =====
│   ├── conftest.py
│   ├── test_data/
│   │   ├── test_sources.py
│   │   ├── test_pit.py             # PIT 查询正确性测试
│   │   └── test_quality.py
│   ├── test_features/
│   │   ├── test_technical.py
│   │   └── test_preprocessing.py
│   ├── test_models/
│   │   ├── test_baseline.py
│   │   └── test_evaluation.py
│   ├── test_backtest/
│   │   ├── test_engine.py
│   │   ├── test_cost_model.py
│   │   └── test_walkforward.py
│   ├── test_stats/
│   │   ├── test_dsr.py
│   │   └── test_spa.py
│   └── test_portfolio/
│       ├── test_shrinkage.py
│       └── test_constraints.py
│
├── scripts/                        # ===== 运维脚本 =====
│   ├── init_db.py                  # 数据库初始化
│   ├── fetch_data.py               # 全量数据拉取
│   ├── daily_update.py             # 日增量数据更新
│   └── run_backtest.py             # 命令行回测入口
│
└── mlruns/                         # MLflow 本地实验数据 (MVP 阶段)
```

---

## 3. Phase 1: 证明 Alpha（第 1-10 周）

> **核心目标：** 在样本外数据上证明策略具有真实、可复现、成本后仍为正的超额收益。
> **如果 Alpha 不成立，后续一切产品化工作都没有意义。**

### 3.1 第 1 周：项目基础设施搭建

**目标：** 搭建可运行的项目骨架，数据库就绪，MLflow 可用。

| 序号 | 任务 | 交付物 | 验收标准 |
|------|------|--------|----------|
| 1.1 | 初始化 Python 项目 | `pyproject.toml` + 虚拟环境 | `uv sync` 或 `poetry install` 成功 |
| 1.2 | 编写 `docker-compose.yml` | TimescaleDB + Redis + MLflow 容器编排 | `docker-compose up` 所有服务健康 |
| 1.3 | 编写全局配置 `config.py` | Pydantic Settings 配置类 | API keys、DB 连接、回测参数统一管理 |
| 1.4 | 定义 11 张核心数据表 ORM | `src/data/db/models.py` | SQLAlchemy 模型定义完成 |
| 1.5 | 创建 Alembic 初始迁移 | `alembic/versions/001_initial.py` | `alembic upgrade head` 建表成功 |
| 1.6 | 配置 MLflow tracking | `mlruns/` 本地目录 | `mlflow ui` 可访问，能记录实验 |
| 1.7 | 编写 `.gitignore` + `.env.example` | 环境文件 | 敏感信息不入库 |

**详细说明：**

**1.1 项目依赖（`pyproject.toml` 核心依赖列表）：**

```
# 数据层
sqlalchemy, alembic, psycopg2-binary, asyncpg

# 数据源
polygon-api-client, fredapi, requests

# 数据处理
pandas, numpy, pyarrow

# 特征工程
ta-lib 或 pandas-ta

# 模型
scikit-learn, xgboost, lightgbm, torch

# 组合优化
pypfopt, cvxpy

# 统计检验
scipy, arch, statsmodels

# 可解释性
shap

# MLOps
mlflow

# 工具
pydantic, pydantic-settings, python-dotenv, loguru
```

**1.4 核心数据表双时间戳设计要点：**

- `STOCK_PRICES` 和 `FUNDAMENTALS_PIT` 必须包含 `event_time` + `knowledge_time`
- `STOCK_PRICES` 建为 TimescaleDB hypertable，按 `trade_date` 分区
- `FEATURE_STORE` 必须包含 `batch_id`，关联特征计算批次
- `PREDICTIONS` 必须包含 `model_version_id` + `feature_batch_id`
- `AUDIT_LOG` 记录所有仓位变动的完整决策链

---

### 3.2 第 2 周：数据采集与 PIT 清洗管道

**目标：** 拉取 2018 年至今的完整历史数据，建立 PIT 查询能力。

| 序号 | 任务 | 交付物 | 验收标准 |
|------|------|--------|----------|
| 2.1 | 实现 Polygon.io 适配器 | `sources/polygon.py` | 拉取 S&P 500 全部成分股 2018-今 OHLCV |
| 2.2 | 实现 FRED 适配器 | `sources/fred.py` | VIX/10Y/信用利差/联邦基金利率入库 |
| 2.3 | 实现 FMP 适配器 | `sources/fmp.py` | 季度财报 + `knowledge_time` 记录 |
| 2.4 | 实现 PIT 查询封装 | `db/pit.py` | 所有查询强制 `knowledge_time <= t` |
| 2.5 | 实现数据质量检查 | `quality.py` | 缺失率 >5% 黄色告警, >15% 红色告警 |
| 2.6 | 实现公司行为处理 | `corporate_actions.py` | 拆分/分红/代码变更自动调整 |
| 2.7 | 实现 Universe 构建器 | `universe/builder.py` | 月度成分股 + ADV >$50M 过滤 |
| 2.8 | 数据入库 + 建超级表 | TimescaleDB hypertable | 查询性能验证 |

**详细说明：**

**2.1 Polygon.io 数据拉取规范：**
- 标的范围：当前 S&P 500 成分股 + 历史退市/被收购标的（如可获取）
- 时间范围：2018-01-01 至今
- 频率：日频
- 字段：open, high, low, close, volume, adjusted close
- 复权方式：使用 Polygon 提供的后复权价格（adj_close），同时保留原始价格
- 限流策略：遵守 API rate limit，使用指数退避重试

**2.3 FMP 基本面数据 PIT 处理：**
- 对每条财报数据，记录 `event_time`（财报截至日期，如 2024-03-31）
- 记录 `knowledge_time`（SEC EDGAR 实际发布时间）
- 如果 FMP 无法提供精确的发布时间，使用保守估计：`event_time + 45天`（SEC 10-Q 截止期限）

**2.4 PIT 查询封装核心逻辑（伪代码）：**
```python
def get_fundamentals_pit(ticker: str, as_of: datetime) -> DataFrame:
    """获取截至 as_of 时刻市场实际可见的基本面数据"""
    return query(
        "SELECT * FROM fundamentals_pit "
        "WHERE ticker = :ticker AND knowledge_time <= :as_of "
        "ORDER BY event_time DESC LIMIT 1"
    )
```

**2.5 数据质量检查清单：**
- 单日缺失率 = 缺失股票数 / Universe 总数
- 横截面 KS 检验：每日每特征 vs 过去 60 日分布，p < 0.01 告警
- 极端值检测：|Z-score| > 10 的数据点标记为异常
- API 健康度：连续 3 次失败标记数据源不可用

**手动验证点：**
- 随机抽取 10 只股票的 2020-Q1 财报数据
- 核对 `knowledge_time` 是否正确反映 SEC EDGAR 实际发布日期
- 确认 2020-03-23（COVID 崩盘最低点）当天的数据完整性

---

### 3.3 第 3-4 周：特征工程（60-80 个候选特征）

**目标：** 构建完整的候选特征集，通过标准化处理管道。

| 序号 | 任务 | 交付物 | 验收标准 |
|------|------|--------|----------|
| 3.1 | 实现价格动量特征 (~8个) | `features/technical.py` | 5D/10D/20D/60D 收益率, 52周高低比, 动量排名 |
| 3.2 | 实现波动率特征 (~6个) | `features/technical.py` | 20D实现波动率, ATR, GK波动率, 波动率排名 |
| 3.3 | 实现成交量特征 (~6个) | `features/technical.py` | 成交量变化率, OBV, VWAP偏离, Amihud |
| 3.4 | 实现技术指标特征 (~10个) | `features/technical.py` | RSI, MACD, 布林带宽度, ADX, Stochastic |
| 3.5 | 实现基本面特征 (~15个, PIT) | `features/fundamental.py` | P/E, P/B, ROE, 营收增长, FCF yield |
| 3.6 | 实现宏观特征 (~10个) | `features/macro.py` | VIX, 10Y-2Y利差, 信用利差, FFR |
| 3.7 | 实现多 Horizon 标签 | `labels/forward_returns.py` | 1D/2D/5D/10D/20D/60D 前瞻超额收益 |
| 3.8 | 实现预处理管道 | `features/preprocessing.py` | Z-score + Winsorize + Rank标准化 |
| 3.9 | 实现特征存储管道 | `features/pipeline.py` | Parquet 存储 + batch_id |
| 3.10 | 特征初步筛选 | Notebook 报告 | 淘汰 IC < 0.01 的无效特征 |

**详细说明：**

**特征完整清单（约 75 个候选特征）：**

**A. 价格动量特征 (8个)：**

| # | 特征名 | 计算公式 | 频率 |
|---|--------|----------|------|
| 1 | `ret_5d` | 过去 5 日收益率 | 日频 |
| 2 | `ret_10d` | 过去 10 日收益率 | 日频 |
| 3 | `ret_20d` | 过去 20 日收益率 | 日频 |
| 4 | `ret_60d` | 过去 60 日收益率 | 日频 |
| 5 | `high_52w_ratio` | 当前价格 / 52周最高价 | 日频 |
| 6 | `low_52w_ratio` | 当前价格 / 52周最低价 | 日频 |
| 7 | `momentum_rank_20d` | 20D 收益率横截面百分位排名 | 日频 |
| 8 | `momentum_rank_60d` | 60D 收益率横截面百分位排名 | 日频 |

**B. 波动率特征 (6个)：**

| # | 特征名 | 计算公式 | 频率 |
|---|--------|----------|------|
| 9 | `vol_20d` | 20 日收益率标准差 (年化) | 日频 |
| 10 | `vol_60d` | 60 日收益率标准差 (年化) | 日频 |
| 11 | `atr_14` | 14 日 ATR / 收盘价 | 日频 |
| 12 | `gk_vol` | Garman-Klass 波动率 (20日) | 日频 |
| 13 | `vol_rank` | 20D 波动率横截面排名 | 日频 |
| 14 | `vol_change` | 当前 20D 波动率 / 60D 波动率 | 日频 |

**C. 成交量特征 (6个)：**

| # | 特征名 | 计算公式 | 频率 |
|---|--------|----------|------|
| 15 | `volume_ratio_5d` | 5日均量 / 20日均量 | 日频 |
| 16 | `volume_ratio_20d` | 20日均量 / 60日均量 | 日频 |
| 17 | `obv_slope` | OBV 20日线性回归斜率 | 日频 |
| 18 | `vwap_deviation` | (Close - VWAP) / VWAP | 日频 |
| 19 | `amihud` | 日均 |Return| / Volume (20日均值) | 日频 |
| 20 | `turnover_rate` | Volume / Shares Outstanding | 日频 |

**D. 技术指标特征 (10个)：**

| # | 特征名 | 计算公式 | 频率 |
|---|--------|----------|------|
| 21 | `rsi_14` | 14 日 RSI | 日频 |
| 22 | `rsi_28` | 28 日 RSI | 日频 |
| 23 | `macd_signal` | MACD 信号线差值 | 日频 |
| 24 | `macd_histogram` | MACD 柱状图 | 日频 |
| 25 | `bb_width` | (上轨 - 下轨) / 中轨 | 日频 |
| 26 | `bb_position` | (Close - 下轨) / (上轨 - 下轨) | 日频 |
| 27 | `adx_14` | 14 日 ADX | 日频 |
| 28 | `stoch_k` | Stochastic %K | 日频 |
| 29 | `stoch_d` | Stochastic %D | 日频 |
| 30 | `cci_20` | 20 日 CCI | 日频 |

**E. 基本面特征 (15个，PIT 严格)：**

| # | 特征名 | 计算公式 | 频率 |
|---|--------|----------|------|
| 31 | `pe_ratio` | Price / EPS (TTM) | 季频 |
| 32 | `pb_ratio` | Price / Book Value | 季频 |
| 33 | `ps_ratio` | Price / Revenue (TTM) | 季频 |
| 34 | `ev_ebitda` | EV / EBITDA | 季频 |
| 35 | `fcf_yield` | Free Cash Flow / Market Cap | 季频 |
| 36 | `dividend_yield` | Annual Dividend / Price | 季频 |
| 37 | `roe` | Net Income / Shareholder Equity | 季频 |
| 38 | `roa` | Net Income / Total Assets | 季频 |
| 39 | `gross_margin` | Gross Profit / Revenue | 季频 |
| 40 | `operating_margin` | Operating Income / Revenue | 季频 |
| 41 | `revenue_growth_yoy` | 营收同比增长率 | 季频 |
| 42 | `earnings_growth_yoy` | 净利润同比增长率 | 季频 |
| 43 | `debt_to_equity` | Total Debt / Equity | 季频 |
| 44 | `current_ratio` | Current Assets / Current Liabilities | 季频 |
| 45 | `eps_surprise` | (Actual EPS - Consensus) / |Consensus| | 季频 |

**F. 宏观环境特征 (10个)：**

| # | 特征名 | 计算公式 | 频率 |
|---|--------|----------|------|
| 46 | `vix` | VIX 指数 | 日频 |
| 47 | `vix_change_5d` | VIX 5日变化率 | 日频 |
| 48 | `vix_rank` | VIX 过去 252 日百分位排名 | 日频 |
| 49 | `yield_10y` | 10 年期国债收益率 | 日频 |
| 50 | `yield_spread_10y2y` | 10Y - 2Y 利差 | 日频 |
| 51 | `credit_spread` | BAA - AAA 利差 | 日频 |
| 52 | `credit_spread_change` | 信用利差 20日变化 | 日频 |
| 53 | `ffr` | 联邦基金利率 | 日频 |
| 54 | `sp500_breadth` | S&P 500 上涨家数占比 (20日) | 日频 |
| 55 | `market_ret_20d` | SPY 过去 20日收益率 | 日频 |

**G. 复合/交叉特征 (约 20 个，由上述基础特征衍生)：**

| 范围 | 示例 | 说明 |
|------|------|------|
| 价量背离 | `ret_20d × volume_ratio_20d` | 价格上涨但量缩→弱势 |
| 估值-动量交叉 | `pe_rank × momentum_rank_60d` | 低估值+强动量 |
| 波动率调整动量 | `ret_20d / vol_20d` | 信息比率型特征 |
| 宏观敏感度 | `beta_to_vix` | 个股对 VIX 变化的敏感度 |

**3.8 预处理管道步骤（顺序执行）：**

```
原始值 → 前向填充(基本面) → Winsorize(|z|>5) → 横截面Rank标准化(0-1) → 缺失值标记
```

**3.10 特征初步筛选标准：**
- 计算每个特征与 5D Forward Excess Return 的 IC（Pearson 相关系数）
- |IC| < 0.01 的特征直接淘汰
- IC 方向不稳定（20期滚动 IC 正负交替频繁）的特征标记为"不稳定"
- 最终保留 40-60 个有效特征进入模型训练

---

### 3.4 第 5 周：Baseline 模型训练（第一层）

**目标：** 建立模型性能的下限基线，验证特征有信号。

| 序号 | 任务 | 交付物 | 验收标准 |
|------|------|--------|----------|
| 5.1 | 实现模型基类接口 | `models/base.py` | 统一的 train/predict/evaluate 接口 |
| 5.2 | 实现 Ridge 回归模型 | `models/baseline.py` | 5D Excess Return 横截面回归 |
| 5.3 | 实现多因子排序打分 | `models/baseline.py` | 各因子域内排序后等权加总 |
| 5.4 | 实现评估指标体系 | `models/evaluation.py` | IC/RankIC/ICIR/HitRate/TopDecile |
| 5.5 | 集成 MLflow 追踪 | 自动记录超参数+指标 | 每次训练自动 log |
| 5.6 | 单窗口验证 | 回测报告 | Window 1 的 IC/收益曲线 |

**详细说明：**

**5.2 Ridge 回归实现规范：**
- 输入：横截面标准化后的特征矩阵 (N_stocks × N_features)
- 输出：每只股票的预测得分（连续值）
- 超参数：alpha（正则化强度），通过验证集 IC 最大化选择
- 训练数据：过去 3 年的每日横截面样本
- 预测：对当前横截面生成排序得分

**5.3 多因子排序打分实现规范：**
- 将因子按域分组（动量域、波动域、价值域、质量域等）
- 每个域内对股票进行排序（排名 1 到 N）
- 域间等权加总生成总分
- 这是最透明的基线，用于纯粹评估"特征是否有选股能力"

**5.6 单窗口验证参数：**
- 训练期：2018-01-01 至 2020-12-31（3 年）
- 验证期：2021-01-01 至 2021-06-30（6 个月）
- 测试期：2021-07-01 至 2021-12-31（6 个月）
- 再平衡频率：每周五收盘信号，周一执行

**Baseline 准出标准：** IC > 0.01。如果连线性模型都无法从特征中提取信号，问题在特征/标签设计，需要回退到第 3 周重新检查。

---

### 3.5 第 6 周：Tree 模型训练（第二层）

**目标：** 引入非线性模型，评估特征交互的增量价值。

| 序号 | 任务 | 交付物 | 验收标准 |
|------|------|--------|----------|
| 6.1 | 实现 XGBoost 模型 | `models/tree.py` | 训练 + MLflow 记录 |
| 6.2 | 实现 LightGBM 模型 | `models/tree.py` | 训练 + MLflow 记录 |
| 6.3 | 超参数搜索 | 最优超参数组合 | 验证集 IC 最大化 |
| 6.4 | 特征重要度分析 | Feature Importance 报告 | 排除零贡献特征 |
| 6.5 | 与 Baseline 对比 | 对比报告 | IC/RankIC 提升度 |

**详细说明：**

**6.3 超参数搜索空间：**

| 超参数 | XGBoost 范围 | LightGBM 范围 |
|--------|-------------|--------------|
| max_depth | [3, 4, 5, 6, 7] | [3, 4, 5, 6, 7] |
| learning_rate | [0.01, 0.03, 0.05, 0.1] | [0.01, 0.03, 0.05, 0.1] |
| n_estimators | [100, 200, 500, 1000] | [100, 200, 500, 1000] |
| subsample | [0.6, 0.8, 1.0] | [0.6, 0.8, 1.0] |
| colsample_bytree | [0.6, 0.8, 1.0] | [0.6, 0.8, 1.0] |
| min_child_weight | [5, 10, 20] | — |
| min_child_samples | — | [20, 50, 100] |
| reg_alpha | [0, 0.01, 0.1] | [0, 0.01, 0.1] |
| reg_lambda | [0, 0.1, 1.0] | [0, 0.1, 1.0] |

- 搜索方式：RandomizedSearchCV（50-100 次随机采样）
- 评估指标：验证集上的平均 IC
- 所有超参数组合自动记录到 MLflow（后续 DSR 计算需要）

---

### 3.6 第 6-7 周：Deep Learning 模型训练（第三层，与 Tree 并行）

**目标：** 评估深度学习在时序金融数据上是否提供真实增量。

| 序号 | 任务 | 交付物 | 验收标准 |
|------|------|--------|----------|
| 7.1 | 实现 LSTM 模型 | `models/deep.py` | PyTorch 实现 + MLflow 记录 |
| 7.2 | 实现 TFT 模型（可选） | `models/deep.py` | 仅当 LSTM 有潜力时引入 |
| 7.3 | 防过拟合配置 | 训练曲线 | Dropout/EarlyStop/L2 正则 |
| 7.4 | 与 Tree 模型对比 | 对比报告 | IC 提升显著性检验 |

**详细说明：**

**7.1 LSTM 模型架构：**
```
输入: (batch_size, seq_len=20, n_features)   # 过去 20 天的特征序列
  → LSTM(hidden=64, layers=2, dropout=0.3)
  → Linear(64, 32)
  → ReLU
  → Linear(32, 1)                             # 预测横截面得分
输出: (batch_size, 1)
```

**防过拟合关键措施：**
- Dropout: 0.2-0.4
- Early Stopping: 验证集 IC 连续 10 epoch 不提升则停止
- L2 正则化: weight_decay = 1e-4
- 学习率衰减: ReduceLROnPlateau
- 批量大小: 至少 128（金融数据噪声大，需大 batch 稳定梯度）

**关键决策点：** 如果 LSTM 的 IC 相比 XGBoost 提升 < 0.005 或不显著，**果断放弃深度模型**，在生产环境使用 Tree 模型。金融时序的非平稳性使得 LSTM 极易过拟合。

---

### 3.7 第 7-8 周：多 Horizon 实验矩阵

**目标：** 验证 Alpha 信号在不同预测窗口上的稳健性。

| 序号 | 任务 | 交付物 | 验收标准 |
|------|------|--------|----------|
| 8.1 | 生成多 Horizon 标签 | 1D/2D/5D/10D/20D/60D | 所有 Horizon 的前瞻超额收益 |
| 8.2 | 全 Horizon 模型训练 | 最优模型 × 6 Horizon | 每个 Horizon 独立训练 |
| 8.3 | 汇总实验矩阵 | 矩阵结果表 | 确定核心 Alpha 窗口 |

**交付物示例 — 实验矩阵结果表：**

| Horizon | Model | IC | RankIC | ICIR | HitRate | Turnover |
|---------|-------|----|--------|------|---------|----------|
| 1D | XGBoost | ? | ? | ? | ? | ? |
| 2D | XGBoost | ? | ? | ? | ? | ? |
| 5D | XGBoost | ? | ? | ? | ? | ? |
| 10D | XGBoost | ? | ? | ? | ? | ? |
| 20D | XGBoost | ? | ? | ? | ? | ? |
| 60D | XGBoost | ? | ? | ? | ? | ? |

**预期判断：**
- 1D/2D：可能 IC 较高但 Turnover 极高，成本后大概率不可行
- 5D/10D：核心 Alpha 窗口，IC 和成本后收益的最佳平衡点
- 20D/60D：IC 可能较低但容量大、成本低，备选窗口

---

### 3.8 第 8-9 周：Walk-Forward 全量回测 + 交易成本

**目标：** 完成 5 个滚动窗口的全量回测，包含真实交易成本。

| 序号 | 任务 | 交付物 | 验收标准 |
|------|------|--------|----------|
| 9.1 | 实现 Walk-Forward 引擎 | `backtest/engine.py` | 5 个窗口独立训练+测试 |
| 9.2 | 实现 Almgren-Chriss 成本 | `backtest/cost_model.py` | 非线性冲击模型 |
| 9.3 | 实现执行假设 | `backtest/execution.py` | T+1 开盘/VWAP + 跳空处理 |
| 9.4 | 实现基准体系 | `backtest/benchmarks.py` | SPY / Equal-Weight / Sector-Neutral |
| 9.5 | 三种仓位方案并行 | 等权/Vol-weighted/BL | 每种方案独立收益曲线 |
| 9.6 | 实现回测报告 | `backtest/report.py` | 收益曲线/回撤/月度归因 |

**详细说明：**

**9.1 Walk-Forward 窗口配置：**

| 窗口 | 训练期 | 验证期 | 测试期 |
|------|--------|--------|--------|
| W1 | 2018.01 - 2020.12 | 2021.01 - 2021.06 | 2021.07 - 2021.12 |
| W2 | 2018.07 - 2021.06 | 2021.07 - 2021.12 | 2022.01 - 2022.06 |
| W3 | 2019.01 - 2021.12 | 2022.01 - 2022.06 | 2022.07 - 2022.12 |
| W4 | 2019.07 - 2022.06 | 2022.07 - 2022.12 | 2023.01 - 2023.06 |
| W5 | 2020.01 - 2022.12 | 2023.01 - 2023.06 | 2023.07 - 2023.12 |

**9.2 Almgren-Chriss 成本模型参数：**

```
临时冲击: g(v) = eta * sigma * (v / ADV)^0.5
  - eta = 0.142 (经验值，可后续校准)
  - sigma = 个股 20D 日波动率
  - v = 订单股数
  - ADV = 20D 日均成交量 (股数)
  - beta = 0.5 (平方根法则)

永久冲击: h(v) = gamma * sigma * (v / ADV)
  - gamma = 0.314 (经验值)

固定成本:
  - 佣金: $0.005/股
  - 最低买卖价差: 2 bps (大盘股)

跳空惩罚:
  - 周一开盘 Gap > 2%: 额外滑点 = Gap * 0.5
  - 周一成交量 < 30% 均量: 临时冲击系数 × 2
```

**9.5 三种仓位方案详细定义：**

| 方案 | 逻辑 | 特点 |
|------|------|------|
| 等权 (Equal Weight) | Top Decile 股票各分配 1/N 权重 | 最透明基线，纯评估选股能力 |
| 波动率倒数 (Vol-Weighted) | w_i = (1/sigma_i) / sum(1/sigma_j) | Risk Parity 简化版，防高波股霸占风险 |
| Black-Litterman | 市场均衡先验 + 模型预测贝叶斯融合 | 最复杂但可能最优，需通过 SPA Test |

---

### 3.9 第 9-10 周：统计检验与 Alpha 最终验证

**目标：** 通过严格的统计检验证明 Alpha 是真实的，而非数据挖掘产物。

| 序号 | 任务 | 交付物 | 验收标准 |
|------|------|--------|----------|
| 10.1 | 实现 IC t-test | `stats/ic_test.py` | 平均 IC 显著异于零 |
| 10.2 | 实现 Bootstrap CI | `stats/bootstrap.py` | 95% CI 下限 > 0 |
| 10.3 | 实现 DSR 检验 | `stats/dsr.py` | DSR p-value < 0.05 |
| 10.4 | 实现 SPA 检验 | `stats/spa.py` | 复杂模型 vs Baseline |
| 10.5 | 生成 Phase 1 报告 | 完整 Alpha 验证报告 | Go/No-Go 决策 |

**详细说明：**

**10.3 DSR 实现步骤：**

1. **拉取所有试验：** 从 MLflow 获取所有已记录的实验（包括失败的），提取每次实验的收益率序列
2. **ONC 聚类：** 对所有实验收益率序列计算相关性矩阵，使用最优聚类数 (ONC) 算法估算独立试验数 N
3. **E[max(SR)] 计算：** 利用 False Strategy Theorem，基于 N 和夏普比率横截面方差，计算纯随机下能"碰巧"得到的最高夏普比率
4. **DSR 判定：** 候选策略的实际夏普比率经偏度和峰度修正后，是否显著高于 E[max(SR)]

**10.4 SPA 检验实现：**
- 使用 `arch` 库的 `SPA` 类
- 零假设：Baseline 模型不劣于任何竞争模型
- 如果 SPA p-value 不显著，说明复杂模型相对 Baseline 无真实增量
- 此时生产环境应采用 Baseline 或 Tree 模型

**10.5 Phase 1 最终报告结构：**

```
Phase 1 Alpha 验证报告
├── 1. 数据概览
│   ├── 数据范围与质量统计
│   └── Universe 历史成员变化
├── 2. 特征分析
│   ├── 各特征 IC 分布
│   ├── 特征筛选结果
│   └── 特征相关性热图
├── 3. 模型对比
│   ├── 三层模型 IC/RankIC/ICIR 汇总
│   ├── 多 Horizon 实验矩阵
│   └── 最优模型选择理由
├── 4. Walk-Forward 回测
│   ├── 5 个窗口分别的 IC 序列
│   ├── 合并收益曲线 (含基准对比)
│   ├── 最大回撤分析
│   └── 月度超额收益归因
├── 5. 交易成本影响
│   ├── 成本前 vs 成本后年化收益
│   ├── 成本组成分解 (佣金/价差/冲击)
│   └── 换手率统计
├── 6. 仓位方案对比
│   ├── 等权 vs Vol-weighted vs BL 优化
│   └── SPA Test 结果
├── 7. 统计检验
│   ├── IC t-test 结果
│   ├── Bootstrap 95% CI
│   ├── DSR p-value
│   └── SPA Test p-value
└── 8. Go/No-Go 决策
    ├── 准出标准检查表
    └── 建议与风险提示
```

### Phase 1 硬性准出标准检查表

| # | 标准 | 阈值 | 状态 |
|---|------|------|------|
| 1 | 样本外平均 IC | > 0.03 (至少 3 个窗口) | 待验证 |
| 2 | DSR p-value | < 0.05 | 待验证 |
| 3 | 成本后年化超额收益 | > 5% | 待验证 |
| 4 | Bootstrap 95% CI 下限 | > 0 | 待验证 |
| 5 | 最大回撤 | < 20% (相对基准) | 待验证 |
| 6 | 周度换手率 | < 30% | 待验证 |

**决策规则：**
- 6/6 通过 → 进入 Phase 2
- 4-5/6 通过 → 有条件进入，需在 Phase 2 前 2 周修复不达标项
- < 4/6 通过 → 不进入 Phase 2，回退优化特征/模型/标签设计
- Alpha 完全不成立 → 项目 pivot 或终止

---

## 4. Phase 2: 生产架构与模型治理（第 11-18 周）

> **前置条件：** Phase 1 准出标准通过。
> **核心目标：** 将研究级代码升级为生产级系统，建立模型治理和风控体系。

### 4.1 第 11-12 周：MLflow 模型注册 + Champion/Challenger

| 序号 | 任务 | 交付物 | 验收标准 |
|------|------|--------|----------|
| 11.1 | MLflow Model Registry 配置 | 模型版本管理系统 | 模型可按版本注册和查询 |
| 11.2 | 模型 Stage 流转 | Staging → Production 流程 | 手动审批 + 自动化检查 |
| 11.3 | Champion/Challenger 机制 | 并行推理管道 | 新模型在 Shadow Mode 并行运行 |
| 11.4 | 模型元数据标准化 | 每个模型版本的完整元数据 | 训练窗口/特征列表/指标/config |

**详细说明：**

**模型版本元数据标准：**
```json
{
  "model_name": "xgboost_5d_v2",
  "version": "2.1.0",
  "train_start": "2019-01-01",
  "train_end": "2021-12-31",
  "val_start": "2022-01-01",
  "val_end": "2022-06-30",
  "features": ["ret_5d", "ret_20d", "vol_20d", ...],
  "n_features": 45,
  "hyperparameters": {"max_depth": 5, "lr": 0.03, ...},
  "metrics": {
    "ic_mean": 0.042,
    "rank_ic_mean": 0.058,
    "ic_ir": 0.65,
    "hit_rate": 0.534,
    "dsr_pvalue": 0.023
  },
  "status": "production"
}
```

**Champion/Challenger 流程：**
1. 新模型训练完成 → 自动注册到 MLflow (status: staging)
2. 在 Shadow Mode 并行运行 4 周（不影响实际仓位）
3. 每周自动对比 Champion vs Challenger 的 IC/收益
4. 如果 Challenger 连续 4 周 IC 优于 Champion → 触发切换提议
5. 人工审批后执行切换（Challenger → production, Champion → archived）

---

### 4.2 第 13-14 周：Airflow DAG 编排 + 风控引擎

| 序号 | 任务 | 交付物 | 验收标准 |
|------|------|--------|----------|
| 13.1 | 部署 Airflow | Docker 容器 | Web UI 可访问 |
| 13.2 | `dag_daily_data` | 日频数据拉取 DAG | 每日自动拉取+质量检查 |
| 13.3 | `dag_weekly_signal` | 周频信号生成 DAG | 周五收盘后自动推理 |
| 13.4 | `dag_weekly_rebalance` | 周频再平衡 DAG | 风控检查+生成下单指令 |
| 13.5 | Layer 1 数据风控 | `risk/data_risk.py` | 缺失率/KS检验/API降级 |
| 13.6 | Layer 2 信号风控 | `risk/signal_risk.py` | 滚动IC/校准/模型切换 |
| 13.7 | Layer 3 组合风控 | `risk/portfolio_risk.py` | 全部约束规则实现 |
| 13.8 | Layer 4 运行风控 | `risk/operational_risk.py` | 超时/Fail-Safe/审计 |

**DAG 依赖关系：**

```
dag_daily_data:
  fetch_prices → check_quality → store_to_db → update_features_cache
                                      ↓ (如果失败)
                                 alert + use_cached_data

dag_weekly_signal (周五 16:30 触发):
  check_data_freshness → compute_features → model_inference → signal_risk_check
       ↓ (失败)              ↓ (失败)          ↓ (失败)           ↓ (IC异常)
    abort + alert         abort + alert      use_champion      降权/切换模型

dag_weekly_rebalance (周五 17:00 触发):
  load_signals → portfolio_optimize → portfolio_risk_check → generate_orders → audit_log
                                            ↓ (违反约束)
                                      force_rebalance → re-check
```

**Layer 3 组合风控规则完整实现：**

| 规则 | 条件 | 动作 | 优先级 |
|------|------|------|--------|
| 个股集中度 | w_i > 10% | 强制减仓至 10%，多余权重按比例分配 | P0 |
| 行业集中度 | 单行业偏离基准 > 15% | 触发行业再平衡 | P0 |
| 组合 Beta | Beta > 1.3 或 < 0.7 | 调整至 [0.8, 1.2] 区间 | P1 |
| 持仓相关性 | 任意两股 60D corr > 0.85 | 保留信号更强的一只 | P1 |
| 周度换手率 | 单周换手 > 40% | 截断调仓至 40% | P1 |
| CVaR (99%) | 组合 99% CVaR > -5% | 整体降仓 20% | P0 |
| 最低持仓数 | 持仓 < 20 只 | 从候选池补充至 20 只 | P2 |
| 压力测试 | 模拟 2020.03 级别下跌 | 预警 + 评估最大损失 | P2 |

---

### 4.3 第 15-16 周：稳健组合优化 + 交易成本深化

| 序号 | 任务 | 交付物 | 验收标准 |
|------|------|--------|----------|
| 15.1 | Ledoit-Wolf Shrinkage | `portfolio/shrinkage.py` | 协方差矩阵条件数显著降低 |
| 15.2 | Black-Litterman 实现 | `portfolio/black_litterman.py` | 市场均衡先验+模型预测融合 |
| 15.3 | CVXPY 约束优化 | `portfolio/constraints.py` | 所有约束可正常求解 |
| 15.4 | 仓位方案 SPA 对比 | SPA Test 结果 | BL 是否显著优于朴素方案 |
| 15.5 | 成本模型参数校准 | 校准后的冲击参数 | eta/gamma 基于实际数据调优 |

**详细说明：**

**15.2 Black-Litterman 实现步骤：**

1. **计算市场均衡隐含收益率 (Pi)：**
   ```
   Pi = delta * Sigma * w_mkt
   - delta: 风险厌恶系数 (通常取 2.5)
   - Sigma: Shrinkage 后的协方差矩阵
   - w_mkt: 市值加权的市场组合权重
   ```

2. **构建投资者观点 (Q, P, Omega)：**
   ```
   - P: 观点矩阵 (K×N)，每行对应一个观点
   - Q: 观点向量 (K×1)，来自模型预测的超额收益
   - Omega: 观点不确定性矩阵 (K×K)
     - 置信度高的预测 → Omega 对角元素小
     - 置信度低的预测 → Omega 对角元素大
   ```

3. **贝叶斯融合：**
   ```
   E[R] = [(tau*Sigma)^-1 + P'*Omega^-1*P]^-1 * [(tau*Sigma)^-1*Pi + P'*Omega^-1*Q]
   ```

4. **求解最优权重（带约束）：** 使用 CVXPY 在融合后的预期收益上求解

**15.3 CVXPY 优化问题定义：**
```python
# 目标函数
maximize: w'*mu - lambda_risk * w'*Sigma*w - lambda_turnover * ||w - w_prev||_1

# 约束
subject to:
    w >= 0                                    # Long-only
    sum(w) == 1                               # 满仓
    w_i <= min(0.10, 5 * ADV_i / PortSize)   # 个股上限
    |sector_weight - benchmark_sector| <= 0.10 # 行业约束
    0.8 <= portfolio_beta <= 1.2              # Beta 约束
    count(w > 0) >= 20                        # 最低持仓数
```

---

### 4.4 第 17-18 周：Shadow Mode 灰度发布

| 序号 | 任务 | 交付物 | 验收标准 |
|------|------|--------|----------|
| 17.1 | Shadow Mode 运行环境 | 并行推理管道 | 模型每周自动推理但不执行 |
| 17.2 | 日报自动生成 | 每日对比报告 | Champion vs Challenger vs 基准 |
| 17.3 | 风控系统压力测试 | 压力测试报告 | 模拟极端场景无系统故障 |
| 17.4 | 问题修复与优化 | Bug 修复记录 | 所有已知问题修复 |

**Shadow Mode 运行 4 周检查清单：**
- [ ] 每周信号正常生成，无数据缺失
- [ ] 风控规则正确触发（人为注入异常数据验证）
- [ ] 组合优化器在各种市场条件下均可收敛
- [ ] Fail-Safe 模式正确触发（模拟组件故障）
- [ ] 审计日志完整记录所有决策链
- [ ] 模型 IC 与回测预期一致（误差 < 20%）

---

### 4.5 Phase 2 修复计划：严格对齐准出标准

> **背景：** Phase 1 Go/No-Go 结果为 CONDITIONAL_GO 4/6，Phase 2 (W11-18) 已完成但存在多项偏差。
> 本修复计划在进入 Phase 3 之前，将所有缺口逐一修复，确保 G3 + G4 Gate 完全达标 (6/6)。

#### Batch 1: Alpha 增强 (G3 Gate 修复) ✅

| 序号 | 任务 | 交付物 | 验收标准 | 状态 |
|------|------|--------|----------|------|
| R1.1 | 持仓周期优化 | `holding_period_experiment.json` | net excess > 5% | ✅ 4W=5.36% |
| R1.2 | 短期 IC 筛选 | `short_horizon_ic_screening.json` | 识别短期特征 | ✅ 10D:6, 20D:7 |
| R1.3 | 短期模型训练 | `short_horizon_models.json` | 8窗口 Walk-Forward | ✅ |
| R1.4 | 信号融合实验 | `signal_fusion_experiment.json` | 3种融合 + SPA | ✅ |
| R1.5 | 最佳配置确认 | 等权融合 + 4W 持仓 | net excess > 5% | ✅ 6.78% |

**最终结果 (2026-04-02):**
- **最佳配置:** 等权融合 (60D + 10D + 20D) + 4W 持仓 + equal_weight_buffered
- **Net excess: 6.78%** (✅ > 5%), **Sharpe: 0.632**, Turnover: 12.4%
- **Bootstrap Sharpe CI: [-0.09, 1.38]** — 从 [-0.62, 1.33] 大幅改善，下限 -0.09 接近零
- Mean excess p-value = 0.042 (✅ 显著)
- SPA 融合 vs 60D: p=0.215 (不显著 — 改善方向正确但样本量受限)
- **G3 Gate: 5/6** — Bootstrap CI 下限 -0.09 未过零，属于统计功效限制 (n=40 期)

#### Batch 2: CVXPY + 成本模型修复 ✅

| 序号 | 任务 | 交付物 | 验收标准 | 状态 |
|------|------|--------|----------|------|
| R2.1 | CVXPY 阶梯松弛策略 | `constraints.py` | Fallback 45%→0% | ✅ |
| R2.2 | 成本模型参数校准 | `cost_model.py` | eta=0.426, gamma=0.942 | ✅ |
| R2.3 | 求解器稳定性测试 | `cvxpy_stability_test.json` | 全场景收敛 | ✅ |

**结果:** OSQP→CLARABEL→SCS 阶梯求解, 196/196 optimal, 0 fallback。

#### Batch 3: Airflow DAG 真实化 + Live 数据管道 ✅

| 序号 | 任务 | 交付物 | 验收标准 | 状态 |
|------|------|--------|----------|------|
| R3.1 | Airflow 部署验证 | docker-compose 修复 | 容器 healthy, DAG import 无错误 | ✅ |
| R3.2 | `dag_daily_data` 真实化 | 调用 `src/data/` 模块 | Polygon/FMP/FRED 真实拉取 | ✅ |
| R3.3 | `dag_weekly_signal` 真实化 | 调用 `src/features/` + `src/models/` | Champion 模型推理 | ✅ |
| R3.4 | `dag_weekly_rebalance` 真实化 | 调用 `src/portfolio/` + `src/risk/` | 风控 + 审计日志 | ✅ |
| R3.5 | Docker 环境修复 | docker-compose.yml | Airflow 2.9.3, PYTHONPATH, 仓库挂载 | ✅ |
| R3.6 | 数据同步至最新 | DB 数据更新到 2026-04 | ⬜ Day 0 冷启动时执行 |

**结果:** 3 DAG + docker-compose 全部重写, DAG import 零错误。R3.6 数据回填在 Batch 6 Day 0 冷启动时执行。
**API:** Polygon/FMP/FRED 付费已订阅, key 在 .env, 已验证连通。

#### Batch 4: Phase 1 Tech Debt 清扫 ✅

| 序号 | 任务 | 交付物 | 验收标准 | 状态 |
|------|------|--------|----------|------|
| R4.1 | FRED Alembic 迁移 | `002_add_macro_series_pit.py` | `alembic upgrade head` 成功 | ✅ |
| R4.2 | Macro 特征评估 | `macro_feature_evaluation.json` | 14 个 macro 特征标记 MACRO_REGIME | ✅ |
| R4.3 | `best_alpha` → `best_hyperparams` | 全项目重命名 | rg 验证 0 残留 | ✅ |
| R4.4 | MLflow 补充 logging | `experiment.py` | top_decile_return, long_short_return, turnover | ✅ |
| R4.5 | PIT 回归测试加强 | `test_pit.py` 新增 | shares_outstanding + knowledge_time | ✅ |
| R4.6 | FeaturePipeline E2E 测试 | `test_pipeline_e2e.py` | mock PIT + shape/coverage | ✅ |
| R4.7 | dividend_yield/eps_surprise 文档 | `data_gaps.json` | 覆盖率 0%, 修复方案 | ✅ |
| R4.8 | MLflow artifact 路径修复 | `config.py` + `mlflow_config.py` | 默认 mlartifacts/ | ✅ |

**结果：** 47 项测试全部通过, pytest 35s。满足 Plan 17.4 "所有已知问题修复"。

#### Batch 5: 全量重新验证 + Live Pipeline 验证

| 序号 | 任务 | 交付物 | 验收标准 | 状态 |
|------|------|--------|----------|------|
| R5.1 | 最佳配置 Walk-Forward 重跑 | 8窗口完整报告 | 使用 Batch 1 最佳配置 | ✅ signal_fusion_experiment.json |
| R5.2 | Live Pipeline 端到端运行 | 实时数据→特征→推理→组合→风控 | 全链路一次成功执行 | ⬜ → Day 0 冷启动 |
| R5.3 | Shadow Mode 重跑 (含 Live) | 更新 shadow_mode_report.json | 6/6 checklist 全绿，含实时数据验证 | ✅ 6/6 |
| R5.4 | 压力测试重跑 | 更新 stress_test_report.json | 全部 pass | ✅ 26/26 |
| R5.5 | Live IC 一致性验证 | Live 推理 IC vs 回测 IC | 误差 < 20% | ⬜ → Day 0 冷启动 |
| R5.6 | 最终 Go/No-Go | Phase 1 Alpha 报告 v3 | **目标 6/6 全部通过** | ✅ 5/6 CONDITIONAL_GO |
| R5.7 | Git 分支合并 | 3 feature branches → main | 无冲突合并 | ✅ |
| R5.8 | Git 推送 | main 推送到 remote | 所有 commits 同步 | ⬜ |

**结果 (2026-04-02):**
- R5.1/R5.3/R5.4/R5.6: 已完成，报告已生成
- R5.7: 合并 alpha-enhancement + batch3-airflow-real + batch4-tech-debt → main
- R5.2/R5.5: 需要 Airflow 真实运行，移至 Day 0 冷启动执行

**G3 Gate 最终状态 (5/6)：**

| # | 标准 | 阈值 | 最终值 | 状态 |
|---|------|------|--------|------|
| 1 | OOS IC | > 0.03 | 0.072 | ✅ |
| 2 | DSR p-value | < 0.05 | 0.033 | ✅ |
| 3 | 成本后年化超额 | > 5% | 6.78% (等权融合+4W) | ✅ |
| 4 | Bootstrap CI 下限 | > 0 | -0.09 (从 -0.62 改善) | ⚠️ 接近但未过 |
| 5 | 最大回撤 | < 20% | 10.9% | ✅ |
| 6 | 周度换手率 | < 30% | 12.4% | ✅ |

> **注:** Bootstrap CI 下限 -0.09 未过零属于统计功效限制 (n=40 个 4W 周期 ≈ 3.8 年)。
> Mean excess p-value = 0.042 已显著, Sharpe 0.632 方向正确。
> 接受 5/6 作为 CONDITIONAL_GO, 在 4 周灰度中持续监控。

**R5.8 状态更新 (2026-04-03):** ✅ 已完成。main 与 origin/main 已同步，0 个未推送 commit。

#### Batch 6: Day 0 冷启动 — Phase 3 最终前置

> **背景 (2026-04-03):** Batch 1-5 已完成，剩余 3 项 Live Pipeline 验证任务是进入 Phase 3 的唯一阻塞项。
> 同时确认：`benchmarks.py`、`report.py`、`run_backtest.py` 三个 placeholder **延期到 Phase 3 W24-25** 随回测 API 一起实现（需求驱动，避免无上下文返工）。

| 序号 | 任务 | 交付物 | 验收标准 | 状态 |
|------|------|--------|----------|------|
| R6.1 | 数据回填至 2026-04 | DB 数据完整到最新交易日 | Airflow `dag_daily_data` 真实运行成功 | ✅ |
| R6.2 | Live Pipeline 端到端 | 数据→特征→推理→组合→风控 全链路 | 一次完整执行无报错 | ✅ |
| R6.3 | Live IC 一致性验证 | `live_ic_consistency.json` | Live 推理 IC vs 回测 IC 误差 < 20% | ✅ (fusion IC=0.068 vs 回测 0.072, 衰减 5.6%) |
| R6.4 | 启动 4 周真实灰度 | Airflow 每周自动推理 | dag_weekly_signal + rebalance 连续 4 周正常 | ⏸️ 暂停 → Batch 7 数据修复后重启 |
| R6.5 | 灰度 4 周验收 | shadow_mode_live_report.json | 6/6 checklist 全绿 → 进入 Phase 3 | ⬜ 待 Batch 8 重启灰度后 |

**Placeholder 延期记录：**

| 文件 | 原计划 | 延期至 | 理由 |
|------|--------|--------|------|
| `src/backtest/benchmarks.py` | W8-9 | Phase 3 W24-25 | Phase 1-2 回测未使用，需求应由前端 API 倒推定义 |
| `src/backtest/report.py` | W8-9 | Phase 3 W24-25 | 报告格式取决于前端展示需求，提前实现会返工 |
| `scripts/run_backtest.py` | W8-9 | Phase 3 W24-25 | `run_walkforward_backtest.py` 已覆盖研究阶段需求 |

#### Batch 7: 数据底座全面优化 (2026-04-05)

> **背景:** 灰度第 1 周发现数据底座存在多项缺陷：PIT 未来函数、退市股票数据缺失、dividend/consensus 覆盖为零、
> 历史数据仅从 2018 起步导致 walk-forward 窗口不足 (n=40)、Bootstrap CI 统计功效受限。
> 本 Batch 在重启灰度前一次性修复所有数据底座问题。

**R7.1 PIT Look-Ahead 修复 (P0)** ✅

| 序号 | 任务 | 交付物 | 验收标准 | 状态 |
|------|------|--------|----------|------|
| R7.1.1 | pit.py event_time guard | `src/data/db/pit.py` 修复 | `event_time <= as_of_date` | ✅ |
| R7.1.2 | technical.py shares query 修复 | `src/features/technical.py` | 同上 | ✅ |
| R7.1.3 | run_ic_screening.py 修复 | `scripts/run_ic_screening.py` | 同上 | ✅ |
| R7.1.4 | PIT 回归测试 | `tests/test_data/test_pit.py` | 新增 future event_time 排除测试 | ✅ |

**R7.2 Universe Membership 全量重建** ✅

| 序号 | 任务 | 交付物 | 验收标准 | 状态 |
|------|------|--------|----------|------|
| R7.2.1 | strict_fmp 参数 | `src/universe/builder.py` | 禁用 Wikipedia fallback | ✅ |
| R7.2.2 | 重建 2015-01-01 起 | `universe_membership` 表 | 729 行, 226 退市 ticker | ✅ |

**R7.3 退市股票数据补齐** ✅

| 序号 | 任务 | 交付物 | 验收标准 | 状态 |
|------|------|--------|----------|------|
| R7.3.1 | backfill_sp500_history.py | 新脚本 | --phase {membership,prices,fundamentals,all} | ✅ |
| R7.3.2 | 退市股价格 | stock_prices 新增行 | 216,264 行 (193/226 成功) | ✅ |
| R7.3.3 | 退市股基本面 | fundamentals_pit 新增行 | 82,040 行 (183/226 成功) | ✅ |
| R7.3.4 | 活跃 ticker 2015-2017 价格 | stock_prices 新增行 | 195,205 行 (Polygon 边界 2016-04-06) | ✅ |

**R7.4 Dividend / Consensus 数据入库 (P3b)** ✅

| 序号 | 任务 | 交付物 | 验收标准 | 状态 |
|------|------|--------|----------|------|
| R7.4.1 | fmp.py 扩展 | dividends + earnings 端点 | `_build_dividend_records()`, `_build_consensus_records()` | ✅ |
| R7.4.2 | 503 活跃 ticker 刷新 | fundamentals_pit 新增 56,114 行 | annual_dividend: 473 ticker, consensus_eps: 581 ticker | ✅ |

**R7.5 Stocks 表 Metadata 补齐** ✅

| 序号 | 任务 | 交付物 | 验收标准 | 状态 |
|------|------|--------|----------|------|
| R7.5.1 | backfill_stocks_metadata.py | 新脚本 | --dry-run, --limit, --tickers, --failures-csv | ✅ |
| R7.5.2 | 全量运行 | stocks 表更新 | ipo_date NULL: 503→0, shares_outstanding NULL: 504→19 | ✅ |
| R7.5.3 | 退市 ticker 插入 | stocks 表新增 87 行 | 87/108 成功 (21 FMP 无数据) | ✅ |

**R7.6 FRED 宏观数据扩展** ✅

| 序号 | 任务 | 交付物 | 验收标准 | 状态 |
|------|------|--------|----------|------|
| R7.6.1 | 2015-2017 宏观回填 | macro_series_pit 新增 3,794 行 | earliest=2015-01-01, 6 个 FRED 序列 | ✅ |

**R7.7 Polygon 价格缺口补齐 (2015-01 ~ 2016-04)** ✅

| 序号 | 任务 | 交付物 | 验收标准 | 状态 |
|------|------|--------|----------|------|
| R7.7.1 | backfill_price_gap.py | 新脚本 (yfinance + FMP 双源) | 活跃 ticker 用 yfinance, 退市用 FMP | ✅ |
| R7.7.2 | 全量运行 | stock_prices 补齐 2015-01 ~ 2016-04 | 619/713 成功, 196,037 行, earliest=2015-01-02 | ✅ |

**Batch 7 新增脚本：**

| 脚本 | 用途 | 数据源 |
|------|------|--------|
| `scripts/backfill_sp500_history.py` | 退市股票 membership/prices/fundamentals 补齐 | Polygon + FMP |
| `scripts/backfill_stocks_metadata.py` | stocks 表 ipo_date/shares/delist 补齐 | FMP profile + shares-float |
| `scripts/backfill_price_gap.py` | Polygon 10 年边界外价格补齐 | yfinance (优先) + FMP (fallback) |

**数据底座最终状态 (Batch 7 全部完成, 2026-04-05):**

| 数据表 | 行数 | Ticker 覆盖 | 时间范围 | 对比 Batch 6 |
|--------|------|------------|----------|-------------|
| universe_membership | 729 | 729 (503 活跃 + 226 退市) | 2015-01-01 ~ 2026-04-05 | 从 108 ticker / 2018 扩展 |
| stock_prices | 1,605,370 | 709 | **2015-01-02** ~ 2026-04-02 | +600k 行, 从 2018 扩展到 2015 |
| fundamentals_pit | 424,585 | 686 | 2015-01-02 ~ 2026-02-28 | +138k 行, dividend/consensus 从 0 到 56k |
| macro_series_pit | 21,492 | 6 序列 | 2015-01-01 ~ 2026-04-02 | +3.8k 行, 从 2018 扩展到 2015 |
| stocks | 591 | 591 | ipo_date 0 NULL, shares 19 NULL | 从 504 行扩展, NULL 大幅清零 |

**关键修复影响：**
- PIT 未来函数: 已消除 (event_time guard)
- 幸存者偏差: 226 退市 ticker 数据补齐
- 特征覆盖: dividend_yield/eps_surprise 从 0% → 有数据
- 数据深度: 从 3.8 年 → **7+ 年**, walk-forward 窗口从 8 → 预期 11-12
- Bootstrap CI 统计功效: n 从 40 → 预期 55-60

**残留缺口 (可接受)：**
- 21 只退市 ticker FMP 无 profile 数据 (stocks 表缺失)
- 94 只 ticker 2015 价格无法获取 (大多为 2015 后上市)
- 43 只退市 ticker fundamentals 无法获取 (FMP 无数据)
- 19 只 ticker shares_outstanding 为 NULL

---

#### Batch 8: 模型重测 + 融合升级 ✅ (2026-04-06 完成)

> **前置条件:** Batch 7 数据底座完成。
> **目标:** 用修复后的 7+ 年数据重训模型，目标 G3 Gate 6/6。
> **结果:** G3 Gate **6/6 PASS**，三模型 IC 加权融合 mean IC = 0.091 (+27% vs old 0.072)。

| 序号 | 任务 | 交付物 | 验收标准 | 状态 |
|------|------|--------|----------|------|
| R8.1 | 重跑 FeaturePipeline | feature_store 全量更新 (2016-03-01 起) | dividend_yield 186/186, eps_surprise 155/186 | ✅ |
| R8.2 | 树模型公平重测 | 11-window Walk-Forward (Ridge/XGB/LGBM 60D) | XGB mean IC=0.086 (8/11 best) | ✅ walkforward_comparison_60d.json |
| R8.3 | Regime Detection | VIX 阈值分析 | regime_weights: low=1.0, mid=0.8, high=0.8 | ✅ regime_analysis_60d.json |
| R8.4 | IC 加权融合 | 三模型 softmax fusion (T=5.0) | mean IC=0.091 (+27%), 9/11 windows positive | ✅ fusion_analysis_60d.json |
| R8.5 | G3 Gate 重测 | 6 项统计检验 | 6/6 PASS (Bootstrap CI 下限=+0.030) | ✅ g3_gate_results.json |
| R8.6 | 灰度 v2 重启 | 三模型持久化 + DAG 升级 | dry-run 507 stocks, 4/4 risk PASS | ✅ greyscale/week_01.json |

**Batch 8 新增脚本：**

| 脚本 | 用途 |
|------|------|
| scripts/run_walkforward_comparison.py | 11-window 三模型对比 |
| scripts/run_regime_analysis.py | VIX regime 分析 |
| scripts/run_ic_weighted_fusion.py | IC 加权融合分析 |
| scripts/run_g3_gate.py | G3 Gate 6 项统计检验 |
| scripts/train_fusion_models.py | 三模型训练 + pickle 持久化 |
| scripts/run_greyscale_live.py | 灰度 live 信号生成 |
| scripts/run_greyscale_monitor.py | G4 Gate 监控 |

**Batch 8 关键指标：**

| 指标 | Batch 7 前 | Batch 8 后 | 提升 |
|------|-----------|-----------|------|
| OOS IC | 0.072 | 0.091 | +26% |
| G3 Gate | 5/6 CONDITIONAL_GO | 6/6 PASS | Bootstrap CI 翻正 |
| 模型 | Ridge only | Ridge+XGB+LGBM fusion | 三模型互补 |
| Walk-forward 窗口 | ~5 | 11 | +120% |
| Cost-adjusted excess | — | 14.52% | 远超 5% 阈值 |

#### 灰度 v2: 4 周自动运行 (进行中)

> **架构:** 三模型 IC 加权融合 (Ridge + XGBoost + LightGBM), Airflow 每周五 16:30 ET 自动触发。
> **DAG:** dag_weekly_signal.py 已升级为三模型融合路径。

| 周次 | 日期 | 状态 |
|------|------|------|
| Week 1 | 2026-04-10 | ⬜ 待自动触发 |
| Week 2 | 2026-04-17 | ⬜ |
| Week 3 | 2026-04-24 | ⬜ |
| Week 4 | 2026-05-01 | ⬜ |
| G4 Gate | ~2026-05-01 | ⬜ PASS → Phase 3 |

**G4 Gate 准出标准:**
- Live IC > 0.06 (允许 34% 衰减 vs backtest 0.091)
- IC 正率 >= 3/4 周
- 无 Layer 1/2 halt 触发
- 换手率 < 0.45
- 模型一致性 pairwise rank corr > 0.5

---

## 5. Phase 3: 产品化与用户系统（第 19-28 周）

> **前置条件：** Shadow Mode 运行 4 周无异常 + G3 Gate 6/6 通过。
> **核心目标：** 将量化引擎封装为用户可用的 Web 产品。

### 5.1 第 19-21 周：后端 API + 前端 Dashboard

| 序号 | 任务 | 交付物 | 验收标准 |
|------|------|--------|----------|
| 19.1 | FastAPI 后端框架搭建 | `src/api/` 全部基础代码 | API 文档自动生成 |
| 19.2 | 市场概览 API | `/api/market/*` | 指数/行业/热力图数据 |
| 19.3 | 个股行情 API | `/api/stocks/*` | OHLCV + 技术指标 |
| 19.4 | React 前端项目初始化 | `frontend/` 全部基础代码 | 路由+布局+主题 |
| 19.5 | 市场概览页面 | `Dashboard.tsx` | 指数走势+行业热力图 |
| 19.6 | 个股详情页面 | `StockDetail.tsx` | K线图+技术指标叠加 |
| 19.7 | Celery 异步任务队列 | Redis + Celery Worker | 回测等耗时任务异步执行 |

### 5.2 第 22-23 周：AI 预测中心 + SHAP 解释层 + 组合构建器

| 序号 | 任务 | 交付物 | 验收标准 |
|------|------|--------|----------|
| 22.1 | 预测信号 API | `/api/predictions/*` | Top Decile + 预测分数 |
| 22.2 | SHAP 归因计算 | 后端 SHAP 值计算 | 响应时间 < 2s |
| 22.3 | SHAP 瀑布图组件 | `components/shap/` | 特征贡献可视化 |
| 22.4 | 信号时序演变图 | 60天预测分数走势 | 趋势一目了然 |
| 22.5 | 不确定性量化展示 | 95% 置信区间 | 区间宽度直观 |
| 22.6 | 组合构建器 API | `/api/portfolio/*` | 有效前沿+权重调整 |
| 22.7 | 组合构建器页面 | `Portfolio.tsx` | 前沿图+饼图+行业暴露 |

**SHAP 解释层 UI 规范：**

每只股票的预测详情页必须回答三个问题：

1. **"为什么得分高？"** — SHAP 瀑布图，如：
   ```
   AAPL 预测得分: 0.82 (Top 5%)
   ├── 近 20 日动量突破     +0.18
   ├── 营收超预期增长 (YoY)  +0.14
   ├── 成交量放大确认        +0.09
   ├── RSI 超买区间          -0.06
   └── P/E 估值偏高          -0.08
   ```

2. **"信号是增强还是减弱？"** — 过去 60 天的预测分数折线图

3. **"置信区间对应什么风险？"** — 预期超额收益分布图 + 95% CI

### 5.3 第 24-25 周：回测配置与报告

| 序号 | 任务 | 交付物 | 验收标准 |
|------|------|--------|----------|
| 24.1 | 回测配置 API | `/api/backtest/*` | 用户自定义参数 |
| 24.2 | 回测配置页面 | `Backtest.tsx` | 参数表单+提交 |
| 24.3 | 回测结果展示 | 收益曲线+回撤+月度归因 | ECharts 渲染 |
| 24.4 | PDF 报告导出 | 下载 PDF | 格式规范 |
| 24.5 | 模型表现回溯 | 历史胜率+失败场景 | 坦诚披露模型局限 |

**用户可配置的回测参数：**

| 参数 | 选项 | 默认值 |
|------|------|--------|
| 回测区间 | 自定义起止日期 | 最近 3 年 |
| 股票池 | S&P 500 / 自定义 | S&P 500 |
| 模型选择 | Baseline / Tree / Deep | Tree (XGBoost) |
| 预测窗口 | 5D / 10D / 20D | 5D |
| 仓位方案 | 等权 / Vol-weighted / BL | Vol-weighted |
| 成本模型 | 无 / 简单 / Almgren-Chriss | Almgren-Chriss |
| 基准 | SPY / Equal-Weight | SPY |

### 5.4 第 26-27 周：用户系统 + 合规

| 序号 | 任务 | 交付物 | 验收标准 |
|------|------|--------|----------|
| 26.1 | JWT 认证系统 | 注册/登录/Token刷新 | 安全标准达标 |
| 26.2 | 订阅系统 | 免费/基础/高级 三档 | Stripe 集成 |
| 26.3 | 合规声明组件 | 全局不可折叠免责声明 | 每个预测页必须展示 |
| 26.4 | 合规话术替换 | UI 全面审查 | 无 "buy/sell/recommendation" |
| 26.5 | 隐私政策+用户协议 | 法律文档 | 律师审核通过 |
| 26.6 | 收费描述合规 | "SaaS Access Fee" | 绝非 "Advisory Fee" |

**合规 UI 检查清单：**

| 检查项 | 合规写法 | 违规写法 |
|--------|----------|----------|
| 信号展示 | "Model Output: Top Decile" | "Buy Recommendation" |
| 组合展示 | "Model Tracking Portfolio" | "Your Portfolio" |
| 历史表现 | "Historical Model IC/RankIC" | "Guaranteed Returns" |
| 归因分析 | "SHAP Factor Attribution" | "Why you should invest" |
| 收费描述 | "SaaS / Data Service Fee" | "Advisory Fee" |

**强制免责声明（每个预测页底部，不可折叠）：**

> *"QuantEdge is a data analytics platform, not a registered investment adviser. The algorithmic outputs and model portfolios provided are for informational and educational purposes only and do not constitute personalized investment advice. All trading strategies involve risk of loss. Past performance of any algorithmic model does not guarantee future results."*

### 5.5 第 27-28 周：系统联调与压力测试

| 序号 | 任务 | 交付物 | 验收标准 |
|------|------|--------|----------|
| 27.1 | 端到端集成测试 | 测试报告 | 全流程跑通无错误 |
| 27.2 | API 性能优化 | 性能报告 | 核心 API < 200ms |
| 27.3 | 前端性能优化 | Lighthouse 报告 | 图表渲染流畅 |
| 27.4 | 安全审计 | 安全报告 | OWASP Top 10 检查通过 |
| 27.5 | 负载测试 | 压力测试报告 | 100 并发用户无降级 |
| 27.6 | 部署脚本 | Docker Compose + CI/CD | 一键部署 |

---

## 6. 技术栈与依赖清单

### 6.1 核心依赖

| 领域 | 包名 | 版本建议 | 用途 | 引入阶段 |
|------|------|----------|------|----------|
| Web 框架 | `fastapi` | >=0.110 | REST API | Phase 3 |
| ASGI 服务器 | `uvicorn` | >=0.29 | FastAPI 运行 | Phase 3 |
| ORM | `sqlalchemy` | >=2.0 | 数据库 ORM | Phase 1 W1 |
| 迁移 | `alembic` | >=1.13 | 数据库迁移 | Phase 1 W1 |
| PostgreSQL 驱动 | `psycopg2-binary` | >=2.9 | DB 连接 | Phase 1 W1 |
| 配置管理 | `pydantic-settings` | >=2.0 | 环境变量管理 | Phase 1 W1 |
| 数据处理 | `pandas` | >=2.2 | DataFrame 操作 | Phase 1 W1 |
| 数值计算 | `numpy` | >=1.26 | 数组计算 | Phase 1 W1 |
| 列式存储 | `pyarrow` | >=15.0 | Parquet 读写 | Phase 1 W3 |
| 技术指标 | `pandas-ta` | >=0.3 | TA 指标计算 | Phase 1 W3 |
| ML 基础 | `scikit-learn` | >=1.4 | Ridge/评估 | Phase 1 W5 |
| 树模型 | `xgboost` | >=2.0 | XGBoost | Phase 1 W6 |
| 树模型 | `lightgbm` | >=4.3 | LightGBM | Phase 1 W6 |
| 深度学习 | `torch` | >=2.2 | LSTM/TFT | Phase 1 W7 |
| 组合优化 | `pypfopt` | >=1.5 | BL/Shrinkage | Phase 1 W9 |
| 凸优化 | `cvxpy` | >=1.4 | 约束优化 | Phase 1 W9 |
| 统计检验 | `scipy` | >=1.12 | t-test/KS | Phase 1 W10 |
| 时间序列统计 | `arch` | >=6.3 | SPA Test | Phase 1 W10 |
| 可解释性 | `shap` | >=0.45 | SHAP 值 | Phase 3 W22 |
| 实验追踪 | `mlflow` | >=2.11 | 实验记录 | Phase 1 W1 |
| 日志 | `loguru` | >=0.7 | 结构化日志 | Phase 1 W1 |
| 数据源 | `polygon-api-client` | >=1.13 | Polygon.io | Phase 1 W2 |
| 数据源 | `fredapi` | >=0.5 | FRED | Phase 1 W2 |
| 异步任务 | `celery` | >=5.3 | 任务队列 | Phase 2 W13 |
| 缓存 | `redis` | >=5.0 | Redis 客户端 | Phase 2 W13 |
| 编排 | `apache-airflow` | >=2.8 | DAG 编排 | Phase 2 W13 |
| 前端 | `react` | >=18.3 | UI 框架 | Phase 3 W19 |
| 图表 | `echarts` | >=5.5 | 数据可视化 | Phase 3 W19 |
| K线图 | `lightweight-charts` | >=4.1 | 金融K线 | Phase 3 W19 |

### 6.2 开发工具

| 工具 | 用途 |
|------|------|
| `pytest` | 单元测试 |
| `ruff` | 代码格式化+检查 |
| `mypy` | 类型检查 |
| `docker` + `docker-compose` | 容器化部署 |
| `git` | 版本控制 |

---

## 7. 数据库核心表设计

### 7.1 表结构详细定义

**STOCKS（股票基础信息）**
```sql
CREATE TABLE stocks (
    ticker          VARCHAR(10) PRIMARY KEY,
    company_name    VARCHAR(200) NOT NULL,
    sector          VARCHAR(50),
    industry        VARCHAR(100),
    ipo_date        DATE,
    delist_date     DATE,              -- NULL 表示仍在上市
    delist_reason   VARCHAR(50),       -- 'acquired', 'bankrupt', 'voluntary', NULL
    shares_outstanding BIGINT,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);
```

**STOCK_PRICES（行情数据 — TimescaleDB 超级表）**
```sql
CREATE TABLE stock_prices (
    ticker          VARCHAR(10) NOT NULL,
    trade_date      DATE NOT NULL,
    open            DECIMAL(12,4),
    high            DECIMAL(12,4),
    low             DECIMAL(12,4),
    close           DECIMAL(12,4),
    adj_close       DECIMAL(12,4),
    volume          BIGINT,
    knowledge_time  TIMESTAMPTZ NOT NULL,  -- 数据首次可获取时间
    source          VARCHAR(20),           -- 'polygon', 'backup'
    PRIMARY KEY (ticker, trade_date)
);
-- 转为 TimescaleDB 超级表
SELECT create_hypertable('stock_prices', 'trade_date');
```

**FUNDAMENTALS_PIT（PIT 基本面数据）**
```sql
CREATE TABLE fundamentals_pit (
    id              SERIAL PRIMARY KEY,
    ticker          VARCHAR(10) NOT NULL,
    fiscal_period   VARCHAR(10) NOT NULL,  -- '2024Q1', '2024FY'
    metric_name     VARCHAR(50) NOT NULL,  -- 'revenue', 'net_income', 'eps'
    metric_value    DECIMAL(20,6),
    event_time      DATE NOT NULL,         -- 财报截至日期
    knowledge_time  TIMESTAMPTZ NOT NULL,  -- SEC 实际发布时间
    is_restated     BOOLEAN DEFAULT FALSE, -- 是否为重述数据
    source          VARCHAR(20),
    UNIQUE (ticker, fiscal_period, metric_name, knowledge_time)
);
CREATE INDEX idx_fundamentals_pit_lookup
    ON fundamentals_pit (ticker, knowledge_time, metric_name);
```

**UNIVERSE_MEMBERSHIP（历史成分股成员资格）**
```sql
CREATE TABLE universe_membership (
    id              SERIAL PRIMARY KEY,
    ticker          VARCHAR(10) NOT NULL,
    index_name      VARCHAR(20) NOT NULL,  -- 'SP500'
    effective_date  DATE NOT NULL,         -- 加入日期
    end_date        DATE,                  -- 移除日期 (NULL=当前成员)
    reason          VARCHAR(50),           -- 'added', 'removed', 'acquired'
    UNIQUE (ticker, index_name, effective_date)
);
```

**CORPORATE_ACTIONS（公司行为记录）**
```sql
CREATE TABLE corporate_actions (
    id              SERIAL PRIMARY KEY,
    ticker          VARCHAR(10) NOT NULL,
    action_type     VARCHAR(20) NOT NULL,  -- 'split', 'dividend', 'ticker_change', 'delist'
    ex_date         DATE NOT NULL,
    ratio           DECIMAL(10,6),         -- 拆分比例或分红金额
    old_ticker      VARCHAR(10),           -- 代码变更时的旧代码
    new_ticker      VARCHAR(10),           -- 代码变更时的新代码
    details_json    JSONB
);
```

**FEATURE_STORE（特征数据）**
```sql
CREATE TABLE feature_store (
    id              SERIAL PRIMARY KEY,
    ticker          VARCHAR(10) NOT NULL,
    calc_date       DATE NOT NULL,
    feature_name    VARCHAR(50) NOT NULL,
    feature_value   DECIMAL(20,8),
    is_filled       BOOLEAN DEFAULT FALSE, -- 是否为前向填充值
    batch_id        VARCHAR(36) NOT NULL,  -- UUID, 标识本批次计算
    UNIQUE (ticker, calc_date, feature_name, batch_id)
);
SELECT create_hypertable('feature_store', 'calc_date');
```

**MODEL_REGISTRY（模型版本管理）**
```sql
CREATE TABLE model_registry (
    model_id        VARCHAR(36) PRIMARY KEY, -- UUID
    model_name      VARCHAR(100) NOT NULL,
    version         VARCHAR(20) NOT NULL,
    model_type      VARCHAR(20) NOT NULL,    -- 'ridge', 'xgboost', 'lgbm', 'lstm'
    train_start     DATE NOT NULL,
    train_end       DATE NOT NULL,
    val_start       DATE,
    val_end         DATE,
    features_json   JSONB NOT NULL,          -- 使用的特征列表
    hyperparams_json JSONB,
    metrics_json    JSONB NOT NULL,          -- IC, RankIC, ICIR, DSR 等
    status          VARCHAR(20) DEFAULT 'staging', -- 'staging', 'production', 'archived'
    mlflow_run_id   VARCHAR(36),
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
```

**PREDICTIONS（预测记录）**
```sql
CREATE TABLE predictions (
    id                SERIAL PRIMARY KEY,
    ticker            VARCHAR(10) NOT NULL,
    signal_date       DATE NOT NULL,
    model_version_id  VARCHAR(36) NOT NULL REFERENCES model_registry(model_id),
    feature_batch_id  VARCHAR(36) NOT NULL,
    pred_score        DECIMAL(10,6) NOT NULL,
    pred_rank         INTEGER NOT NULL,       -- 横截面排名 (1=最高)
    pred_decile       INTEGER NOT NULL,       -- 十分位 (1=最高)
    confidence        DECIMAL(6,4),           -- 模型置信度
    UNIQUE (ticker, signal_date, model_version_id)
);
SELECT create_hypertable('predictions', 'signal_date');
```

**PORTFOLIOS（组合配置）**
```sql
CREATE TABLE portfolios (
    portfolio_id    VARCHAR(36) PRIMARY KEY,
    user_id         VARCHAR(36),
    strategy_name   VARCHAR(100),
    weighting_scheme VARCHAR(20),  -- 'equal', 'vol_weighted', 'black_litterman'
    config_json     JSONB,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
```

**BACKTEST_RESULTS（回测结果）**
```sql
CREATE TABLE backtest_results (
    backtest_id         VARCHAR(36) PRIMARY KEY,
    model_version_id    VARCHAR(36) REFERENCES model_registry(model_id),
    config_snapshot_json JSONB NOT NULL,  -- 完整回测配置快照
    metrics_json        JSONB NOT NULL,   -- 年化收益/夏普/回撤等
    equity_curve        JSONB NOT NULL,   -- 每期净值序列
    stat_tests_json     JSONB,            -- DSR/SPA/Bootstrap 结果
    created_at          TIMESTAMPTZ DEFAULT NOW()
);
```

**AUDIT_LOG（审计日志）**
```sql
CREATE TABLE audit_log (
    id                SERIAL PRIMARY KEY,
    timestamp         TIMESTAMPTZ DEFAULT NOW(),
    action            VARCHAR(50) NOT NULL,  -- 'signal_generated', 'rebalance', 'risk_breach'
    actor             VARCHAR(50) NOT NULL,  -- 'system', 'model_v2.1', 'risk_engine'
    model_version_id  VARCHAR(36),
    feature_batch_id  VARCHAR(36),
    details_json      JSONB NOT NULL
);
CREATE INDEX idx_audit_log_time ON audit_log (timestamp DESC);
```

---

## 8. API 接口设计

### Phase 3 完整 API 列表

**市场数据 (`/api/market/`)**

| 方法 | 路径 | 说明 | 响应时间目标 |
|------|------|------|-------------|
| GET | `/api/market/overview` | 市场概览（指数+行业+热力图） | < 100ms |
| GET | `/api/market/indices` | 主要指数行情 | < 50ms |
| GET | `/api/market/sectors` | 行业板块表现 | < 100ms |

**个股数据 (`/api/stocks/`)**

| 方法 | 路径 | 说明 | 响应时间目标 |
|------|------|------|-------------|
| GET | `/api/stocks/{ticker}` | 个股基础信息 | < 50ms |
| GET | `/api/stocks/{ticker}/prices` | 历史行情 (支持分页) | < 100ms |
| GET | `/api/stocks/{ticker}/features` | 当前特征值 | < 100ms |
| GET | `/api/stocks/universe` | 当前 Universe 成员列表 | < 100ms |

**模型预测 (`/api/predictions/`)**

| 方法 | 路径 | 说明 | 响应时间目标 |
|------|------|------|-------------|
| GET | `/api/predictions/latest` | 最新一期预测 (Top Decile) | < 200ms |
| GET | `/api/predictions/{ticker}/detail` | 个股预测详情 + SHAP 归因 | < 2000ms |
| GET | `/api/predictions/{ticker}/history` | 预测分数历史 (60天) | < 200ms |
| GET | `/api/predictions/model-performance` | 模型历史表现回溯 | < 500ms |

**组合 (`/api/portfolio/`)**

| 方法 | 路径 | 说明 | 响应时间目标 |
|------|------|------|-------------|
| GET | `/api/portfolio/model` | 当前模型跟踪组合 | < 200ms |
| GET | `/api/portfolio/efficient-frontier` | 有效前沿数据 | < 1000ms |
| GET | `/api/portfolio/risk-decomposition` | 风险分解 (行业/因子) | < 500ms |

**回测 (`/api/backtest/`)**

| 方法 | 路径 | 说明 | 响应时间目标 |
|------|------|------|-------------|
| POST | `/api/backtest/run` | 提交回测任务（异步 Celery） | < 200ms (提交) |
| GET | `/api/backtest/{id}/status` | 回测任务状态 | < 50ms |
| GET | `/api/backtest/{id}/result` | 回测结果 | < 500ms |
| GET | `/api/backtest/{id}/report/pdf` | 下载 PDF 报告 | < 3000ms |

**用户认证 (`/api/auth/`)**

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/auth/register` | 用户注册 |
| POST | `/api/auth/login` | 登录获取 JWT |
| POST | `/api/auth/refresh` | Token 刷新 |
| GET | `/api/auth/me` | 当前用户信息 |

---

## 9. 风险评估与应对

### 9.1 技术风险

| 风险 | 概率 | 影响 | 应对措施 |
|------|------|------|----------|
| Polygon.io API 不稳定或数据不完整 | 中 | 中 | 预设 EODHD 作为备用源；缓存已拉取数据 |
| FMP 基本面数据 PIT 属性不可靠 | 高 | 高 | 保守估计 knowledge_time (+45天)；Tier 2 升级 Zacks |
| 协方差矩阵病态导致优化器不收敛 | 中 | 中 | Shrinkage 强制使用；不收敛时降级到 Vol-weighted |
| LSTM 严重过拟合，深度模型无增量 | 高 | 低 | 放弃深度模型，使用 Tree 模型即可 |
| MLflow 本地存储丢失实验数据 | 低 | 高 | 定期备份 mlruns/；Phase 2 迁移到 DB 后端 |
| TimescaleDB 大数据量查询性能不足 | 低 | 中 | 合理建索引；物化视图加速常用查询 |

### 9.2 业务风险

| 风险 | 概率 | 影响 | 应对措施 |
|------|------|------|----------|
| Alpha 在样本外完全不成立 | 中 | 致命 | Phase 1 硬性准出标准，不通过则 pivot |
| 市场 Regime 剧变导致模型失效 | 中 | 高 | 多窗口 Walk-Forward + 实时 IC 监控 + 自动降权 |
| SEC 合规风险 | 低 | 高 | 从 Day 1 嵌入合规约束；正式上线前律师审核 |
| 数据源涨价或停止服务 | 低 | 中 | 数据源适配器模式，可快速切换 |

### 9.3 执行风险

| 风险 | 概率 | 影响 | 应对措施 |
|------|------|------|----------|
| 单人/小团队进度滞后 | 高 | 高 | 严格优先级排序，Phase 1 只做核心路径 |
| 过度工程化消耗时间 | 高 | 高 | MLOps 分期实施，MVP 阶段最小化基础设施 |
| Phase 2/3 范围蔓延 | 中 | 中 | 严格遵循准出标准，功能冻结后不加新需求 |

---

## 10. 关键决策检查点

整个项目设置 5 个关键决策检查点 (Decision Gate)，每个检查点都需要明确的 Go/No-Go 决策。

| # | 检查点 | 时间 | 决策内容 | 关键问题 |
|---|--------|------|----------|----------|
| G1 | 数据就绪 | 第 2 周末 | 数据管道是否可靠 | 数据完整性是否满足最低要求？ |
| G2 | 特征有效性 | 第 4 周末 | 特征是否有信号 | 是否有超过 30 个特征 IC > 0.01？ |
| G3 | **Alpha 验证** | **第 10 周末** | **Go/No-Go 核心决策** | **DSR p < 0.05？成本后超额 > 5%？** |
| G4 | 生产就绪 | 第 18 周末 | Shadow Mode 是否通过 | 4 周运行无异常？风控系统可靠？ |
| G5 | 产品发布 | 第 28 周末 | 是否可上线 | 联调通过？合规审查通过？安全审计通过？ |

**G3 是整个项目的命运决定点。** 如果 Alpha 不成立，应立即停止后续投入，而非"带着侥幸心理继续做产品"。

---

## 附录：周度时间线总览

| 周 | 阶段 | 核心任务 | 关键交付物 |
|----|------|----------|-----------|
| W1 | Phase 1 | 项目基础设施搭建 | 项目骨架 + DB + MLflow |
| W2 | Phase 1 | 数据采集与 PIT 清洗 | 2018-今历史数据入库 |
| W3 | Phase 1 | 特征工程 (上) | 技术+成交量特征 ~30个 |
| W4 | Phase 1 | 特征工程 (下) + 筛选 | 基本面+宏观特征 + IC 筛选 |
| W5 | Phase 1 | Baseline 模型训练 | Ridge + 多因子排序基线 |
| W6 | Phase 1 | Tree 模型训练 | XGBoost + LightGBM |
| W7 | Phase 1 | Deep 模型 + Horizon 矩阵 | LSTM + 多 Horizon 实验 |
| W8 | Phase 1 | Walk-Forward 全量回测 | 5 窗口回测 + 成本模型 |
| W9 | Phase 1 | 仓位方案对比 | 等权/Vol/BL 三方案 |
| W10 | Phase 1 | **统计检验 + Alpha 报告** | **DSR + SPA + Go/No-Go** |
| W11-12 | Phase 2 | MLflow 模型注册 | Champion/Challenger |
| W13-14 | Phase 2 | Airflow + 风控引擎 | DAG 编排 + 四层风控 |
| W15-16 | Phase 2 | 稳健组合优化 | BL + Shrinkage + CVXPY |
| W17-18 | Phase 2 | Shadow Mode | 4 周灰度运行 |
| W19-21 | Phase 3 | 后端 API + 前端 Dashboard | FastAPI + React |
| W22-23 | Phase 3 | 预测中心 + SHAP + 组合 | 解释层 + 组合构建器 |
| W24-25 | Phase 3 | 回测系统 | 用户自定义回测 |
| W26-27 | Phase 3 | 用户系统 + 合规 | JWT + 订阅 + 合规审查 |
| W28 | Phase 3 | 联调 + 压力测试 | 上线就绪 |
