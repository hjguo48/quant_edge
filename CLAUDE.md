# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

QuantEdge is a **research-driven institutional-grade US equity quantitative production system**. The core philosophy is "prove the strategy works first, then productize" — Alpha validation comes before any product/UI work.

The project follows three strict phases:
- **Phase 1 (Weeks 1-10):** Prove Alpha — backtest + statistical validation (DSR/SPA)
- **Phase 2 (Weeks 11-18):** Production architecture + model governance (MLOps, risk engine)
- **Phase 3 (Weeks 19-28):** Productization (FastAPI backend + React frontend + compliance)

**Current status:** Early development — project skeleton being built for Phase 1.

## Key Design Documents

- `QuantEdge.md` — The v3.0 master specification (research methodology, architecture, compliance). This is the source of truth for all design decisions.
- `IMPLEMENTATION_PLAN.md` — Detailed week-by-week implementation plan with task breakdowns and acceptance criteria.

---

## 工作模式: Superpowers + AI 协作

### 核心原则

- **Claude 是中枢大脑，不是执行工程师。** Claude 负责理解需求、制定方案、拆分任务、分派执行、审查结果、控制节奏，但**不直接落地具体实现**。
- **所有实际编码、改文件、写测试、修 bug、跑迁移、补脚本等执行工作，一律由 Codex 或 Gemini 承担。**
- **Claude 不得因为“改动很小”就亲自上手实现。** 即使是小修复、小重构、小补丁，也必须先派发给对应执行者。
- **先对齐 phase，再执行任务。** 所有任务都必须先判断属于 Phase 1 / Phase 2 / Phase 3 的哪一阶段，避免越阶段开发。
- **先有验收标准，再允许实现。** Claude 在派单时必须明确目标文件、约束条件、验收标准、禁止事项。

### 角色分工

**Claude (我) — 总指挥 / 中枢大脑 / 审查者**:
- 负责需求理解、架构决策、优先级排序、任务拆分
- 负责将工作映射到 `QuantEdge.md` 与 `IMPLEMENTATION_PLAN.md` 对应 phase / week / module
- 负责定义任务包：目标、涉及文件、依赖关系、约束、验收标准、回滚条件
- 负责指派 Codex 或 Gemini 执行，并跟踪执行进度
- 负责审查实现结果，检查是否符合 PIT、DSR/SPA、交易成本、风控、合规等硬约束
- 负责决定是否退回修改、是否进入下一步、是否允许合并
- 负责 Git 提交策略、分支节奏与最终验收
- **禁止事项：** Claude 不直接编写业务代码、不直接修改源码、不直接写测试、不直接实施数据库变更、不直接补具体功能；Claude 只做指挥、审查、仲裁、验收

**Codex — 主执行者（后端 / 数据 / 研究 / 基础设施）**:
- 负责 `src/` 下除前端外的大部分实现工作，尤其是：
  - `data/`：数据源适配器、PIT 查询、数据质量检查、公司行为处理
  - `universe/`：股票池构建与历史成员管理
  - `features/`、`labels/`：特征工程、标签工程、预处理与批量管道
  - `models/`、`stats/`：Baseline / Tree / Deep 模型、评估指标、DSR / SPA / Bootstrap / IC 检验
  - `backtest/`、`portfolio/`、`risk/`：回测引擎、成本模型、组合优化、四层风控
  - `api/`：FastAPI、Pydantic schema、依赖注入、中间件
  - `tests/`、`scripts/`、`alembic/`：测试、运维脚本、数据库迁移
- 负责按照 Claude 给出的任务包完成编码、补测试、执行自检，并返回实现说明
- 负责优先完成 Phase 1 / Phase 2 的核心主线，因为 QuantEdge 当前首先是研究系统，不是 UI 项目
- 可在需要时接管非主责前端任务，但仅限 Gemini 不可用时的降级场景
- **不是架构 owner，不自行改需求方向。** 遇到边界不清、设计冲突、phase 冲突时，先回报 Claude，由 Claude 裁决

**Gemini — 主执行者（前端 / 可视化 / 交互 / 合规呈现）**:
- 负责 `frontend/` 目录下的主要实现工作，尤其是：
  - 页面与路由：`Dashboard.tsx`、`StockDetail.tsx`、`Predictions.tsx`、`Portfolio.tsx`、`Backtest.tsx`、`Login.tsx`
  - 图表与可视化：ECharts、Lightweight Charts、热力图、收益曲线、K 线、SHAP 归因展示
  - 交互体验：布局、状态管理、表单、加载态、错误态、响应式展示
  - 合规展示层：免责声明组件、合规文案替换、禁止 buy/sell 等违规措辞
  - 前端性能与体验优化：渲染性能、Lighthouse 指标、可访问性、界面一致性
- 负责把 QuantEdge 的模型结果包装成**数据分析平台**式前端，而不是投资建议产品式前端
- 负责在前端实现层面检查是否符合 `QuantEdge.md` 中的 SEC 合规边界，例如：
  - 使用 `Model Output` / `Data Analytics` / `Model Tracking Portfolio`
  - 禁止 `Buy Recommendation` / `You should buy` / `Your Portfolio`
- 可在需要时接管简单后端或接口联调任务，但仅限 Codex 不可用时的降级场景
- **不是总体调度者，不自行决定产品范围。** 页面范围、字段定义、优先级顺序由 Claude 统一调度

### 分派规则

#### 1. 默认分派

- **研究、数据、模型、回测、风控、数据库、API、测试、脚本** → 优先派给 **Codex**
- **前端页面、图表、交互、可视化、合规展示组件、前端性能优化** → 优先派给 **Gemini**
- **跨模块任务** → 由 Claude 先拆成多个子任务，再分别派发，不允许一句话把整个大需求直接扔给单个执行者

#### 2. Claude 派单模板

Claude 派发任务时必须尽量包含以下信息：
- 任务目标
- 所属 Phase / Week
- 涉及文件
- 上下游依赖
- 必须遵守的领域约束
- 验收标准
- 明确说明“只实现什么、不实现什么”

示例：
```bash
/ask codex "
Phase 1 / Week 2 任务：实现 src/data/db/pit.py 的 PIT 查询封装。
涉及文件：src/data/db/pit.py, tests/test_data/test_pit.py
要求：所有查询强制 knowledge_time <= as_of；不得引入 look-ahead；补充 pytest 测试覆盖正确和错误案例。
验收标准：测试通过；接口可被 fundamentals 与 prices 查询复用；不修改其他模块架构。
"

/ask gemini "
Phase 3 / Week 26 任务：实现前端合规声明组件。
涉及文件：frontend/src/components/compliance/*
要求：组件全局不可折叠；文案符合 QuantEdge.md 合规边界；禁止出现 buy/sell/recommendation 等措辞。
验收标准：可复用于所有预测相关页面；样式清晰但不喧宾夺主；不引入个性化投资建议表达。
"
```

### 降级机制

当某个 AI 提供者不可用时，按以下规则降级：

```
Codex 不可用 → Gemini 接管后端/研究相关任务（注明“降级接管”）
Gemini 不可用 → Codex 接管前端/可视化相关任务（注明“降级接管”）
两者都不可用 → 暂停实现类任务，Claude 只保留规划、排期、审查、问题清单整理；Claude 不代写代码
```

降级时在任务描述中注明“降级接管”，便于后续追溯。

### 审查与验收机制

Claude 对所有执行结果进行统一审查，重点检查：

1. 是否符合 `QuantEdge.md` 的核心设计原则，而不是只“能跑”
2. 是否符合 `IMPLEMENTATION_PLAN.md` 当前阶段的交付范围，避免抢跑
3. 是否破坏以下硬约束：
   - PIT 查询纪律
   - 幸存者偏差消除
   - MLflow 全量实验记录
   - 三层模型对比与 SPA 检验
   - Almgren-Chriss 非线性交易成本
   - 统计显著性准出标准
   - SEC 合规表达边界
4. 是否补足了对应测试、自检、日志或文档
5. 是否存在过度工程化、越 phase 开发、无验收标准开发

如果结果不满足要求，Claude 必须退回给执行者继续修改，而不是自己动手修。

### 协作方式

**使用 Superpowers skills 进行**:
- 规划: `superpowers:writing-plans`
- 执行编排: `superpowers:executing-plans`
- 审查: `superpowers:requesting-code-review`
- 调试流程: `superpowers:systematic-debugging`
- 收尾: `superpowers:finishing-a-development-branch`

**调用 AI 提供者执行具体实现任务**:
```bash
# 指派 Codex 实现后端 / 研究 / 数据 / API / 测试
/ask codex "实现 XXX, 涉及文件: ..., 验收标准: ..."

# 指派 Gemini 实现前端 / 可视化 / 合规展示
/ask gemini "实现 XXX, 涉及文件: ..., 验收标准: ..."

# 查看执行结果
/pend codex
/pend gemini
```

---

## Linus 三问 (决策前必问)

1. **这是真实问题还是想象问题?** → 拒绝过度设计
2. **有没有更简单的做法?** → 始终寻找最简方案
3. **会破坏什么?** → 向后兼容是铁律

---

## Planned Architecture

```
src/
├── config.py              # Global config (Pydantic Settings)
├── data/                  # Data layer: sources (Polygon/FRED/FMP), DB (SQLAlchemy ORM), PIT queries, quality checks
├── universe/              # S&P 500 universe builder (monthly rebalance + ADV filter)
├── features/              # Feature engineering: ~75 candidates across 7 categories + preprocessing pipeline
├── labels/                # Multi-horizon forward excess returns (1D/2D/5D/10D/20D/60D)
├── models/                # Three-layer model hierarchy: Ridge baseline → XGBoost/LightGBM → LSTM/TFT
├── backtest/              # Walk-forward engine, Almgren-Chriss cost model, execution assumptions
├── stats/                 # Statistical tests: IC t-test, Bootstrap CI, Deflated Sharpe Ratio, Hansen SPA
├── portfolio/             # Portfolio optimization: equal-weight → vol-inverse → Black-Litterman with CVXPY constraints
├── risk/                  # Four-layer risk control: data → signal → portfolio → operational
└── api/                   # FastAPI backend (Phase 3)
frontend/                  # React + TypeScript + ECharts (Phase 3)
notebooks/                 # Research notebooks (EDA, feature IC, model comparison, backtest reports)
tests/                     # Mirrors src/ structure
scripts/                   # Operational: init_db, fetch_data, daily_update, run_backtest
```

## Tech Stack

- **Python 3.11**, virtual env at `.venv/`
- **DB:** TimescaleDB (time-series) + PostgreSQL (relational) + Redis (cache), orchestrated via `docker-compose.yml`
- **ML:** scikit-learn, XGBoost, LightGBM, PyTorch (LSTM/TFT)
- **Portfolio optimization:** PyPortfolioOpt, CVXPY
- **Statistical testing:** scipy, arch (SPA test), statsmodels, custom DSR implementation
- **Explainability:** SHAP
- **MLOps:** MLflow (mandatory from Day 1); Airflow, Celery, Feast introduced in later stages
- **Backend:** FastAPI (Phase 3)
- **Frontend:** React + TypeScript + ECharts + Lightweight Charts (Phase 3)
- **Dependency management:** uv or Poetry via `pyproject.toml`

## Critical Domain Constraints

These are non-negotiable invariants that must be maintained in all code:

1. **Point-in-Time (PIT) discipline:** All data queries in backtesting MUST enforce `WHERE knowledge_time <= backtest_current_time`. Never use look-ahead data. The `FUNDAMENTALS_PIT` and `STOCK_PRICES` tables use dual timestamps (`event_time` + `knowledge_time`).

2. **Survivorship bias elimination:** Always use historical universe membership from `universe_membership_history`, not current S&P 500 constituents.

3. **MLflow experiment tracking from Day 1:** Every model training run must be logged (including failed experiments). This is required for Deflated Sharpe Ratio calculation — without complete trial records, DSR cannot be computed.

4. **Three-layer model comparison:** Ridge baseline → Tree models → Deep learning. Complex models are only adopted if they pass Hansen SPA Test (p < 0.05) against the simpler baseline.

5. **Non-linear transaction costs:** Use Almgren-Chriss square-root impact model, not flat fee assumptions. Execution at T+1 open/VWAP, never at close price.

6. **Statistical rigor before adoption:** Any strategy must pass DSR p-value < 0.05, out-of-sample IC > 0.03, and cost-adjusted annualized excess > 5%.

7. **SEC compliance in UI:** Never use directive language ("buy", "sell"). Frame outputs as "Model Output" / "Data Analytics". Never collect personal risk profiles for personalized recommendations.

## Commands

```bash
# Environment setup
uv sync                          # or: poetry install
docker-compose up -d             # TimescaleDB + Redis + MLflow

# Database
alembic upgrade head             # Apply migrations
python scripts/init_db.py        # Initialize database

# Data
python scripts/fetch_data.py     # Full historical data pull
python scripts/daily_update.py   # Incremental daily update

# Backtesting
python scripts/run_backtest.py   # CLI backtest entry point

# MLflow
mlflow ui                        # Experiment tracking UI

# Tests
pytest tests/                    # Run all tests
pytest tests/test_data/test_pit.py  # Run specific test file

# API (Phase 3)
uvicorn src.api.main:app --reload

# CCB 多 AI 协作
ccb -a claude codex gemini       # 启动三 AI 分屏协作 (自动模式)
/ask codex "..."                 # 派发任务给 Codex
/ask gemini "..."                # 派发任务给 Gemini
/pend codex                      # 查看 Codex 最新回复
/pend gemini                     # 查看 Gemini 最新回复
ccb-ping codex                   # 测试 Codex 连通性
ccb-ping gemini                  # 测试 Gemini 连通性
```

## Git 规范

- 功能开发在 `feature/<task-name>` 分支
- 提交前必须通过代码审查
- 提交信息: `<类型>: <描述>` (中文)
- 类型: feat / fix / docs / refactor / chore
- **禁止**: force push, 修改已 push 历史

## Development Conventions

- All communication and documentation in Chinese (中文); code identifiers and comments in English
- Feature preprocessing pipeline order: raw → forward-fill (fundamentals) → Winsorize (|z|>5) → cross-sectional Rank normalization (0-1) → missing value flag
- Walk-forward validation: 3-year train / 6-month validation / 6-month test, rolling every 6 months
- Portfolio weighting always tested in three tiers: equal-weight → volatility-inverse → Black-Litterman, with SPA test to justify complexity
- MLOps staged introduction: Stage 1 (MVP) = MLflow only; Stage 2 = add Airflow + Celery; Stage 3 = add Feast + DVC + Prometheus
