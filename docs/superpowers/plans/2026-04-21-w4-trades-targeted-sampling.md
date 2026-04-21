# Week 4: Massive Trades 定向抽样 Implementation Plan

> **For agentic workers:** 本 plan 作为任务包交给 Codex 执行. Claude 负责审查/退回/验收, 不写代码.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 定向拉取 Polygon trades 数据 (top 200 liquidity + event windows), 入库 `stock_trades_sampled`, 产出 4 个 trade-level microstructure 特征, 诊断 5D/弱窗口 alpha.

**Architecture:**
- **不全量入库** — trades 数据量巨大 (单日 TB 级), 只抽取有信息量的样本
- 两个维度圈样: (1) top 200 liquidity 股票 (滚动 ADV), (2) 事件窗口 (earnings ±3d, 大 gap 日, S1 弱窗口 W5/W6/W11)
- 数据源优先 Polygon REST `/v3/trades/{ticker}` (逐股逐日), 避免 flat files 全量成本
- 特征计算在入库后一次 batch, 不走 streaming — Week 4 先出 4 个核心 feature, 不追求 production pipeline

**Tech Stack:**
- 数据: Polygon REST API (trades endpoint)
- 存储: TimescaleDB hypertable `stock_trades_sampled` + state 表 `trades_sampling_state`
- 特征: `src/features/trade_microstructure.py` (新模块)
- 脚本: `scripts/build_trades_sample_universe.py` + `scripts/run_trades_sampling.py` + `scripts/build_trade_microstructure_features.py`

**关键约束 (硬性):**
- PIT: 每条 trade 记录必须带 `knowledge_time` (默认 = trade_timestamp + 15min 延迟, 符合 SIP 披露规则)
- 不得改动 Week 3 minute pipeline / universe PIT 逻辑
- 不得扩大 scope 到非定向抽样 (no 全 S&P 500 × 全历史)
- 单日 API 调用预算 < 50k (粗略估算 top200 × 4 event 窗 × 60 day ≈ 48k, 实际更少)

**非 Week 4 范围 (明确不做):**
- Flat files trades 整批入库 — 留给未来 Phase 2 production
- Trade-level SHAP / 归因 — Week 8+
- Intraday realtime ingestion — Phase 3
- Minute-level bid/ask imbalance (需要 quotes) — 单独 spike

---

## 文件结构 (File Structure)

### 新建文件

| 文件 | 职责 |
|---|---|
| `alembic/versions/007_add_stock_trades_sampled.py` | DB migration — trades hypertable + state 表 |
| `src/data/polygon_trades.py` | Polygon `/v3/trades` 客户端封装 (分页/重试/速率) |
| `src/universe/top_liquidity.py` | Top-N liquidity universe 构造 (PIT ADV rank) |
| `src/data/event_calendar.py` | 事件日历聚合: earnings / 大 gap / 弱窗口 |
| `src/features/trade_microstructure.py` | 4 个 trade-level 特征 + 辅助 helper |
| `scripts/build_trades_sample_universe.py` | 一次性构造 sample (ticker, date) 清单 |
| `scripts/run_trades_sampling.py` | 按清单调用 Polygon, 批量入库, 断点续跑 |
| `scripts/build_trade_microstructure_features.py` | 基于 sampled trades 计算 4 特征 |
| `scripts/run_week4_gate_verification.py` | 产出 Gate 报告 (覆盖 / 缺失 / per-window IC 可行性) |
| `tests/test_data/test_polygon_trades.py` | Polygon trades 客户端测试 (mock HTTP) |
| `tests/test_universe/test_top_liquidity.py` | ADV rank 单测 |
| `tests/test_data/test_event_calendar.py` | 事件聚合单测 |
| `tests/test_features/test_trade_microstructure.py` | 4 特征单测 |
| `tests/test_scripts/test_week4_gate.py` | Gate 报告 smoke |

### 修改文件

| 文件 | 修改点 |
|---|---|
| `src/features/registry.py` | 注册 4 个新特征到 `trade_microstructure` family |
| `configs/research/data_lineage.yaml` | 添加 trade features 血统记录 |
| `src/config.py` | 新增 `TRADES_SAMPLING_MAX_DAILY_CALLS = 50000` (预算) |
| `IMPLEMENTATION_PLAN.md` | Week 4 段落更新进度 |

### 不得修改

- `src/data/polygon_flat_files.py` / `src/data/polygon_minute.py` — Week 3 已冻结
- `src/universe/builder.py` — PIT universe 逻辑 (Week 2.5 已修复)
- `src/features/intraday.py` — Week 3.2 已稳定

---

## 任务拆分 (Tasks)

### Task 1: DB Schema — stock_trades_sampled + 状态表

**Files:**
- Create: `alembic/versions/007_add_stock_trades_sampled.py`

- [ ] **Step 1.1: 写 migration (up + down)**

```python
"""add stock_trades_sampled hypertable + state table

Revision ID: 007
Revises: 006
"""
from alembic import op
import sqlalchemy as sa

revision = "007"
down_revision = "006"

def upgrade():
    op.create_table(
        "stock_trades_sampled",
        sa.Column("ticker", sa.String(16), nullable=False),
        sa.Column("event_time", sa.DateTime(timezone=True), nullable=False),   # trade timestamp (UTC)
        sa.Column("knowledge_time", sa.DateTime(timezone=True), nullable=False),  # event_time + 15min
        sa.Column("price", sa.Numeric(12, 4), nullable=False),
        sa.Column("size", sa.Integer, nullable=False),
        sa.Column("exchange", sa.SmallInteger, nullable=True),
        sa.Column("conditions", sa.ARRAY(sa.SmallInteger), nullable=True),
        sa.Column("trade_id", sa.String(64), nullable=False),
        sa.Column("sampled_reason", sa.String(32), nullable=False),  # 'top_liquidity' | 'earnings' | 'gap' | 'weak_window'
        sa.PrimaryKeyConstraint("ticker", "event_time", "trade_id"),
    )
    op.execute(
        "SELECT create_hypertable('stock_trades_sampled', 'event_time', "
        "chunk_time_interval => INTERVAL '1 day', if_not_exists => TRUE);"
    )
    op.create_index(
        "ix_trades_sampled_knowledge_time",
        "stock_trades_sampled",
        ["ticker", "knowledge_time"],
    )
    # compression policy — compress chunks older than 7 days
    op.execute(
        "ALTER TABLE stock_trades_sampled SET ("
        "timescaledb.compress, "
        "timescaledb.compress_segmentby = 'ticker', "
        "timescaledb.compress_orderby = 'event_time DESC'"
        ");"
    )
    op.execute(
        "SELECT add_compression_policy('stock_trades_sampled', INTERVAL '7 days', if_not_exists => TRUE);"
    )

    op.create_table(
        "trades_sampling_state",
        sa.Column("ticker", sa.String(16), nullable=False),
        sa.Column("trading_date", sa.Date, nullable=False),
        sa.Column("sampled_reason", sa.String(32), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),  # pending | in_progress | completed | failed | skipped_holiday | skipped_no_data
        sa.Column("rows_ingested", sa.Integer, nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_message", sa.Text, nullable=True),
        sa.PrimaryKeyConstraint("ticker", "trading_date", "sampled_reason"),
    )
    op.create_index(
        "ix_trades_sampling_state_status",
        "trades_sampling_state",
        ["status", "trading_date"],
    )

def downgrade():
    op.drop_index("ix_trades_sampling_state_status", table_name="trades_sampling_state")
    op.drop_table("trades_sampling_state")
    op.drop_index("ix_trades_sampled_knowledge_time", table_name="stock_trades_sampled")
    op.drop_table("stock_trades_sampled")
```

- [ ] **Step 1.2: 验证 migration**

```bash
alembic upgrade head
alembic downgrade -1
alembic upgrade head
```
Expected: 两表创建成功, hypertable chunks 表可见, compression policy `_timescaledb_config.bgw_policy_compression` 有记录.

- [ ] **Step 1.3: Commit**

```bash
git add alembic/versions/007_add_stock_trades_sampled.py
git commit -m "feat(week4): 新增 stock_trades_sampled + trades_sampling_state 表"
```

---

### Task 2: Top-N Liquidity Universe (PIT ADV rank)

**Files:**
- Create: `src/universe/top_liquidity.py`
- Test: `tests/test_universe/test_top_liquidity.py`

**接口:**

```python
def get_top_liquidity_tickers(
    as_of_date: date,
    *,
    top_n: int = 200,
    lookback_days: int = 20,
    session_factory: Callable | None = None,
) -> list[str]:
    """返回 as_of_date 前 lookback_days 窗口内 ADV (dollar volume) 排名 top_n 的 ticker.
    查询强制 knowledge_time <= as_of_date (PIT).
    ADV = mean(close * volume) over past lookback_days sessions.
    只纳入当日 universe_membership_history 中的 ticker.
    """
```

- [ ] **Step 2.1: 写 4 个测试**

```python
# tests/test_universe/test_top_liquidity.py
def test_top_n_ranks_by_dollar_volume(db_session):
    # 种子 5 ticker, 不同 close*volume, 验证排序
    ...

def test_respects_pit_knowledge_time(db_session):
    # 种子 knowledge_time > as_of 的数据, 不应出现
    ...

def test_filters_by_universe_membership(db_session):
    # ticker A 在 stock_prices 但不在 universe_membership_history, 应排除
    ...

def test_handles_missing_history(db_session):
    # lookback 内 < 5 个交易日, 应 fallback 到可用日并 log warning
    ...
```

- [ ] **Step 2.2: 实现**

SQL 骨架:
```sql
WITH eligible_tickers AS (
    SELECT DISTINCT ticker FROM universe_membership_history
    WHERE effective_date <= :as_of
      AND (expiry_date IS NULL OR expiry_date > :as_of)
),
adv AS (
    SELECT sp.ticker, AVG(sp.close * sp.volume) AS dollar_adv
    FROM stock_prices sp
    JOIN eligible_tickers et USING (ticker)
    WHERE sp.event_time BETWEEN :lookback_start AND :as_of
      AND sp.knowledge_time <= :as_of_eod
    GROUP BY sp.ticker
)
SELECT ticker FROM adv ORDER BY dollar_adv DESC NULLS LAST LIMIT :top_n;
```

- [ ] **Step 2.3: 跑测试 + commit**

```bash
pytest tests/test_universe/test_top_liquidity.py -v
# Expected: 4 passed
git add src/universe/top_liquidity.py tests/test_universe/test_top_liquidity.py
git commit -m "feat(week4): top-N liquidity universe (PIT ADV rank)"
```

---

### Task 3: Event Calendar Builder

**Files:**
- Create: `src/data/event_calendar.py`
- Test: `tests/test_data/test_event_calendar.py`

**接口:**

```python
@dataclass(frozen=True)
class SamplingEvent:
    ticker: str
    trading_date: date
    reason: str  # 'top_liquidity' | 'earnings' | 'gap' | 'weak_window'

def build_sampling_plan(
    *,
    start_date: date,
    end_date: date,
    top_n_liquidity: int = 200,
    earnings_window_days: int = 3,     # ±3d around earnings
    gap_threshold_pct: float = 0.03,    # |overnight ret| > 3% triggers sampling
    weak_windows: list[tuple[date, date]] | None = None,
) -> list[SamplingEvent]:
    """产出 (ticker, trading_date, reason) 清单, 用于后续 trades 拉取.
    同一 (ticker, date) 可能命中多个 reason — 保留全部, 入库时 reason 用 pipe-join 可行,
    但当前方案: 记录 4 条不同 reason 的 state, 实际 trades 数据只拉 1 次, 去重逻辑在 scripts 层.
    """
```

**关键决策: 事件源**
- earnings: 从 `earnings_estimates` 表 (已存在) 查 `event_time`
- gap: 从 `stock_prices` 计算 `(open_t - close_{t-1}) / close_{t-1}`
- weak_window: 硬编码 S1 walk-forward 的 W5/W6/W11 日期区间 (从 IMPLEMENTATION_PLAN 附录或直接传入)
- top_liquidity: 每日前 200 液化名单, reason=`top_liquidity`

**SEC filings 窗口 (spec 原文提到)**: Week 4 不做, 延后到 Week 5 — `ftd_pit` / `fmp_earnings_calendar` 数据到位后再补 SEC event window. Week 4 用 earnings + gap + weak_window 四项已能覆盖 Gate.

- [ ] **Step 3.1: 写 5 个测试** (每个 reason 一个 + 去重 1 个)
- [ ] **Step 3.2: 实现 build_sampling_plan**
- [ ] **Step 3.3: 跑测试**

```bash
pytest tests/test_data/test_event_calendar.py -v
# Expected: 5 passed
```

- [ ] **Step 3.4: Commit**

```bash
git add src/data/event_calendar.py tests/test_data/test_event_calendar.py
git commit -m "feat(week4): 事件日历聚合 (earnings/gap/weak_window/top_liquidity)"
```

---

### Task 4: Polygon Trades 客户端

**Files:**
- Create: `src/data/polygon_trades.py`
- Test: `tests/test_data/test_polygon_trades.py`

**接口:**

```python
class PolygonTradesClient(DataSource):
    source_name = "polygon_trades"

    def fetch_trades_for_day(
        self,
        ticker: str,
        trading_date: date,
        *,
        limit_per_page: int = 50000,
        max_pages: int = 200,
    ) -> Iterator[dict[str, Any]]:
        """流式返回单日所有 trade (含 pre/after hours).
        URL: https://api.polygon.io/v3/trades/{ticker}?timestamp={YYYY-MM-DD}
        分页: response['next_url']
        速率: 遵循 self.min_request_interval (默认 0.1s, 可调)
        重试: 5xx / 429 指数退避, max 3 次
        """
```

- [ ] **Step 4.1: 写 4 个测试** (成功路径 / 分页 / 429 重试 / 无数据空返回)
- [ ] **Step 4.2: 实现 (复用 `src/data/sources/base.py` 的 DataSource pattern)**
- [ ] **Step 4.3: 跑测试**
- [ ] **Step 4.4: Commit**

```bash
git add src/data/polygon_trades.py tests/test_data/test_polygon_trades.py
git commit -m "feat(week4): Polygon /v3/trades REST 客户端"
```

---

### Task 5: Sampling Plan Builder Script

**Files:**
- Create: `scripts/build_trades_sample_universe.py`

**CLI:**

```bash
uv run python scripts/build_trades_sample_universe.py \
    --start-date 2016-04-17 \
    --end-date 2026-04-17 \
    --top-n 200 \
    --earnings-window-days 3 \
    --gap-threshold 0.03 \
    --output data/reports/week4/trades_sampling_plan.parquet
```

**产出:**
- Parquet: `(ticker, trading_date, reason)` 去重后 rows
- 同时写入 `trades_sampling_state` (status=`pending`)
- 汇总日志: 每个 reason 的 (ticker, day) 数量

- [ ] **Step 5.1: 写 smoke test** `tests/test_scripts/test_build_trades_sample_universe.py`
- [ ] **Step 5.2: 实现 CLI**
- [ ] **Step 5.3: 本地跑 2016-04-17 → 2016-04-20 小区间 dry-run**
- [ ] **Step 5.4: Commit**

```bash
git add scripts/build_trades_sample_universe.py tests/test_scripts/test_build_trades_sample_universe.py
git commit -m "feat(week4): sampling plan builder script"
```

---

### Task 6: Trades 拉取执行器 + 断点续跑

**Files:**
- Create: `scripts/run_trades_sampling.py`

**CLI:**

```bash
uv run python scripts/run_trades_sampling.py \
    --plan data/reports/week4/trades_sampling_plan.parquet \
    --max-workers 4 \
    --daily-call-budget 50000 \
    --resume
```

**核心逻辑:**
- 从 plan 读 pending (ticker, date), 每个调用 `PolygonTradesClient.fetch_trades_for_day`
- 入库 batch size 10000 rows (COPY 或 executemany)
- 每完成一个 (ticker, date) 原子更新 `trades_sampling_state.status='completed'`
- 失败 (非 auth) 标 `failed` + `error_message`, 下次 `--resume` 从 pending/failed 捡起
- 达到 `--daily-call-budget` 后当日停, 记录 `last_processed_at`
- 遵循 feedback_auto_restart_backfill.md: 适配 supervisor 包装

- [ ] **Step 6.1: 写 smoke test** `tests/test_scripts/test_run_trades_sampling.py` (mock client)
- [ ] **Step 6.2: 实现**
- [ ] **Step 6.3: 本地跑 1 天 (2025-01-10) × 5 ticker dry-run**
- [ ] **Step 6.4: Commit**

```bash
git add scripts/run_trades_sampling.py tests/test_scripts/test_run_trades_sampling.py
git commit -m "feat(week4): trades 拉取执行器 + 断点续跑"
```

---

### Task 7: 4 Trade-Microstructure 特征

**Files:**
- Create: `src/features/trade_microstructure.py`
- Test: `tests/test_features/test_trade_microstructure.py`
- Modify: `src/features/registry.py` (注册 family `trade_microstructure`)

**4 特征:**

```python
def compute_trade_imbalance_proxy(trades: pd.DataFrame) -> float:
    """Lee-Ready tick-rule:
       每 trade 相对前一成交价分正负 (up=+1, down=-1, equal=carry).
       imbalance = sum(sign * size) / sum(size)
       Range: [-1, 1]
    """

def compute_large_trade_ratio(trades: pd.DataFrame, *, size_threshold: int = 10000) -> float:
    """size >= size_threshold 的 trade 成交量占比.
       阈值: 10000 shares (可调, 代表 block trade proxy)
       Range: [0, 1]
    """

def compute_late_day_aggressiveness(trades: pd.DataFrame) -> float:
    """15:00-16:00 ET 区间的 |tick-rule imbalance| / full-day imbalance.
       > 1 意味尾盘更 aggressive (机构挂单).
       返回 ratio (clipped [0, 5]).
    """

def compute_offhours_trade_ratio(trades: pd.DataFrame) -> float:
    """盘前 (04:00-09:30 ET) + 盘后 (16:00-20:00 ET) 成交量 / 全日成交量.
       Range: [0, 1]
    """
```

**registry 挂载** (follow existing pattern — `FeatureDefinition(name, category, description, compute_fn)` + `_TRADE_MICROSTRUCTURE_FEATURE_METADATA` dict + `_register_defaults` 增加一条 for-loop):

```python
# src/features/registry.py
_TRADE_MICROSTRUCTURE_FEATURE_METADATA = {
    "trade_imbalance_proxy": "Tick-rule buy/sell imbalance from sampled trades.",
    "large_trade_ratio": "Share volume from trades >= 10k shares divided by daily share volume.",
    "late_day_aggressiveness": "Imbalance in last market hour vs full day (absolute ratio).",
    "offhours_trade_ratio": "Pre/post-market trade volume share.",
}
# and in _register_defaults:
for name, description in _TRADE_MICROSTRUCTURE_FEATURE_METADATA.items():
    self.register(name, "trade_microstructure", description, compute_trade_microstructure_features)
```

Family 约定通过 `category="trade_microstructure"` 表达 (既有 registry 只有 category 维度, 没有 horizon_affinity). horizon affinity 将在 Week 6 (family 归一) 统一处理, Week 4 不提前.

- [ ] **Step 7.1: 写 8 个测试** (每特征 2 个: 正常 + 边界)
- [ ] **Step 7.2: 实现 4 函数 + registry 注册**
- [ ] **Step 7.3: 跑测试**
- [ ] **Step 7.4: Commit**

```bash
git add src/features/trade_microstructure.py tests/test_features/test_trade_microstructure.py src/features/registry.py
git commit -m "feat(week4): 4 个 trade-microstructure 特征 + family 注册"
```

---

### Task 8: 特征批量构建脚本

**Files:**
- Create: `scripts/build_trade_microstructure_features.py`

**CLI:**

```bash
uv run python scripts/build_trade_microstructure_features.py \
    --start-date 2016-04-17 \
    --end-date 2026-04-17 \
    --output data/features/trade_microstructure_features.parquet
```

**逻辑:**
- 按 (ticker, trading_date) 从 `stock_trades_sampled` 拉 trades
- 计算 4 特征, 写入 parquet (schema: `event_date`, `knowledge_time`, `ticker`, 4 features)
- knowledge_time = 当日 16:15 ET (所有 trades 披露完成 + 15min buffer)
- 带 `--resume` (检查 parquet 已有 rows 跳过)

- [ ] **Step 8.1: Smoke test**
- [ ] **Step 8.2: 实现**
- [ ] **Step 8.3: Commit**

```bash
git commit -m "feat(week4): trade microstructure features batch builder"
```

---

### Task 9: Week 4 Gate Verification

**Files:**
- Create: `scripts/run_week4_gate_verification.py`
- Test: `tests/test_scripts/test_week4_gate.py`

**Gate 三项:**

1. **Coverage Gate**: `trades_sampling_state.completed / total >= 95%` (不含 skipped_holiday)
2. **Feature Quality Gate**: 4 特征每个 missing rate < 30%, outlier rate < 5% (跨 top200 × 10y)
3. **Horizon IC Feasibility**: 对 5D horizon, 4 特征 spearman(|IC|) > 0.015 至少 2 个通过 (弱窗口诊断够用标准)

**产出:**
- `data/reports/week4/gate_summary.json`
- Console: PASS / FAIL per-gate, 如 FAIL 给出详细原因

- [ ] **Step 9.1: 测试** (smoke + fail case)
- [ ] **Step 9.2: 实现**
- [ ] **Step 9.3: 跑全量 (等 Task 6/8 backfill 完成后)**
- [ ] **Step 9.4: Commit**

```bash
git commit -m "feat(week4): Week 4 Gate 验证脚本 (coverage/quality/IC)"
```

---

### Task 10: 端到端跑通 + PR

- [ ] **Step 10.1: 起 supervisor 跑 `scripts/run_trades_sampling.py`**

参考 `/tmp/backfill_supervisor.sh` pattern, 包装 retry + state reset.

- [ ] **Step 10.2: 等 `trades_sampling_state` 完成率 >= 95%**

监控命令:
```bash
uv run python -c "
from src.data.db.session import get_session_factory
from sqlalchemy import text
with get_session_factory()() as s:
    rows = s.execute(text('''
        select status, count(*) from trades_sampling_state group by status
    ''')).all()
    for r in rows: print(r)
"
```

- [ ] **Step 10.3: 跑 `scripts/build_trade_microstructure_features.py` 全量**

- [ ] **Step 10.4: 跑 `scripts/run_week4_gate_verification.py`**

- [ ] **Step 10.5: PR & code review**

```bash
gh pr create --base main --title "feat(week4): Polygon trades 定向抽样 + 4 特征" \
    --body "..."
```

**Codex code-review 必审清单:**
- PIT 纪律 (knowledge_time 计算)
- 阈值硬编码是否合理 (size_threshold=10000, gap=3%)
- rate limit / 断点续跑可靠性
- tick-rule 实现正确性 (尤其 up/down/zero-tick 边界)
- 特征 registry family 注册是否冲突
- data_lineage.yaml 更新

- [ ] **Step 10.6: Merge after review PASS**

```bash
gh pr merge --merge --delete-branch
```

- [ ] **Step 10.7: 更新 IMPLEMENTATION_PLAN.md**

```bash
# 标 Week 4 DONE, 记录 bonus/统计, 类似 Week 3 pattern
git commit -m "chore: Plan Week 4 完成"
```

---

## Gate (Week 4 准出)

- [x] `stock_trades_sampled` 表有数据, 覆盖 top200 + 事件窗口
- [x] 4 特征计算完成, 写入 parquet
- [x] Gate report 三项 PASS
- [x] PR merged to main
- [x] IMPLEMENTATION_PLAN.md 更新

---

## 风险 & 回滚

**主要风险:**
1. Polygon REST `/v3/trades` 速率限制 (Massive tier: 未明确文档限速, 经验值 100 req/s) — 若实际 throttle 严重, fallback 到 flat files + 后处理
2. 单日 trades 体量大 (top200 每日可能 ~10M trades), 批量入库 IO 瓶颈 — 用 COPY + 预分区
3. tick-rule 实现 bug → imbalance 特征分布偏 → 在 Task 7 用公开文献数据校验 (e.g., 某 FAANG 某日的 imbalance 应接近 0)
4. 事件日历 PIT 不严 (earnings 公告时刻 vs disclose 时刻) → Task 3 必须用 `earnings_estimates.knowledge_time`

**回滚路径:**
- migration down 保留 → `alembic downgrade 006`
- 特征 registry 不注册即不入 preprocessing pipeline, 零侵入
- Feature flag `ENABLE_TRADE_MICROSTRUCTURE_FEATURES=false` (在 `src/config.py`) — 默认 OFF

---

## 预估工作量

| Task | 工时 | 备注 |
|---|---|---|
| 1-3 | 0.5 天 | schema / universe / calendar |
| 4-5 | 0.5 天 | Polygon client + plan builder |
| 6 | 0.5 天 + backfill 等待 | executor, 挂 supervisor |
| 7-8 | 0.5 天 | 特征 + batch |
| 9-10 | 0.5 天 | Gate + PR |
| **合计** | **~2-3 天** | 比 Week 3 小一半 (定向抽样, 不全量) |
