# Week 4: Massive Trades 定向抽样 Implementation Plan (v3.2)

> **For agentic workers:** 本 plan 作为任务包交给 Codex 执行. Claude 负责审查/退回/验收, 不写代码.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**版本:** v3.2 (2026-04-21, round 4 passed 7.6 + stable-hash fix)
**v2→v3 修订 (P1 + P2):**
- P1 scope: weak_window 收窄至 `weak_window_top_n=100` (防 180k ticker-day 超预算)
- P1 schema: exchange/sequence_number 改 NOT NULL + deterministic fallback (sentinel -1)
- P2 trf: 加 `trf_id` 字段 + off_exchange 检测用 (exchange_in_TRF OR trf_id NOT NULL OR trf_timestamp NOT NULL)
- P2 IC gate: 用 `abs(t_stat) >= 2` + `sign_consistent_windows >= 7` (不歧视天然负 IC 特征)
- P2 registry gating: register 只写 metadata, FeaturePipeline default excludes trade_microstructure; 加 V5 bundle / default pipeline 回归测试
- P2 estimator: `api_calls = ceil(rows / page_size)` (去首页双计), probe_calls 独立统计

**v3→v3.1 修订 (round 3 cleanup):**
- P1 (round 3): Task 3 event_calendar weak_window 文本与 yaml 对齐 — 显式调 `get_top_liquidity_tickers(top_n=100)`, 不再写"全 universe 命中"
- P2 (round 3): 缺失 `sequence_number` 的 collision 风险 — 加 deterministic hash fallback (trade_id/price/size/participant_ts/conditions → 负值空间)
- P3: 文档版本标签统一 v3.1

**v3.1→v3.2 修订 (round 4):**
- P1 (round 4): Python `hash()` 依赖 `PYTHONHASHSEED` 不 deterministic → 改 `sha256` stable digest (取前 8 字节映射到负 int64). Test 加 "跨进程/重跑 fallback 值稳定" assertion.
- P2 (round 4): migration 注释澄清 — `server_default='-1'` 仅作 DB 最后兜底, 正常 ingestion 不用; client 代码用 stable hash fallback 取负值
- P3 (round 4): H1 标题补 "(v3.2)"

**v1→v2 修订:** 分阶段 scope (pilot → 扩展 gated by estimator); PIT 改用 sip_timestamp + delay; schema 补 4 个时间戳 + exchange + sequence + decimal_size; 特征从 4 增至 5 (补 off_exchange_volume_ratio); 阈值外移 yaml; 修正 `universe_membership` 表名.

**Goal:** 定向拉取 Polygon trades 数据 (先 pilot: earnings/gap/weak-window, 再视 estimator 决定是否扩 top200), 入库 `stock_trades_sampled`, 产出 5 个 trade microstructure 诊断特征, 验证 5D 弱窗口 alpha 可行性.

**Architecture:**
- **阶段化抽样**: Pilot (earnings/gap/weak-window ≈ 10-30k ticker-day) → Stage-2 扩 top200 仅当 preflight estimator 显示预算可控
- **Per-row PIT**: 存 sip/participant/trf 3 个 timestamp, `knowledge_time = sip_timestamp + ENTITLEMENT_DELAY` (Massive=15min 默认, 可配)
- **Split feature knowledge_time**: regular-session 特征 knowledge_time=T 16:15 ET; off-hours 特征 knowledge_time=T 20:15 ET. 避免用 16:15 汇总 20:00 盘后 trade 造成 look-ahead
- 数据源: Polygon REST `/v3/trades/{ticker}` (分页) — trades flat files 量级 (单日 TB) 不在 Week 4 范围
- 特征计算在入库后 batch, Week 4 出 5 个诊断特征 (不入 production preprocessing pipeline, 用 feature flag gate)

**Tech Stack:**
- 数据: Polygon REST `/v3/trades` + 保留 flat files fallback 路径
- 存储: TimescaleDB hypertable `stock_trades_sampled` + state 表 `trades_sampling_state`
- 配置: `configs/research/week4_trades_sampling.yaml` (所有阈值/窗口集中)
- 特征: `src/features/trade_microstructure.py`
- 脚本: 6 个 (preflight / plan / sampling / features / gate / verification)

**关键约束 (硬性):**
- PIT per-row: `knowledge_time = sip_timestamp + entitlement_delay_minutes` (来自 yaml)
- Feature PIT: regular_session_features.knowledge_time=T 16:15 ET; offhours_features.knowledge_time=T 20:15 ET
- 不改动 Week 3 minute pipeline / Week 2.5 universe PIT
- Feature flag `ENABLE_TRADE_MICROSTRUCTURE_FEATURES=false` 默认 OFF
- 必须用 `src.universe.history.get_historical_members(as_of, "SP500")` 读 PIT universe — 不得直接 SQL 查表
- 存储上限: Pilot 阶段 `stock_trades_sampled` < 200 GB (未压缩), 超出 kill-switch 停止

**非 Week 4 范围 (明确延后):**
- Flat files trades 全量 → Phase 2 production (未排期)
- SEC 13F/8-K event window → **Week 5** (需 `sec_ftd` + `fmp_earnings_calendar` 数据到位)
- Trade-level SHAP / 归因 → Week 8+
- Quotes / NBBO 数据 → 独立 spike (不在 v5.1)
- 扩展到全 S&P 500 top200 continuous 10 年历史 → Stage-2, 仅 estimator PASS 后开

---

## 文件结构 (File Structure)

### 新建文件

| 文件 | 职责 |
|---|---|
| `alembic/versions/007_add_stock_trades_sampled.py` | DB migration — trades hypertable + state 表 |
| `configs/research/week4_trades_sampling.yaml` | 所有阈值 / 窗口 / 预算配置 (run hash 写入报告) |
| `src/data/polygon_trades.py` | Polygon `/v3/trades` 客户端 (分页 / 重试 / 429 budget) |
| `src/universe/top_liquidity.py` | Top-N liquidity PIT 包装 (复用 `universe.history.get_historical_members`) |
| `src/data/event_calendar.py` | 事件日历聚合 (earnings / gap / weak-window) |
| `src/features/trade_microstructure.py` | 5 个 trade 特征 + condition filter helper |
| `scripts/preflight_trades_estimator.py` | **新加**: 预飞行估算 calls/pages/rows/storage/wall-time |
| `scripts/build_trades_sample_universe.py` | 生成 (ticker, date, reason) plan parquet + 写 state |
| `scripts/run_trades_sampling.py` | 按 plan 调用 Polygon + 入库 + 断点续跑 |
| `scripts/build_trade_microstructure_features.py` | 从 sampled trades 计算 5 特征 |
| `scripts/run_week4_gate_verification.py` | Gate 报告 (coverage / quality / per-reason IC / t-stat) |
| `tests/test_data/test_polygon_trades.py` | 分页 / 429 / 重试 / 空返回 / 错误 |
| `tests/test_universe/test_top_liquidity.py` | ADV rank + PIT 合规 |
| `tests/test_data/test_event_calendar.py` | 每 reason 聚合 + 去重 |
| `tests/test_features/test_trade_microstructure.py` | 5 特征 + PIT split + condition filter |
| `tests/test_scripts/test_preflight_trades_estimator.py` | Estimator 算术正确性 |
| `tests/test_scripts/test_week4_gate.py` | Gate pass/fail 分支 |
| `tests/test_alembic/test_migration_007.py` | Alembic upgrade/downgrade smoke |

### 修改文件

| 文件 | 修改点 |
|---|---|
| `src/features/registry.py` | 注册 5 个新特征 (category=`trade_microstructure`) |
| `configs/research/data_lineage.yaml` | 添加 5 个 trade features 血统 (split knowledge_time 标记) |
| `src/config.py` | 新增 `ENABLE_TRADE_MICROSTRUCTURE_FEATURES`, `TRADES_MAX_STORAGE_GB` |
| `IMPLEMENTATION_PLAN.md` | Week 4 段落更新进度, 补 Stage-2 尾注 |

### 不得修改

- `src/data/polygon_flat_files.py` / `src/data/polygon_minute.py` / `src/features/intraday.py` — Week 3 已冻结
- `src/universe/builder.py` / `src/universe/history.py` — Week 2.5 P3 修复后冻结

---

## 任务拆分 (Tasks)

### Task 0: Research Config + Preflight Estimator (先做, 门槛任务)

**Files:**
- Create: `configs/research/week4_trades_sampling.yaml`
- Create: `scripts/preflight_trades_estimator.py`
- Create: `tests/test_scripts/test_preflight_trades_estimator.py`

**yaml schema:**

```yaml
# configs/research/week4_trades_sampling.yaml
version: 1
stage: pilot   # pilot | stage2

sampling:
  pilot:
    reasons: [earnings, gap, weak_window]   # 不含 top_liquidity continuous
    earnings_window_days: 3
    gap_threshold_pct: 0.03
    weak_window_top_n: 100      # P1 修订: 每个 weak_window 仅抽 top 100 liquidity, 控总量
    weak_windows:
      - {name: W5, start: "2022-10-01", end: "2023-03-31"}
      - {name: W6, start: "2023-04-01", end: "2023-09-30"}
      - {name: W11, start: "2025-10-01", end: "2026-03-31"}
  stage2:
    top_n_liquidity: 200
    top_liquidity_lookback_days: 20

polygon:
  entitlement_delay_minutes: 15   # SIP trade 披露 → knowledge_time 加成
  rest_max_pages_per_request: 50
  rest_page_size: 50000
  rest_min_interval_seconds: 0.05
  retry_max: 3

budgets:
  max_daily_api_calls: 50000
  max_storage_gb: 200              # kill-switch
  max_rows_per_ticker_day: 2_000_000
  expected_pilot_ticker_days: 30000  # 实际由 estimator 填入

features:
  size_threshold_dollars: 1_000_000      # block trade proxy (替换旧 10k-share)
  size_threshold_min_cap_dollars: 250_000
  # P3 修订: 初始为空 allow-list (= 不过滤, 但在 gate 报告记录分布). Codex 实现时从
  # Polygon condition codes 文档提取 "Regular" / "Regular Way" 对应值填入, 并加 yaml.safe_load test.
  # 参考: https://api.polygon.io/v3/reference/conditions
  condition_allow_list: []
  trf_exchange_codes: [4, 202]           # Polygon exchange codes 中 TRF 系列 (NYSE TRF=4, FINRA ADF=202); Codex 实现时核对
  late_day_window_et: ["15:00", "16:00"]
  offhours_window_et_pre: ["04:00", "09:30"]
  offhours_window_et_post: ["16:00", "20:00"]

gate:
  coverage_min_pct: 95.0
  feature_missing_max_pct: 30.0
  feature_outlier_max_pct: 5.0
  min_passing_features: 2
  ic_threshold: 0.015
  # P2 修订: 用 abs(t_stat) + sign_consistent_windows, 不歧视天然负 IC 特征
  # (tree models 天然会产生负 IC 可用的 short signal; 禁止只对正 IC 放行)
  abs_tstat_threshold: 2.0
  sign_consistent_windows_min: 7   # 11 window 中至少 7 个与 mean_IC 同向
```

**preflight estimator CLI:**

```bash
uv run python scripts/preflight_trades_estimator.py \
    --config configs/research/week4_trades_sampling.yaml \
    --start-date 2016-04-17 \
    --end-date 2026-04-17 \
    --output data/reports/week4/preflight_estimate.json
```

**Estimator 算法 (v3 P2 修订):**
1. 展开 plan ⇒ `(ticker, trading_date)` 总数
2. 用 minute aggs 的 per-day `transactions` 总和作为 trades 行数代理
3. 页数估算 = `ceil(trades / page_size)` (**首页已计入, 不再 +1**)
4. API call 估算 = 页数总和 (即 `sum(ceil(rows_td / page_size))`)
5. **probe_calls 独立统计** (e.g., 预热 metadata / ticker 存在性 check), 单列报告
6. 存储估算 = trades × ~150 bytes (后压缩 ~40 bytes/row)
7. Wall-time 估算 = (api_calls + probe_calls) × rest_min_interval / concurrency
8. 输出: JSON 报告 `{api_calls, probe_calls, rows, storage_gb, wall_time_hours, verdict}` + PASS/FAIL (与 budgets 对比)

- [ ] **Step 0.1:** 写 yaml + test_preflight_trades_estimator.py (6 个 case: 正常 / 超预算 / 数据缺失 / stage2 扩量 / daily_call_budget overflow / **yaml.safe_load roundtrip 验证 所有 key 非占位**)
- [ ] **Step 0.2:** 实现 estimator + config loader (pydantic)
- [ ] **Step 0.3:** 跑 estimator 两次: pilot stage + stage2 stage, 检查 JSON 输出合理
- [ ] **Step 0.4:** Commit

```bash
git add configs/research/week4_trades_sampling.yaml scripts/preflight_trades_estimator.py tests/test_scripts/test_preflight_trades_estimator.py
git commit -m "feat(week4): trades sampling config + preflight estimator"
```

**门槛: pilot estimator 必须 PASS (calls/storage 在预算内) 才允许进 Task 1.** stage2 扩展延后决策.

---

### Task 1: DB Schema — stock_trades_sampled + state 表

**Files:**
- Create: `alembic/versions/007_add_stock_trades_sampled.py`
- Create: `tests/test_alembic/test_migration_007.py`

**关键 schema 修订 (vs v1, 含 v3 P1 PK fix + round3 P2 collision fix):**
- 新增 `sip_timestamp`, `participant_timestamp`, `trf_timestamp` (participant/trf nullable)
- `size` 改 Numeric(18, 4), 新增 `decimal_size` Numeric(18, 8) nullable
- 新增 `sequence_number` BigInt **NOT NULL**, `tape` SmallInt, `correction` SmallInt
- 新增 `trf_id` String(32) nullable (P2 off_exchange 检测用)
- `exchange` SmallInt **NOT NULL** (sentinel=-1 当未知)
- PK: `(ticker, sip_timestamp, exchange, sequence_number)`
- **Round 3/4 防冲突 + 跨进程稳定**: `sequence_number` 缺失时 Codex 入库前生成 deterministic fallback 用 **sha256** (Python 内建 `hash()` 依 `PYTHONHASHSEED` 非稳定, 禁用):

  ```python
  import hashlib
  def stable_sequence_fallback(
      trade_id: str | None,
      price: Decimal,
      size: Decimal,
      participant_timestamp_ns: int | None,
      conditions: tuple[int, ...],
  ) -> int:
      key = "|".join([
          trade_id or "",
          f"{price:.6f}",
          f"{size:.4f}",
          str(participant_timestamp_ns or 0),
          ",".join(str(c) for c in conditions),
      ])
      digest = hashlib.sha256(key.encode("utf-8")).digest()
      int64_from_first_8 = int.from_bytes(digest[:8], "big", signed=False) & ((1 << 62) - 1)
      return -(int64_from_first_8 + 1)   # 负值空间, 保证和真实 >=0 的 sequence_number 不冲突
  ```

  同 ticker+sip_timestamp+exchange 下两条都缺 sequence 的 trades, 只要其他字段 (price/size/ts/conditions) 有差异, fallback 就不同, 不覆盖. 跨进程重跑同一 trade 保证产生相同 PK (idempotent).

**up:**

```python
def upgrade():
    op.create_table(
        "stock_trades_sampled",
        sa.Column("ticker", sa.String(16), nullable=False),
        sa.Column("trading_date", sa.Date, nullable=False),
        sa.Column("sip_timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("participant_timestamp", sa.DateTime(timezone=True), nullable=True),
        sa.Column("trf_timestamp", sa.DateTime(timezone=True), nullable=True),
        sa.Column("knowledge_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("price", sa.Numeric(14, 6), nullable=False),
        sa.Column("size", sa.Numeric(18, 4), nullable=False),
        sa.Column("decimal_size", sa.Numeric(18, 8), nullable=True),
        # P1 修订: exchange / sequence_number 因进 PK, 必须 NOT NULL.
        # exchange: client 缺失时 normalize 到 sentinel -1 (所有未知走同一桶可接受).
        # sequence_number: client 缺失时必须用 stable_sequence_fallback (sha256) 算负值,
        #   不要用 -1. server_default='-1' 只是 DB 最后兜底, 正常 ingestion 不应命中.
        sa.Column("exchange", sa.SmallInteger, nullable=False, server_default="-1"),
        sa.Column("tape", sa.SmallInteger, nullable=True),
        sa.Column("conditions", sa.ARRAY(sa.SmallInteger), nullable=True),
        sa.Column("correction", sa.SmallInteger, nullable=True),
        sa.Column("sequence_number", sa.BigInteger, nullable=False, server_default="-1"),
        sa.Column("trade_id", sa.String(64), nullable=True),
        sa.Column("trf_id", sa.String(32), nullable=True),      # P2: off_exchange 检测
        sa.Column("sampled_reason", sa.String(32), nullable=False),
        sa.PrimaryKeyConstraint("ticker", "sip_timestamp", "exchange", "sequence_number"),
    )
    op.execute(
        "SELECT create_hypertable('stock_trades_sampled', 'sip_timestamp', "
        "chunk_time_interval => INTERVAL '1 day', if_not_exists => TRUE);"
    )
    op.create_index("ix_trades_sampled_knowledge_time", "stock_trades_sampled", ["ticker", "knowledge_time"])
    op.create_index("ix_trades_sampled_trading_date", "stock_trades_sampled", ["ticker", "trading_date"])
    op.execute(
        "ALTER TABLE stock_trades_sampled SET ("
        "timescaledb.compress, "
        "timescaledb.compress_segmentby = 'ticker', "
        "timescaledb.compress_orderby = 'sip_timestamp DESC, sequence_number DESC'"
        ");"
    )
    op.execute("SELECT add_compression_policy('stock_trades_sampled', INTERVAL '7 days', if_not_exists => TRUE);")

    op.create_table(
        "trades_sampling_state",
        sa.Column("ticker", sa.String(16), nullable=False),
        sa.Column("trading_date", sa.Date, nullable=False),
        sa.Column("sampled_reason", sa.String(32), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("rows_ingested", sa.Integer, nullable=True),
        sa.Column("pages_fetched", sa.Integer, nullable=True),
        sa.Column("api_calls_used", sa.Integer, nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_message", sa.Text, nullable=True),
        sa.PrimaryKeyConstraint("ticker", "trading_date", "sampled_reason"),
    )
    op.create_index("ix_trades_sampling_state_status", "trades_sampling_state", ["status", "trading_date"])
```

- [ ] **Step 1.1:** 写 migration + `tests/test_alembic/test_migration_007.py`:
    - upgrade → insert sample rows → downgrade → re-upgrade, 验证 PK 防重复, 跨 venue 不冲突
    - 缺失 exchange → sentinel -1 入库不报错
    - **round3 P2**: 两条同 ticker+sip_timestamp+exchange 都缺 sequence_number, price/size/ts 不同的 trades, 经 `stable_sequence_fallback` 后不冲突, 两条都成功入库
    - **round4 P1**: 同一笔缺失 sequence_number 的 trade 反复调用 `stable_sequence_fallback` **必须返回相同值** (idempotent, 跨进程稳定); 用 subprocess + 自定义 `PYTHONHASHSEED` 验证
- [ ] **Step 1.2:** 跑测试 + alembic upgrade head
- [ ] **Step 1.3:** Commit

```bash
git commit -m "feat(week4): stock_trades_sampled schema (含 sip/participant/trf timestamps)"
```

---

### Task 2: Top-N Liquidity PIT Wrapper

**Files:**
- Create: `src/universe/top_liquidity.py`
- Test: `tests/test_universe/test_top_liquidity.py`

**接口 (修正: 使用 `get_historical_members` 而非直接 SQL universe_membership):**

```python
from src.universe.history import get_historical_members

def get_top_liquidity_tickers(
    as_of_date: date,
    *,
    top_n: int = 200,
    lookback_days: int = 20,
    session_factory: Callable | None = None,
) -> list[str]:
    """返回 as_of_date 前 lookback_days 窗口内 ADV (dollar volume) 排名 top_n 的 ticker.
    步骤:
        1. members = get_historical_members(as_of_date, 'SP500')  # PIT
        2. 从 stock_prices WHERE ticker IN members AND event_time BETWEEN lookback_start AND as_of
           AND knowledge_time <= as_of_eod
        3. ADV = AVG(close * volume) over 窗口
        4. 按 ADV desc 取 top_n
    """
```

- [ ] **Step 2.1:** 5 个测试 (排序 / PIT / membership / 缺失 fallback / `get_historical_members` 集成)
- [ ] **Step 2.2:** 实现
- [ ] **Step 2.3:** Commit

```bash
git commit -m "feat(week4): top-N liquidity PIT wrapper (复用 get_historical_members)"
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
    reason: str  # 'earnings' | 'gap' | 'weak_window' | 'top_liquidity'

def build_sampling_plan(
    *,
    start_date: date,
    end_date: date,
    config: dict,   # 从 yaml load
    session_factory: Callable | None = None,
) -> list[SamplingEvent]:
    """按 config.stage 决定是否包含 top_liquidity.
    同一 (ticker, date) 多 reason 命中时, 产生多条 SamplingEvent (state 表每行独立进度),
    但 Task 6 实际 API 调用对相同 (ticker, date) 只拉 1 次 (去重).
    """
```

**事件源:**
- earnings: `earnings_estimates.event_time` (仅用于 PIT-safe 后 ±earnings_window_days)
- gap: 从 `stock_prices` 计算 `|open - prev_close| / prev_close >= config.sampling.pilot.gap_threshold_pct`
- weak_window (**v3 P1 修订**): 从 yaml 读 (name, start, end), 对每个窗口内每日调 `get_top_liquidity_tickers(as_of_date, top_n=config.sampling.pilot.weak_window_top_n)` (默认 100), **不全 universe**
- top_liquidity: 仅 stage2 — 每日 `get_top_liquidity_tickers(top_n=200)`

- [ ] **Step 3.1:** 7 个测试 (每 reason 一个 + stage2/pilot 切换 + 去重 + **weak_window 每日 emit ≤ 100 ticker**)
- [ ] **Step 3.2:** 实现
- [ ] **Step 3.3:** Commit

```bash
git commit -m "feat(week4): 事件日历聚合 (pilot/stage2 可切换)"
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
        page_size: int | None = None,
        max_pages: int | None = None,
    ) -> Iterator[TradeRecord]:
        """GET /v3/trades/{ticker}?timestamp=YYYY-MM-DD&limit=50000
        分页: 读 response['next_url']
        重试: 5xx/429 指数退避 max 3
        速率: self.min_request_interval
        返回 TradeRecord dataclass (含 sip_timestamp/participant_timestamp/trf_timestamp/conditions/sequence_number 等所有字段)
        """
```

**TradeRecord dataclass:**

```python
@dataclass(frozen=True)
class TradeRecord:
    ticker: str
    trading_date: date
    sip_timestamp: datetime            # UTC, derived from response['sip_timestamp']/1e9
    participant_timestamp: datetime | None
    trf_timestamp: datetime | None
    price: Decimal
    size: Decimal
    decimal_size: Decimal | None
    exchange: int                       # v3 P1: NOT NULL, normalize to -1 if missing
    tape: int | None
    conditions: list[int]
    correction: int | None
    sequence_number: int                # v3 P1: NOT NULL, normalize to -1 if missing
    trade_id: str | None
    trf_id: str | None                  # v3 P2: 新增, off_exchange 检测用
```

- [ ] **Step 4.1:** 6 个测试:
    - 成功路径 + 分页跳转 (3 页)
    - 429 响应 → 指数退避重试 → 最终成功
    - 429 → 3 次全败 → raise DataSourceTransientError
    - 5xx 重试
    - 空响应 (ticker 当日无交易) 返回空 iterator
    - max_pages 截断保护 (警告日志 + 提前终止)
- [ ] **Step 4.2:** 实现 (复用 `src/data/sources/base.py` 的 DataSource)
- [ ] **Step 4.3:** Commit

```bash
git commit -m "feat(week4): Polygon /v3/trades 客户端 (含 sip/participant/trf 时间戳)"
```

---

### Task 5: Sampling Plan Builder Script

**Files:**
- Create: `scripts/build_trades_sample_universe.py`

**CLI:**

```bash
uv run python scripts/build_trades_sample_universe.py \
    --config configs/research/week4_trades_sampling.yaml \
    --start-date 2016-04-17 \
    --end-date 2026-04-17 \
    --output data/reports/week4/trades_sampling_plan.parquet
```

**逻辑:**
1. Load yaml 配置
2. 调 `build_sampling_plan(start, end, config)`
3. 写 parquet + 写 state 表 (status=pending)
4. 打印每 reason (ticker, day) 数量 + 合计
5. **同步调 preflight estimator**, 若超预算则 stderr WARN + exit 1 (强制 user 人工确认或调 yaml)

- [ ] **Step 5.1:** Smoke test
- [ ] **Step 5.2:** 实现
- [ ] **Step 5.3:** Dry-run 2016-04-17 → 2016-04-25
- [ ] **Step 5.4:** Commit

---

### Task 6: Trades 拉取执行器 (含 budget kill-switch)

**Files:**
- Create: `scripts/run_trades_sampling.py`

**关键功能 (在 v1 基础上强化):**
- 读 yaml 的 `budgets.max_daily_api_calls` + `max_storage_gb`
- 每入库 batch 查询当前 `stock_trades_sampled` 总大小 (`pg_total_relation_size`), 超 `max_storage_gb` → kill
- 每日 api_calls 超 `max_daily_api_calls` → 当日停, 记 state
- 每条 trade 入库时同步计算 `knowledge_time = sip_timestamp + entitlement_delay_minutes`
- 条件过滤: 入库时仅 keep `conditions ⊆ config.features.condition_allow_list` 或 allow_list empty (warn)
- 支持 `--dry-run` (只拉取不入库, 用于 API 抽样验证)

**CLI:**

```bash
uv run python scripts/run_trades_sampling.py \
    --plan data/reports/week4/trades_sampling_plan.parquet \
    --config configs/research/week4_trades_sampling.yaml \
    --max-workers 4 \
    --resume
```

- [ ] **Step 6.1:** Smoke tests:
    - 分页截断处理 (max_pages 达上限 → WARN + partial commit)
    - 重复 trade ID 跨 exchange 不冲突 (PK 允许)
    - 429 budget exhausted → gracefully 停
    - storage kill-switch 触发
    - resume 从 pending/failed 重入
- [ ] **Step 6.2:** 实现
- [ ] **Step 6.3:** 本地跑 2025-01-10 × 5 ticker (WMT, AAPL, TSLA, NVDA, SPY)
- [ ] **Step 6.4:** Commit

---

### Task 7: 5 Trade Microstructure 特征 (v1 改 + 新增 1 个)

**Files:**
- Create: `src/features/trade_microstructure.py`
- Test: `tests/test_features/test_trade_microstructure.py`
- Modify: `src/features/registry.py`

**5 特征 (v2):**

```python
def compute_trade_imbalance_proxy(trades: pd.DataFrame, *, condition_allow: set[int]) -> float:
    """Lee-Ready tick-rule, 先按 condition_allow 过滤.
       每 trade 相对前一合格 trade 定符号 (up=+1/down=-1/carry).
       imbalance = sum(sign * size) / sum(size), range [-1, 1]
       Regular session only (09:30-16:00 ET).
    """

def compute_large_trade_ratio(trades: pd.DataFrame, *, size_threshold_dollars: float) -> float:
    """DOLLAR-based (v1 改): trade_dollar = price * size.
       成交额 >= size_threshold_dollars 的 trade 占比 / 全日 dollar volume.
       默认 $1M (block trade proxy); 可由 yaml 调.
       Regular session only.
    """

def compute_late_day_aggressiveness(trades: pd.DataFrame, *, late_day_window_et: tuple[str, str]) -> float:
    """|imbalance(late-day-window)| / |imbalance(full-session)|.
       Late-day window 从 yaml (默认 15:00-16:00 ET).
       Ratio clipped [0, 5]; nan 若 full-session imbalance = 0.
       Regular session only.
    """

def compute_offhours_trade_ratio(
    trades: pd.DataFrame,
    *,
    pre_window_et: tuple[str, str],
    post_window_et: tuple[str, str],
) -> float:
    """(pre-market volume + after-hours volume) / full-day volume.
       knowledge_time = T 20:15 ET (off-hours feature, 不可与 regular-session 特征同 KT).
       Range [0, 1].
    """

def compute_off_exchange_volume_ratio(trades: pd.DataFrame, *, trf_exchange_codes: set[int]) -> float:
    """**新增 (Codex P2 建议, v3 强化识别)**: TRF / off-exchange trades 占全日 dollar volume 比例.

       识别规则 (三者任一):
           1. exchange ∈ trf_exchange_codes (yaml `features.trf_exchange_codes`, e.g. NYSE TRF, FINRA ADF)
           2. trf_id IS NOT NULL (Polygon 显式标记 TRF 报送)
           3. trf_timestamp IS NOT NULL (某些实现只给 timestamp 无 id)

       Regular session only.
       Range [0, 1]; 高值 = 大量 dark pool / off-exchange 活动.
    """
```

**PIT split (强制):**

```python
REGULAR_SESSION_FEATURES = {"trade_imbalance_proxy", "large_trade_ratio",
                             "late_day_aggressiveness", "off_exchange_volume_ratio"}
OFFHOURS_FEATURES = {"offhours_trade_ratio"}

def compute_knowledge_time(trading_date: date, feature_name: str) -> datetime:
    if feature_name in OFFHOURS_FEATURES:
        return datetime.combine(trading_date, time(20, 15), tzinfo=EASTERN).astimezone(timezone.utc)
    return datetime.combine(trading_date, time(16, 15), tzinfo=EASTERN).astimezone(timezone.utc)
```

**registry (v3 P2: 强制 default-off gating):**

```python
# src/features/registry.py (新增 metadata 字典 + for-loop 注册)
_TRADE_MICROSTRUCTURE_FEATURE_METADATA = {
    "trade_imbalance_proxy": "Lee-Ready tick-rule buy/sell imbalance (regular session, condition-filtered).",
    "large_trade_ratio": "Dollar volume share from trades >= $1M (block proxy, regular session).",
    "late_day_aggressiveness": "Late-day (15:00-16:00 ET) |imbalance| ratio vs full session.",
    "offhours_trade_ratio": "Pre/post-market volume share vs full day.",
    "off_exchange_volume_ratio": "TRF / off-exchange volume share (dark pool proxy).",
}
for name, description in _TRADE_MICROSTRUCTURE_FEATURE_METADATA.items():
    self.register(name, "trade_microstructure", description, compute_trade_microstructure_features)
```

**Gating 规则 (v3 P2 新增):**
- registry 只存 **metadata**, `get_feature` 可查, 但不自动进 FeaturePipeline 默认 output
- `FeaturePipeline.run()` / `FeaturePipeline.build_bundle()` 的 default `feature_list` **不含** `trade_microstructure` category
- 启用路径:
  1. 显式传入 `feature_list` 含 trade_microstructure 成员 (研究模式), 或
  2. 设置 `settings.ENABLE_TRADE_MICROSTRUCTURE_FEATURES=true` 开启 family-level 默认注入
- **必须加回归测试**:
  - `test_feature_pipeline_default_unchanged`: default FeaturePipeline output 特征列与 v5 bundle 完全一致 (snapshot compare)
  - `test_v5_bundle_validation_unchanged`: 跑 `scripts/validate_feature_bundle.py` (若存在) 或 `FeaturePipeline.build_bundle(bundle='v5')`, 特征 count/missing-rate 无变化
  - `test_trade_microstructure_opt_in`: ENABLE flag ON 时 default 路径含新特征

- [ ] **Step 7.1:** 14 个测试 (5 特征 × 正常/边界 + 1 condition filter + 1 PIT split + **1 PIT leak 测试** 19:55 trade 不能进 16:15 KT 特征 + **1 registry default-off** ENABLE=false 时 default FeaturePipeline output 未变)
- [ ] **Step 7.2:** 实现 + registry
- [ ] **Step 7.3:** 用公开参考数据 cross-validation (某日 AAPL 的 imbalance 应接近 0, 某 earnings day 的 block ratio 应明显高)
- [ ] **Step 7.4:** Commit

---

### Task 8: 特征批量构建脚本

**Files:**
- Create: `scripts/build_trade_microstructure_features.py`

**CLI:**

```bash
uv run python scripts/build_trade_microstructure_features.py \
    --config configs/research/week4_trades_sampling.yaml \
    --start-date 2016-04-17 \
    --end-date 2026-04-17 \
    --output data/features/trade_microstructure_features.parquet
```

**逻辑:**
- 按 (ticker, trading_date) 从 `stock_trades_sampled` 读 trades (含 condition filter via yaml)
- 分别计算 regular-session (4 个) 和 off-hours (1 个) 特征
- 产出 parquet schema:
  `event_date`, `ticker`, `knowledge_time_regular`, `knowledge_time_offhours`, 5 features
- resume: 检查 parquet 已有 rows 跳过
- 最终 parquet 写入 `(run_config_hash)` 列便于追溯

- [ ] **Step 8.1:** Smoke test
- [ ] **Step 8.2:** 实现
- [ ] **Step 8.3:** Commit

---

### Task 9: Week 4 Gate Verification (升级版)

**Files:**
- Create: `scripts/run_week4_gate_verification.py`
- Test: `tests/test_scripts/test_week4_gate.py`

**Gate 四项 (v1→v2→v3 强化, 按 Codex P2 建议):**

1. **Coverage Gate**: `trades_sampling_state.completed / (total - skipped_holiday) >= config.gate.coverage_min_pct`
2. **Feature Quality Gate**: 每特征 missing_rate <= `feature_missing_max_pct` AND outlier_rate <= `feature_outlier_max_pct`
3. **Data-readiness IC Gate** (非 alpha 准出, 标记为 diagnostic; **v3 P2 改为 sign-aware**):
    - 对 5D horizon 计算 IC per feature per window
    - 对每特征: `mean_IC`, `abs_t_stat`, `positive_windows`, `negative_windows`
    - `sign_consistent_windows` = max(positive_windows, negative_windows) (以 mean_IC 的正负判方向)
    - 通过条件: `|mean_IC| >= ic_threshold` AND `abs_t_stat >= abs_tstat_threshold` AND `sign_consistent_windows >= sign_consistent_windows_min`
    - 要求至少 `min_passing_features=2` 个特征通过 (允许天然负 IC 特征, 即 short-signal)
4. **Per-reason IC breakdown**: 对 (earnings / gap / weak_window) 三类 sample 分别算 IC — 报告 top-1 最强 reason, 指导 Week 5+ 使用

**产出:**
- `data/reports/week4/gate_summary.json`:

```json
{
  "config_hash": "abc123",
  "run_stage": "pilot",
  "gates": {
    "coverage": {"pass": true, "value": 96.3, "threshold": 95.0},
    "feature_quality": {"pass": true, "per_feature": {...}},
    "data_readiness_ic": {
      "pass": true,
      "passing_features": ["trade_imbalance_proxy", "off_exchange_volume_ratio"],
      "details": {
        "trade_imbalance_proxy": {
          "mean_ic": 0.022, "abs_t_stat": 2.8,
          "positive_windows": 8, "negative_windows": 3,
          "sign_consistent_windows": 8, "direction": "positive"
        },
        "late_day_aggressiveness": {
          "mean_ic": -0.018, "abs_t_stat": 2.1,
          "positive_windows": 3, "negative_windows": 8,
          "sign_consistent_windows": 8, "direction": "negative"
        }
      }
    },
    "per_reason_ic": {"earnings": 0.019, "gap": 0.012, "weak_window": 0.021}
  },
  "notes": ["SEC event window deferred to Week 5", "Stage2 top_liquidity 未启用"]
}
```

- Console summary 表格

- [ ] **Step 9.1:** 测试 (每 gate 的 pass/fail 分支 + JSON schema)
- [ ] **Step 9.2:** 实现
- [ ] **Step 9.3:** Commit

---

### Task 10: 端到端 + PR + Merge

- [ ] **Step 10.1:** 起 supervisor 跑 `scripts/run_trades_sampling.py` (pilot stage)

参考 `/tmp/backfill_supervisor.sh` pattern, 包 setsid + auto-restart. 监控 `trades_sampling_state` 完成率.

- [ ] **Step 10.2:** 监控命令

```bash
uv run python -c "
from src.data.db.session import get_session_factory
from sqlalchemy import text
with get_session_factory()() as s:
    rows = s.execute(text('''
        select sampled_reason, status, count(*) from trades_sampling_state
        group by sampled_reason, status order by sampled_reason, status
    ''')).all()
    for r in rows: print(r)
"
```

- [ ] **Step 10.3:** 跑 `scripts/build_trade_microstructure_features.py`
- [ ] **Step 10.4:** 跑 `scripts/run_week4_gate_verification.py` → 若失败, 分析 per-reason IC 定位问题
- [ ] **Step 10.5:** 创建 PR (feature branch `feature/s2-v5.1-week4-trades-sampling`)

**PR 描述清单 (必写):**
- Plan 引用 (docs/superpowers/plans/2026-04-21-w4-trades-targeted-sampling.md)
- Codex review 轮次 + 最终分数
- Stage 状态 (pilot / stage2)
- Gate 报告 JSON 摘要
- SEC event deferred note
- 已知 limitation (未覆盖 top200 continuous)

- [ ] **Step 10.6:** Merge via GitHub PR (不本地 merge)

```bash
gh pr merge --merge --delete-branch
```

- [ ] **Step 10.7:** IMPLEMENTATION_PLAN.md 更新 Week 4 DONE + stats + stage2 尾注

---

## Gate (Week 4 准出)

- [ ] Preflight estimator pilot PASS
- [ ] 所有 7 Task 单元测试 PASS
- [ ] `stock_trades_sampled` 数据 >= 10k ticker-days coverage
- [ ] 5 特征计算完成入 parquet
- [ ] Gate 四项 PASS (data-readiness, 非 alpha adoption)
- [ ] PR merged to main
- [ ] Plan 更新

---

## 风险 & 回滚

**风险 (v1→v2 扩展, 按 Codex P2 建议):**

1. **存储爆炸**: 即使 pilot, 30k ticker-day × 每日 1M trades × 150 bytes ≈ 4.5 TB → kill-switch 必须工作 (`max_storage_gb`)
2. **Page truncation**: 单 ticker-day trades > `max_pages × page_size` → 抽样偏差. 缓解: max_pages 设 50 × 50k=2.5M 覆盖绝大多数; 超限 WARN + 标 state 为 `partial` 不是 `completed`
3. **Duplicate trade ID 跨 venue**: PK 改 (ticker, sip_timestamp, exchange, sequence_number) 后允许
4. **Corrections / late TRF prints**: condition allow-list 过滤 (yaml 配置), 不合格条件 trades 入库时 drop + 在 state 记录 drop count
5. **After-hours PIT leak**: split knowledge_time 严格执行, tests 覆盖 "尾盘 19:55 trade 不进 16:15 KT 特征"
6. **Selection bias IC**: Gate 报告 per-reason 分开算, 明确"这是 data-readiness 不是 alpha adoption"
7. **tick-rule 误差**: condition filter + 公开数据 cross-validation

**回滚:**
- `alembic downgrade 006`
- registry 不注册 = 0 侵入
- Feature flag `ENABLE_TRADE_MICROSTRUCTURE_FEATURES=false` (默认)

---

## 预估工作量 (v3 修订)

| Task | 工时 | 备注 |
|---|---|---|
| 0 | 0.5 天 | config + preflight estimator (门槛) |
| 1 | 0.3 天 | schema 含 sip/participant/trf |
| 2-3 | 0.5 天 | universe + calendar |
| 4 | 0.5 天 | Polygon client |
| 5-6 | 1 天 + backfill 等待 | plan + executor + supervisor |
| 7 | 0.5 天 | 5 特征 + PIT split |
| 8 | 0.3 天 | batch builder |
| 9 | 0.5 天 | gate (4 项升级) |
| 10 | 0.5 天 | PR + review + merge |
| **合计** | **~4 天 + backfill** | v1 预估 2-3 天 (低估), v2 更现实 |

---

## SEC 事件窗延后声明

Week 4 spec 原文含 "earnings / SEC / 大 gap 事件窗口", 本 plan 明确将 **SEC 13F/8-K 窗延至 Week 5** 理由:
- Week 5 计划含 `sec_ftd.py` + `fmp_earnings_calendar.py` — 届时 SEC 数据源到位
- Week 4 用 earnings + gap + weak_window 已能覆盖 data-readiness Gate
- Gate 报告 `notes` 字段明确记录: "SEC event window deferred to Week 5, Week 4 scope 不代表完整事件覆盖"

依赖链: Week 5 完成后 → 回补 SEC event reason → 重跑 Week 4 Gate 第 4 项 (per-reason IC) 覆盖 SEC.
