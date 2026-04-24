# Week 5: 免费公开数据 + FMP 新端点 Implementation Plan

> **For agentic workers:** 本 plan 作为任务包交给 Codex 执行. Claude 负责审查/退回/验收, 不写代码.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**版本:** v1 (2026-04-24)

**Goal:** 补 S1 特征集的 **shorting/crowding** 和 **analyst behavior** 两个空白维度. 目标产出 9-12 个可回测特征, Gate 聚焦**数据可用性**(覆盖 + lag 规则), 不判 alpha 强度 (留 Week 7 screening).

**Architecture:**
- **3 条独立数据源管线**:
  - FINRA Daily Short Sale Volume (免费 public, 日频)
  - SEC FTD (免费 public, 双周)
  - FMP 4 个新端点 (已订, 有 API)
- **每源独立 source adapter** (`src/data/sources/` 或 `src/data/`) + ORM 表 + ingestion
- **9 analyst/shorting features** 注册进 registry, 默认 off 直到 Week 7 per-horizon 激活
- **历史 backfill** 2018-2026 (和 S1 价格范围对齐)

**Tech Stack:**
- FINRA: HTTP scrape `https://cdn.finra.org/equity/regsho/daily/CNMSshvol<YYYYMMDD>.txt` (每日 pipe-delimited CSV)
- SEC FTD: HTTP GET `https://www.sec.gov/data/foiadocsfailsdatahtm` (半月度 ZIP, TXT 格式)
- FMP: 4 个 REST 端点 (历史 grades / price target / ratings / earnings calendar)
- DB: TimescaleDB tables + `knowledge_time` PIT columns

**关键约束 (硬性):**
- PIT: 每条记录 `knowledge_time` = source 实际发布时间 + 合理延迟
  - FINRA: knowledge_time = trade_date + 1 business day (T+1 披露)
  - SEC FTD: knowledge_time = report_settlement_date + 15 calendar days (半月度 lag)
  - FMP grades: knowledge_time = grade_date + 0 (API 实时披露)
- 不得改动 Week 3/4 数据或模型
- 新特征 default-off (不入 V5 bundle, Week 7 screening 决定)
- 所有 backfill 必须 PIT-safe (historical import 不得越时间)

**非 Week 5 范围 (明确延后):**
- FRED 新 macro 数据源 → 已有 (Week 1-2 完成)
- Polygon 其他新端点 → 未涵盖
- Alternative news / sentiment → Week 6+
- 9 个 feature 的 per-horizon IC screening → Week 7

---

## 文件结构 (File Structure)

### 新建文件

| 文件 | 职责 |
|---|---|
| `alembic/versions/008_add_shorting_analyst_tables.py` | DB migration — 3 新表 |
| `src/data/finra_short_sale.py` | FINRA daily short volume HTTP client + ingestion |
| `src/data/sec_ftd.py` | SEC FTD ZIP parser + ingestion |
| `src/data/sources/fmp_grades.py` | FMP historical stock grades 客户端 |
| `src/data/sources/fmp_price_target.py` | FMP price target consensus 客户端 |
| `src/data/sources/fmp_ratings.py` | FMP historical ratings 客户端 |
| `src/data/sources/fmp_earnings_calendar.py` | FMP earnings calendar 客户端 |
| `src/features/shorting.py` | FINRA + FTD 4 特征: `short_sale_ratio_1d/5d` / `short_sale_accel` / `abnormal_off_exchange_shorting` + `ftd_to_float` / `ftd_persistence` / `ftd_shock` |
| `src/features/analyst_proxy.py` | FMP 9 特征: `net_grade_change_5d/20d/60d` / `upgrade_count` / `downgrade_count` / `consensus_upside` / `target_price_drift` / `target_dispersion_proxy` / `coverage_change_proxy` / `financial_health_trend` |
| `scripts/backfill_finra_short_sale.py` | FINRA 历史回填 2018-2026 (~2000 trading days) |
| `scripts/backfill_sec_ftd.py` | SEC FTD 历史回填 (~200 bi-weekly reports) |
| `scripts/backfill_fmp_analyst.py` | FMP 4 端点历史回填 |
| `scripts/run_week5_gate_verification.py` | Week 5 Gate (数据可用性 + lag 验证) |
| Tests (mirror src) | 每 adapter + feature 有单测 |

### 修改文件

| 文件 | 修改点 |
|---|---|
| `src/features/registry.py` | 注册 13 个新 feature (4 shorting + 9 analyst), category `shorting` + `analyst_proxy` |
| `src/features/pipeline.py` | 加新 feature 的 compute 分支 (default-off flag 控制) |
| `src/config/__init__.py` | 新增 `ENABLE_SHORTING_FEATURES` / `ENABLE_ANALYST_PROXY_FEATURES` flags (默认 False) |
| `configs/research/data_lineage.yaml` | 添加 13 个新特征血统记录 |
| `IMPLEMENTATION_PLAN.md` | Week 5 段落标进度 |

---

## 任务拆分 (Tasks)

### Task 0: DB Schema — 3 新表

**Files:**
- Create: `alembic/versions/008_add_shorting_analyst_tables.py`
- Test: `tests/test_alembic/test_migration_008.py`

**Schema 规格:**

```python
# alembic/versions/008_add_shorting_analyst_tables.py

def upgrade():
    # 1. FINRA daily short volume
    op.create_table(
        "short_sale_volume_daily",
        sa.Column("ticker", sa.String(16), nullable=False),
        sa.Column("trade_date", sa.Date, nullable=False),
        sa.Column("knowledge_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("short_volume", sa.BigInteger, nullable=False),  # 做空量
        sa.Column("short_exempt_volume", sa.BigInteger, nullable=True),
        sa.Column("total_volume", sa.BigInteger, nullable=False),
        sa.Column("market", sa.String(16), nullable=True),  # e.g., 'CNMS'
        sa.PrimaryKeyConstraint("ticker", "trade_date", "market"),
    )
    op.create_index("ix_short_sale_kt", "short_sale_volume_daily", ["ticker", "knowledge_time"])

    # 2. SEC FTD (fails to deliver)
    op.create_table(
        "ftd_pit",
        sa.Column("ticker", sa.String(16), nullable=False),
        sa.Column("settlement_date", sa.Date, nullable=False),  # T+2 settle
        sa.Column("knowledge_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("fail_shares", sa.BigInteger, nullable=False),
        sa.Column("close_price", sa.Numeric(12, 4), nullable=True),
        sa.Column("cusip", sa.String(16), nullable=True),
        sa.PrimaryKeyConstraint("ticker", "settlement_date"),
    )
    op.create_index("ix_ftd_kt", "ftd_pit", ["ticker", "knowledge_time"])

    # 3. FMP analyst data (统一 table 存 grades / ratings / targets events)
    op.create_table(
        "analyst_events",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("ticker", sa.String(16), nullable=False),
        sa.Column("event_date", sa.Date, nullable=False),
        sa.Column("knowledge_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("event_type", sa.String(32), nullable=False),  # 'grade' | 'rating' | 'price_target'
        sa.Column("analyst_firm", sa.String(128), nullable=True),
        sa.Column("prior_value", sa.String(64), nullable=True),  # e.g., 'Buy' / '$150'
        sa.Column("current_value", sa.String(64), nullable=True),
        sa.Column("grade_change", sa.SmallInteger, nullable=True),  # +1 upgrade, -1 downgrade, 0 reiterate
        sa.Column("target_price", sa.Numeric(12, 4), nullable=True),
        sa.Column("raw_payload", sa.JSON, nullable=True),
        sa.UniqueConstraint("ticker", "event_date", "analyst_firm", "event_type", name="uq_analyst_event"),
    )
    op.create_index("ix_analyst_kt", "analyst_events", ["ticker", "knowledge_time"])
    op.create_index("ix_analyst_event_type", "analyst_events", ["event_type", "event_date"])

    # 4. FMP earnings_calendar (补足已有 earnings_estimates 的公布日历)
    op.create_table(
        "earnings_calendar",
        sa.Column("ticker", sa.String(16), nullable=False),
        sa.Column("announce_date", sa.Date, nullable=False),
        sa.Column("knowledge_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("timing", sa.String(16), nullable=True),  # 'BMO' / 'AMC' / 'TNS'
        sa.Column("eps_estimate", sa.Numeric(12, 4), nullable=True),
        sa.Column("eps_actual", sa.Numeric(12, 4), nullable=True),
        sa.Column("revenue_estimate", sa.BigInteger, nullable=True),
        sa.Column("revenue_actual", sa.BigInteger, nullable=True),
        sa.PrimaryKeyConstraint("ticker", "announce_date"),
    )
    op.create_index("ix_earnings_cal_kt", "earnings_calendar", ["ticker", "knowledge_time"])
```

- [ ] **Step 0.1:** 写 migration + `tests/test_alembic/test_migration_008.py`:
    - upgrade → insert sample rows → verify PK + index → downgrade → re-upgrade
    - 4 tables create/drop 都 clean
- [ ] **Step 0.2:** `alembic upgrade head` 跑通
- [ ] **Step 0.3:** Commit

```bash
git commit -m "feat(week5): DB schema — short_sale_volume_daily + ftd_pit + analyst_events + earnings_calendar"
```

---

### Task 1: FINRA Short Sale Data Source

**Files:**
- Create: `src/data/finra_short_sale.py`
- Create: `scripts/backfill_finra_short_sale.py`
- Test: `tests/test_data/test_finra_short_sale.py`

**接口 (`src/data/finra_short_sale.py`):**

```python
class FINRAShortSaleSource(DataSource):
    """FINRA Daily Short Sale Volume.

    Data: pipe-delimited text file per day, ~3-5 MB each.
    URL: https://cdn.finra.org/equity/regsho/daily/CNMSshvol{YYYYMMDD}.txt
    Columns: Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market

    Markets (files per day):
    - CNMSshvol (Nasdaq NMS)
    - ADFshvol (Alternative Display Facility)
    - BNYshvol (NYSE)
    - Combined = CNMS + ADF + BNY for a more complete picture

    PIT: knowledge_time = trade_date + 1 business day (T+1 披露).
    """
    source_name = "finra_short_sale"

    def fetch_day(self, trade_date: date, market: str = "CNMSshvol") -> pd.DataFrame:
        """Download 1 day's short volume file, return parsed DataFrame.
        Returns: ticker, trade_date, short_volume, short_exempt_volume, total_volume, market
        """

    def fetch_historical(
        self,
        start_date: date,
        end_date: date,
        markets: list[str] = ["CNMSshvol", "ADFshvol", "BNYshvol"],
        session_factory: Callable | None = None,
    ) -> int:
        """Backfill [start, end] range. Iterates XNYS sessions, downloads + upserts.
        Returns rows inserted. Uses ON CONFLICT DO UPDATE on (ticker, trade_date, market).
        """
```

- [ ] **Step 1.1:** 写 6 个 tests (mock HTTP responses):
    - 成功解析 1 day pipe-delimited file → 50 ticker rows
    - 404 response (weekend / holiday) → empty frame, no error
    - malformed line → skip + warning log
    - rate-limit 429 retry
    - 缺字段 (旧格式) → fallback
    - incremental 调用只抓 1 天增量
- [ ] **Step 1.2:** 实现 client + DB upsert
- [ ] **Step 1.3:** CLI script `scripts/backfill_finra_short_sale.py`:

    ```bash
    uv run python scripts/backfill_finra_short_sale.py \
        --start-date 2018-01-01 \
        --end-date 2026-04-23 \
        --markets CNMS,ADF,BNY
    ```

- [ ] **Step 1.4:** 本地 smoke test: 1 week range, 3 markets, 确认 DB 插入
- [ ] **Step 1.5:** Commit

```bash
git commit -m "feat(week5): FINRA daily short sale volume source + backfill"
```

---

### Task 2: SEC FTD Data Source

**Files:**
- Create: `src/data/sec_ftd.py`
- Create: `scripts/backfill_sec_ftd.py`
- Test: `tests/test_data/test_sec_ftd.py`

**接口:**

```python
class SECFTDSource(DataSource):
    """SEC Fails to Deliver data.

    Data: ZIP files, one per half-month period, containing tab-delimited TXT.
    URL: https://www.sec.gov/data/foiadocsfailsdatahtm (directory)
    Files: fails-deliver-{YYYY}{MM}{half}.zip where half is 'a' (1st half) or 'b' (2nd)

    Columns in TXT:
    - SETTLEMENT DATE | CUSIP | SYMBOL | QUANTITY (FAILS) | DESCRIPTION | PRICE

    PIT: knowledge_time = end_of_period + 15 calendar days (SEC publishes ~15 days after)
    """
    source_name = "sec_ftd"

    def fetch_period(self, year: int, month: int, half: Literal["a", "b"]) -> pd.DataFrame:
        """Download + extract + parse 1 half-month FTD file.
        Returns: ticker, settlement_date, fail_shares, close_price, cusip.
        """

    def fetch_historical(
        self,
        start_year: int = 2018,
        end_year: int | None = None,  # default: current year
        session_factory: Callable | None = None,
    ) -> int:
        """Backfill all half-month periods in [start_year, end_year].
        Upsert to ftd_pit using ON CONFLICT on (ticker, settlement_date).
        """
```

**Edge cases:**
- File name convention varies (pre-2023 vs post-2023)
- Some tickers have multiple rows per settlement_date (aggregate sum of fail_shares)
- Non-equity securities (bonds, options) 需要 filter — only keep equities

- [ ] **Step 2.1:** 5 tests:
    - 解析 sample ZIP (included as test fixture, 几十行)
    - 文件名 fallback (YYYY-MM-{a,b} pattern)
    - 非股票 filter (比如 CUSIP 格式判定)
    - duplicate rows aggregate sum
    - 404 missing file (proposed publish but not available) → skip + log
- [ ] **Step 2.2:** 实现 + CLI
- [ ] **Step 2.3:** Smoke: 1 period (e.g., 2024-01-a), 确认 DB 插入
- [ ] **Step 2.4:** Commit

```bash
git commit -m "feat(week5): SEC FTD source + backfill"
```

---

### Task 3: FMP 4 新端点 (grades / ratings / price_target / earnings_calendar)

**Files:**
- Create: `src/data/sources/fmp_grades.py`
- Create: `src/data/sources/fmp_price_target.py`
- Create: `src/data/sources/fmp_ratings.py`
- Create: `src/data/sources/fmp_earnings_calendar.py`
- Create: `scripts/backfill_fmp_analyst.py` (unified backfill for 4 endpoints)
- Test: `tests/test_data/test_fmp_grades.py`, `test_fmp_price_target.py`, `test_fmp_ratings.py`, `test_fmp_earnings_calendar.py`

**接口 (4 files follow same pattern as existing `src/data/sources/fmp_analyst.py`):**

```python
# fmp_grades.py — historical stock grades (Buy/Hold/Sell change events)
class FMPGradesSource(DataSource):
    """FMP /stable/grades/{ticker} — analyst grade events per ticker."""
    def fetch_ticker_history(self, ticker: str) -> pd.DataFrame:
        # Returns: ticker, event_date, analyst_firm, grade (Buy/Hold/Sell), action
    def fetch_historical(self, tickers: Sequence[str]) -> int:
        # Insert rows to analyst_events with event_type='grade'
        # Compute grade_change: +1 upgrade, -1 downgrade, 0 reiterate
```

```python
# fmp_price_target.py — price target summary + consensus
class FMPPriceTargetSource(DataSource):
    """FMP /stable/price-target-summary/{ticker} + /stable/price-target-consensus/{ticker}."""
    def fetch_ticker(self, ticker: str) -> pd.DataFrame:
        # Returns: ticker, event_date, consensus_target, target_high, target_low, num_analysts
```

```python
# fmp_ratings.py — ratings-historical (FMP's 1-5 composite score)
class FMPRatingsSource(DataSource):
    """FMP /stable/historical-ratings-bulk or /stable/ratings-historical/{ticker}."""
    def fetch_ticker(self, ticker: str) -> pd.DataFrame:
        # Returns: ticker, event_date, rating (1-5 score), rating_recommendation
```

```python
# fmp_earnings_calendar.py — earnings dates upcoming + historical
class FMPEarningsCalendarSource(DataSource):
    """FMP /stable/earnings-calendar."""
    def fetch_range(self, start: date, end: date) -> pd.DataFrame:
        # Returns: ticker, announce_date, timing, eps_estimate, eps_actual, revenue_*
```

**统一 CLI (`scripts/backfill_fmp_analyst.py`):**

```bash
uv run python scripts/backfill_fmp_analyst.py \
    --tickers-file data/universe/sp500_tickers.txt \
    --endpoints grades,price_target,ratings,earnings_calendar \
    --start-date 2018-01-01 \
    --end-date 2026-04-23
```

- [ ] **Step 3.1:** 4 × 4 = 16 个 tests (每 adapter 4 个: 成功 / 空返 / rate-limit 429 / FMP 字段变更)
- [ ] **Step 3.2:** 4 adapters 实现
- [ ] **Step 3.3:** 统一 backfill CLI
- [ ] **Step 3.4:** Smoke: 1 ticker (AAPL) 全 4 端点跑通
- [ ] **Step 3.5:** Commit

```bash
git commit -m "feat(week5): FMP 4 新端点 (grades/ratings/price_target/earnings_calendar) + unified backfill"
```

---

### Task 4: Shorting 特征 (4 个)

**Files:**
- Create: `src/features/shorting.py`
- Test: `tests/test_features/test_shorting.py`

**特征 (PIT-safe, 用 `knowledge_time <= as_of` 查 DB):**

```python
def compute_short_sale_ratio_1d(ticker: str, as_of: date, session_factory=None) -> float:
    """短卖占当日总量比例 (1 day)."""
    # SELECT short_volume / total_volume FROM short_sale_volume_daily
    # WHERE ticker=:t AND trade_date = :as_of AND knowledge_time <= :as_of_eod
    # 合并 CNMS + ADF + BNY 三市场数据
    # range [0, 1]

def compute_short_sale_ratio_5d(ticker: str, as_of: date, session_factory=None) -> float:
    """5-day rolling 短卖比例平均."""

def compute_short_sale_accel(ticker: str, as_of: date, session_factory=None) -> float:
    """做空加速度: 5d MA - 20d MA."""

def compute_abnormal_off_exchange_shorting(ticker: str, as_of: date, session_factory=None) -> float:
    """场外做空异常: ADFshvol / CNMSshvol 比例的 z-score vs 90d 基线."""

def compute_ftd_to_float(ticker: str, as_of: date, session_factory=None) -> float:
    """FTD 股数 / 流通股数 (需 shares_outstanding 从 fundamentals_pit)."""

def compute_ftd_persistence(ticker: str, as_of: date, session_factory=None) -> int:
    """连续 FTD 记录的 settlement_date 数量."""

def compute_ftd_shock(ticker: str, as_of: date, session_factory=None) -> float:
    """最近 FTD 股数 / 90d 平均 - 1 (类似 % spike)."""
```

- [ ] **Step 4.1:** 14 tests (每特征 2 个: 正常 + 边界/缺失):
    - Mock DB with seed data, assert per-feature value correctness
    - Missing data 应返 NaN (not raise)
    - PIT 严格 (knowledge_time > as_of 数据不进入)
- [ ] **Step 4.2:** 实现 7 个特征函数
- [ ] **Step 4.3:** Commit

```bash
git commit -m "feat(week5): shorting 特征 (7 个) — FINRA short_sale + SEC FTD"
```

---

### Task 5: Analyst Proxy 特征 (9 个)

**Files:**
- Create: `src/features/analyst_proxy.py`
- Test: `tests/test_features/test_analyst_proxy.py`

**特征:**

```python
def compute_net_grade_change(ticker, as_of, horizon_days, session_factory=None) -> int:
    """net grade change = upgrade_count - downgrade_count over [as_of-horizon, as_of]."""
    # horizon_days in {5, 20, 60} → 3 flavors

def compute_upgrade_count(ticker, as_of, horizon_days=20, session_factory=None) -> int:
    """20-day 内 upgrade 数."""

def compute_downgrade_count(ticker, as_of, horizon_days=20, session_factory=None) -> int:

def compute_consensus_upside(ticker, as_of, session_factory=None) -> float:
    """(consensus_target - close) / close."""

def compute_target_price_drift(ticker, as_of, horizon_days=60, session_factory=None) -> float:
    """60-day linear regression slope of target price (normalized by current price)."""

def compute_target_dispersion_proxy(ticker, as_of, session_factory=None) -> float:
    """(target_high - target_low) / target_consensus. 高 = 分析师分歧."""

def compute_coverage_change_proxy(ticker, as_of, horizon_days=60, session_factory=None) -> int:
    """60-day 内 num_analysts 变化."""

def compute_financial_health_trend(ticker, as_of, horizon_days=60, session_factory=None) -> float:
    """60d FMP rating 分数趋势 (current - 60d ago)."""
```

- [ ] **Step 5.1:** 18 tests (每特征 2 个: 正常 + 缺失)
- [ ] **Step 5.2:** 实现 9 函数
- [ ] **Step 5.3:** Commit

```bash
git commit -m "feat(week5): analyst_proxy 特征 (9 个) — FMP grades/ratings/price_target"
```

---

### Task 6: Feature Registry 注册 + config flags

**Files:**
- Modify: `src/features/registry.py`
- Modify: `src/config/__init__.py`
- Modify: `configs/research/data_lineage.yaml`
- Test: `tests/test_features/test_engineering.py` (既有 test 新增 registry count 更新)

**Registry 修改:**

```python
# src/features/registry.py (加 2 个 metadata dict + 2 for-loops)

_SHORTING_FEATURE_METADATA = {
    "short_sale_ratio_1d": "FINRA 1-day short sale ratio (short/total volume).",
    "short_sale_ratio_5d": "FINRA 5-day rolling average short sale ratio.",
    "short_sale_accel": "Short sale 5d MA - 20d MA (acceleration).",
    "abnormal_off_exchange_shorting": "ADF/CNMS short volume z-score vs 90d baseline.",
    "ftd_to_float": "SEC FTD shares / shares_outstanding.",
    "ftd_persistence": "Consecutive FTD settlement_date count.",
    "ftd_shock": "FTD shares / 90d average - 1.",
}

_ANALYST_PROXY_FEATURE_METADATA = {
    "net_grade_change_5d": "FMP grade upgrades - downgrades over 5 days.",
    "net_grade_change_20d": "Same, 20 days.",
    "net_grade_change_60d": "Same, 60 days.",
    "upgrade_count": "FMP upgrade count over 20 days.",
    "downgrade_count": "FMP downgrade count over 20 days.",
    "consensus_upside": "(FMP consensus target - close) / close.",
    "target_price_drift": "60-day target price regression slope (normalized).",
    "target_dispersion_proxy": "(target_high - target_low) / target_consensus.",
    "coverage_change_proxy": "60-day num_analysts change.",
    "financial_health_trend": "60-day FMP rating score trend.",
}
```

**Config flags 修改:**

```python
# src/config/__init__.py
class Settings(BaseSettings):
    # ... existing ...
    ENABLE_SHORTING_FEATURES: bool = False      # default off, Week 7+ decide
    ENABLE_ANALYST_PROXY_FEATURES: bool = False # default off
```

- [ ] **Step 6.1:** 写 3 个新测试:
    - `test_shorting_registry_default_off`: `ENABLE_SHORTING_FEATURES=False` 时 default feature set 不含 shorting
    - `test_analyst_proxy_registry_default_off`: 同上 for analyst
    - `test_feature_registry_count_has_new_features`: 既有 test 补 13 个新 features (147 → 160)
- [ ] **Step 6.2:** Registry + config + lineage.yaml 更新
- [ ] **Step 6.3:** 跑 pytest 验证全部 green
- [ ] **Step 6.4:** Commit

```bash
git commit -m "feat(week5): 注册 13 个新特征 (4 shorting + 9 analyst_proxy) default-off"
```

---

### Task 7: Week 5 Gate Verification (数据质量)

**Files:**
- Create: `scripts/run_week5_gate_verification.py`
- Test: `tests/test_scripts/test_week5_gate.py`

**Gate 3 项**:

1. **Coverage**: 3 个数据源每月 >= 90% SP500 tickers 有记录 (2018 起)
2. **Missing rate per feature**: 13 特征 missing rate <= 40%
3. **Lag rule clear**: 每特征知识时间 (`knowledge_time`) 严格 >= event_time

**产出**: `data/reports/week5/gate_summary.json`

- [ ] **Step 7.1:** 3 测试 (每 gate 的 pass/fail 分支)
- [ ] **Step 7.2:** 实现
- [ ] **Step 7.3:** Commit

```bash
git commit -m "feat(week5): Week 5 Gate verification (coverage + missing + lag)"
```

---

### Task 8: 历史 Backfill + 端到端 Gate

- [ ] **Step 8.1:** 起 FINRA backfill (~30 min for 2018-2026, 2000 business days × 3 markets)
- [ ] **Step 8.2:** 起 SEC FTD backfill (~200 半月度 periods × 2 年 = ~400 files, ~1h)
- [ ] **Step 8.3:** 起 FMP 4 endpoint backfill (SP500 × 4 endpoints = 2000 API calls, ~30 min)
- [ ] **Step 8.4:** 跑 `scripts/run_week5_gate_verification.py` → 验证 3 gate PASS
- [ ] **Step 8.5:** 若任一 Gate FAIL, 诊断后补数据, 重跑 Gate
- [ ] **Step 8.6:** Commit Gate 报告

```bash
git add data/reports/week5/
git commit -m "feat(week5): 历史 backfill 完成 + Gate PASS"
```

---

### Task 9: PR + Merge

- [ ] **Step 9.1:** PR 描述模板 (引用 plan + Codex review 分数 + Gate 报告)
- [ ] **Step 9.2:** `gh pr create --base main --head feature/s2-v5.1-week5-shorting-analyst`
- [ ] **Step 9.3:** 派 Codex 做 final code-review
- [ ] **Step 9.4:** 处理 findings 或直接 merge (overall >= 7 + 无 critical)
- [ ] **Step 9.5:** `gh pr merge --merge --delete-branch`
- [ ] **Step 9.6:** 更新 `IMPLEMENTATION_PLAN.md` Week 5 标 DONE

---

## Gate (Week 5 准出)

- [ ] 4 新表 migration clean up/down
- [ ] 3 数据源 backfill 完成 (2018-2026)
- [ ] 13 新特征函数 + 单测 pass
- [ ] Registry 注册 + default-off 生效
- [ ] Gate report JSON 三项 PASS
- [ ] PR merged to main
- [ ] `IMPLEMENTATION_PLAN.md` Week 5 更新

---

## 风险 & 回滚

**主要风险:**

1. **FINRA 文件格式历史变更**: Pre-2023 可能字段不一致, parser 要 defensive
2. **SEC FTD 大 ZIP 解压时间**: 单 ZIP 可能 200 MB, 内存 peak 注意 (streaming parse)
3. **FMP rate limit**: 免费层 300 req/min, Starter 750, 全 SP500 × 4 endpoints 可能 1 小时
4. **Non-equity CUSIP filter (SEC FTD)**: 需要识别去除债券/期权, 否则 ticker 列会混入 non-stock 标识
5. **PIT 违规风险**: FMP API 返回可能含"最新"数据而非 event-time 原值, 需**显式校准 knowledge_time**

**缓解:**
- 每 adapter 的 parser defensive, 字段缺失 → None + warning
- SEC FTD 解压到 `/tmp`, 单文件处理完删
- FMP 限速: 每源 `min_request_interval` 200 ms
- CUSIP 过滤规则: 前 6 char + 固定 pattern check against equities
- 每表 `knowledge_time` 测试: assert per-row `knowledge_time >= event/settlement_date + T+k delay`

**回滚:**
- `alembic downgrade -4` (drop 4 tables)
- Feature functions 不注册 = 0 侵入
- 3 config flags default False → 不影响 V5 bundle

---

## 预估工作量

| Task | 工时 | 备注 |
|---|---|---|
| 0 (DB schema) | 0.3 天 | migration + test |
| 1 (FINRA) | 0.5 天 | HTTP + parser + backfill |
| 2 (SEC FTD) | 0.5 天 | ZIP parser + CUSIP filter |
| 3 (FMP × 4) | 1 天 | 4 个 adapter + unified backfill |
| 4 (shorting 特征) | 0.5 天 | 7 特征 + PIT 测 |
| 5 (analyst_proxy 特征) | 0.5 天 | 9 特征 + PIT 测 |
| 6 (registry + flag) | 0.2 天 | 注册 + config |
| 7 (Week 5 Gate) | 0.3 天 | 3 gate + report |
| 8 (backfill) | 0.5 天 + 2h 等 | 大部分自动化 |
| 9 (PR + merge) | 0.5 天 | |
| **合计** | **~4 天** | |

---

## 特别说明

### 和 Week 4 的关系

- **完全正交** (不依赖 trade data). Week 4 Gate FAIL 不影响 Week 5.
- 复用 universe_membership (Week 2.5 + Week 4 bug fix 的干净数据)

### Default-off 理由

- Week 5 只做**数据采集 + 特征定义**, **不决定 alpha adoption**
- alpha adoption 在 Week 7 per-horizon IC screening 决定
- 期间 S1 V5 bundle 保持不变, 避免 dependencies 问题
