# Week 5: 免费公开数据 + FMP 新端点 Implementation Plan

> **For agentic workers:** 本 plan 作为任务包交给 Codex 执行. Claude 负责审查/退回/验收, 不写代码.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**版本:** v3 (2026-04-24, Codex round 2 PASS 7.4 + 3 micro-edits)

**v2→v3 修订 (round 2 micro-fixes):**
- P1 count freeze: **Tranche A = 14 features** (4 shorting + 10 analyst_proxy), **5 tables**. Goal/architecture/file/task/registry/Gate 所有引用对齐
- P1 FMP price_target 端点契约: per-analyst historical 用 legacy `api/v4/price-target`, consensus 用 stable `/stable/price-target-consensus`. 两路径显式文档化 PIT / rate limit / test policy
- P2 FINRA Gate 方向修正: "ETag change rate <= 5%" (update rate, 不是 consistency rate, 避免实现反转)

**v1→v2 修订 (round 1 findings):**
- ✅ P1 Scope 冻结: 分 **Tranche A** + **Tranche B**
- ✅ P1 Compatibility matrix: 新增 adapters 和 existing 100% 正交
- ✅ P2 PIT: FINRA same-day 18:00 ET, SEC 用实际发布日期, FMP query_time 保守 fallback
- ✅ P2 Technical: FMP 改 query-param 端点格式
- ✅ P2 Schema: analyst_events 拆 3 typed tables + earnings_calendar 独立理由文档
- ✅ P2 Priority: Tranche A 优先, B 按需
- ✅ P2 Backfill: engineering days + wall-clock + operator supervision 分层
- ✅ P3 Gate source-specific: FINRA update rate / CUSIP / FMP field drift
- ✅ P3 非显风险: CUSIP 首级指标 + FINRA re-fetch 策略

---

## 🎯 Goal & 冻结 Scope

**Goal:** 补 S1 特征集的 **shorting/crowding** 和 **analyst behavior** 两个空白维度. 分两阶段:

- **Tranche A (must-have, 本 Week 5 核心)**:
  - 数据源: FINRA Daily Short Sale + FMP 4 新端点 (grades / ratings / price_target / earnings_calendar)
  - **14 个新特征** (4 shorting + 10 analyst_proxy)
  - **5 个新表** (short_sale_volume_daily + grades_events + ratings_events + price_target_events + earnings_calendar)
  - Gate: 数据可用性 (coverage + missing + lag + source-specific integrity)

- **Tranche B (optional, 按 Tranche A 进度决定)**:
  - 数据源: SEC FTD (bi-weekly ZIP + CUSIP 映射)
  - **3 个新特征** (`ftd_to_float`, `ftd_persistence`, `ftd_shock`)
  - 1 个新表 (`ftd_pit`)
  - 单独触发条件: Tranche A 完成并 Gate PASS 后, 若剩余时间 >= 1 天, 补做

> **强制**: 本 plan 完成 Tranche A 即 Week 5 DONE. Tranche B 作为 Week 5.5 或 Week 6 前奏, 独立 PR.

**Architecture:**
- 每源独立 source adapter (`src/data/sources/`) + 独立 ORM 表 (typed columns, 不用 string-typed 多用途表)
- **与 existing 100% 正交**: 已有 `fmp_analyst.py` (EPS/revenue estimates) + `polygon_short_interest.py` (bi-weekly short interest) 不改动, 新 adapter 解决不同业务问题
- **13 个特征 default-off**: 注册但不入 V5 bundle, Week 7 screening 决定

**Tech Stack:**
- FINRA: HTTP GET `https://cdn.finra.org/equity/regsho/daily/{MarketPrefix}{YYYYMMDD}.txt`
- FMP: REST `/stable/{endpoint}?symbol={ticker}` query-param style (per FMP stable docs)
- DB: PostgreSQL/TimescaleDB, typed ORM, `knowledge_time` 每条记录

---

## 🔗 Compatibility Matrix (解 Codex High-2 finding)

### 已存在 (不改, 不重复)

| 既有文件 | 既有表 | 既有功能 | 和 Week 5 关系 |
|---|---|---|---|
| `src/data/sources/fmp_analyst.py` | `analyst_estimates` | EPS / revenue **consensus estimates** (forward-looking), `compute_earnings_revision` 特征 | **正交**. Week 5 的 `fmp_grades` / `fmp_ratings` / `fmp_price_target` 是不同端点/不同业务含义 |
| `src/data/sources/fmp_earnings.py` | `earnings_estimates` | EPS 历史估计 + `compute_earnings_surprise` 特征 | **正交**. Week 5 的 `earnings_calendar` 关注**announce date + timing (BMO/AMC)**, 不含 estimate, 两表 field 互补 |
| `src/data/sources/polygon_short_interest.py` | `short_interest` | **Bi-weekly** short interest (from Polygon), `compute_short_interest_features` (6 features) | **正交**. Week 5 的 FINRA 是**daily short sale volume** (不同频率不同源), 两源互补 |
| `src/data/sources/fmp_insider.py` | — | Insider trading | 不动 |
| `src/data/sources/fmp_sec_filings.py` | — | SEC filing events | **可能小幅重叠** Tranche B (SEC FTD 走独立 URL, 不依赖此) |
| `src/features/alternative.py` | — | 已有 earnings / analyst / short_interest / insider 特征 | Week 5 新增特征都在 NEW 模块 (`shorting.py`, `analyst_proxy.py`), 不污染已有 |

### 新增 (Tranche A)

| 新文件 | 新表 | 新业务 |
|---|---|---|
| `src/data/finra_short_sale.py` | `short_sale_volume_daily` | FINRA daily **short volume** (每日做空量, 含 3 市场: CNMS/ADF/BNY) |
| `src/data/sources/fmp_grades.py` | `grades_events` | Analyst **grade changes** (Buy → Hold 事件) |
| `src/data/sources/fmp_ratings.py` | `ratings_events` | FMP 1-5 **composite rating score** |
| `src/data/sources/fmp_price_target.py` | `price_target_events` | Analyst **price target** + consensus + dispersion |
| `src/data/sources/fmp_earnings_calendar.py` | `earnings_calendar` | Earnings **announce date + timing** (BMO/AMC/TNS) |
| `src/features/shorting.py` | — | 4 新特征, Query `short_sale_volume_daily` + `short_interest` (既有) |
| `src/features/analyst_proxy.py` | — | 9 新特征, Query `grades_events` + `ratings_events` + `price_target_events` |

### 新增 (Tranche B)

| 新文件 | 新表 | 新业务 |
|---|---|---|
| `src/data/sec_ftd.py` | `ftd_pit` | SEC FTD (fails to deliver), 需 CUSIP 映射 |
| `src/features/shorting.py` 扩展 | — | 3 FTD features 追加 |

### 明确**不新建**的 (以防误解)

- 不建 `analyst_events` 单一大表 (per Codex P2-5: typed tables 更清晰, 拆成 grades_events / ratings_events / price_target_events)
- 不重复 polygon_short_interest 的表 (bi-weekly short interest 独立保留)
- 不修改已有 `analyst_estimates` / `earnings_estimates` 表

---

## 📋 File Structure (冻结版)

### Tranche A 新建文件 (13 个)

| 文件 | 职责 |
|---|---|
| `alembic/versions/008_add_shorting_analyst_tables_A.py` | **Tranche A** 表: short_sale_volume_daily + grades_events + ratings_events + price_target_events + earnings_calendar (**5 表**) |
| `src/data/finra_short_sale.py` | FINRA client + ORM + ingestion |
| `src/data/sources/fmp_grades.py` | FMP grades adapter |
| `src/data/sources/fmp_ratings.py` | FMP ratings adapter |
| `src/data/sources/fmp_price_target.py` | FMP price target adapter |
| `src/data/sources/fmp_earnings_calendar.py` | FMP earnings calendar adapter |
| `src/features/shorting.py` | 4 shorting 特征 (Tranche A, FTD 特征 Tranche B 扩展) |
| `src/features/analyst_proxy.py` | 9 analyst 特征 |
| `scripts/backfill_finra_short_sale.py` | FINRA 历史回填 |
| `scripts/backfill_fmp_analyst.py` | FMP 4 端点统一回填 |
| `scripts/run_week5_gate_verification.py` | Gate (coverage + missing + lag + source-integrity) |
| Tests (每 adapter + feature 1 个测试文件) | ~12 新 test 文件 |

### Tranche B 新建文件 (按需)

| 文件 | 职责 |
|---|---|
| `alembic/versions/009_add_ftd_pit.py` | FTD 表 |
| `src/data/sec_ftd.py` | SEC FTD client + CUSIP 映射 |
| `scripts/backfill_sec_ftd.py` | FTD 回填 |
| `src/features/shorting.py` 扩展 | 3 FTD 特征 |
| Tests | 2 新 test 文件 |

### Modify files

| 文件 | 修改点 |
|---|---|
| `src/features/registry.py` | 注册 14 (+3 Tranche B) metadata |
| `src/config/__init__.py` | `ENABLE_SHORTING_FEATURES`, `ENABLE_ANALYST_PROXY_FEATURES` flags default False |
| `configs/research/data_lineage.yaml` | 14 (+3) 特征血统 |
| `IMPLEMENTATION_PLAN.md` | Week 5 段落进度标记 |

---

## 🕐 PIT Rules (Codex P2-3 修订)

| 数据源 | event timestamp | knowledge_time 规则 | 理由 |
|---|---|---|---|
| FINRA short sale | `trade_date` (当日) | `trade_date` 18:00 ET | **FINRA 披露规则**: 同一交易日 ~6PM ET 发布当日 short volume |
| SEC FTD (Tranche B) | `settlement_date` (T+2 settle) | **SEC FTD index page 上该文件的 publication_date** (从 SEC 公告页抓) | 实际日期, 不用 +15 固定值 heuristic |
| FMP grades | `date` (grade change day) | `date` 23:59 ET (**同日收盘后**, API 实时披露不代表首次可见) | Conservative fallback: 确保同日 intra-day 不见 leak |
| FMP ratings | `date` | `date` 23:59 ET | 同上 |
| FMP price_target | `published_date` (each analyst submission) | `published_date` 23:59 ET | 同上 |
| FMP earnings_calendar | `date` (announcement) | `timing` 匹配: BMO = `date-1` 23:59 ET, AMC/TNS = `date` 23:59 ET | timing 驱动真实披露时间 |

**实施**: 每 adapter 在写入 DB 前用 `_compute_knowledge_time(record)` 固定 KT. 不使用 `datetime.now()` 作 fallback (否则 test 失效 + 实际 rerun 会变动).

---

## 🗃️ Schema (Typed Tables, Codex P2-5 修订)

**原 v1 问题**: 单 `analyst_events` 表用 string-typed `prior_value`/`current_value` 放 grades + ratings + price_targets 杂项. Codex 指出弱 typing 难维护.

**v2 改**: 拆 3 个 typed tables.

### Tranche A 5 表

```python
# 1. FINRA daily short volume
op.create_table(
    "short_sale_volume_daily",
    sa.Column("ticker", sa.String(16), nullable=False),
    sa.Column("trade_date", sa.Date, nullable=False),
    sa.Column("knowledge_time", sa.DateTime(timezone=True), nullable=False),
    sa.Column("market", sa.String(16), nullable=False),  # CNMS / ADF / BNY
    sa.Column("short_volume", sa.BigInteger, nullable=False),
    sa.Column("short_exempt_volume", sa.BigInteger, nullable=True),
    sa.Column("total_volume", sa.BigInteger, nullable=False),
    sa.Column("file_etag", sa.String(64), nullable=True),  # 用于 FINRA re-fetch 检测 file update
    sa.PrimaryKeyConstraint("ticker", "trade_date", "market"),
)
op.create_index("ix_short_sale_kt", "short_sale_volume_daily", ["ticker", "knowledge_time"])

# 2. FMP grades (analyst rating events, e.g., "Goldman: Buy → Hold")
op.create_table(
    "grades_events",
    sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
    sa.Column("ticker", sa.String(16), nullable=False),
    sa.Column("event_date", sa.Date, nullable=False),
    sa.Column("knowledge_time", sa.DateTime(timezone=True), nullable=False),
    sa.Column("analyst_firm", sa.String(128), nullable=False),
    sa.Column("prior_grade", sa.String(16), nullable=True),  # 'Buy' / 'Hold' / 'Sell' (typed enum in Python)
    sa.Column("new_grade", sa.String(16), nullable=False),
    sa.Column("action", sa.String(16), nullable=False),  # 'upgrade' / 'downgrade' / 'reiterate' / 'initiate'
    sa.Column("grade_score_change", sa.SmallInteger, nullable=False),  # +1 / -1 / 0 (typed)
    sa.UniqueConstraint("ticker", "event_date", "analyst_firm", name="uq_grade_event"),
)
op.create_index("ix_grades_kt", "grades_events", ["ticker", "knowledge_time"])

# 3. FMP ratings (1-5 composite score)
op.create_table(
    "ratings_events",
    sa.Column("ticker", sa.String(16), nullable=False),
    sa.Column("event_date", sa.Date, nullable=False),
    sa.Column("knowledge_time", sa.DateTime(timezone=True), nullable=False),
    sa.Column("rating_score", sa.SmallInteger, nullable=False),  # 1-5
    sa.Column("rating_recommendation", sa.String(32), nullable=True),  # 'Strong Buy' / 'Hold' etc
    sa.Column("dcf_rating", sa.Numeric(6, 2), nullable=True),
    sa.Column("pe_rating", sa.Numeric(6, 2), nullable=True),
    sa.Column("roe_rating", sa.Numeric(6, 2), nullable=True),
    sa.PrimaryKeyConstraint("ticker", "event_date"),
)
op.create_index("ix_ratings_kt", "ratings_events", ["ticker", "knowledge_time"])

# 4. FMP price target (analyst target events)
op.create_table(
    "price_target_events",
    sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
    sa.Column("ticker", sa.String(16), nullable=False),
    sa.Column("event_date", sa.Date, nullable=False),
    sa.Column("knowledge_time", sa.DateTime(timezone=True), nullable=False),
    sa.Column("analyst_firm", sa.String(128), nullable=True),
    sa.Column("target_price", sa.Numeric(12, 4), nullable=False),
    sa.Column("prior_target", sa.Numeric(12, 4), nullable=True),
    sa.Column("price_when_published", sa.Numeric(12, 4), nullable=True),
    sa.Column("is_consensus", sa.Boolean, nullable=False, server_default="false"),  # 区分 per-analyst vs consensus summary
    sa.UniqueConstraint("ticker", "event_date", "analyst_firm", name="uq_target_event"),
)
op.create_index("ix_target_kt", "price_target_events", ["ticker", "knowledge_time"])

# 5. FMP earnings calendar (announce date + timing, distinct from earnings_estimates)
op.create_table(
    "earnings_calendar",
    sa.Column("ticker", sa.String(16), nullable=False),
    sa.Column("announce_date", sa.Date, nullable=False),
    sa.Column("knowledge_time", sa.DateTime(timezone=True), nullable=False),
    sa.Column("timing", sa.String(16), nullable=True),  # 'BMO' / 'AMC' / 'TNS'
    sa.Column("fiscal_period_end", sa.Date, nullable=True),  # 关联 earnings_estimates 的 fiscal_date
    sa.Column("eps_estimate", sa.Numeric(12, 4), nullable=True),
    sa.Column("eps_actual", sa.Numeric(12, 4), nullable=True),
    sa.Column("revenue_estimate", sa.BigInteger, nullable=True),
    sa.Column("revenue_actual", sa.BigInteger, nullable=True),
    sa.PrimaryKeyConstraint("ticker", "announce_date"),
)
op.create_index("ix_earnings_cal_kt", "earnings_calendar", ["ticker", "knowledge_time"])
```

**为什么 earnings_calendar 独立于 earnings_estimates?**

- `earnings_estimates` (既有) = **fiscal_date** centric, 多分析师 EPS/revenue estimate 快照
- `earnings_calendar` (新) = **announce_date** centric, 含 **timing** (BMO/AMC) 真实披露时刻
- 两表通过 `(ticker, fiscal_period_end)` 可关联, 但**各自 PIT 语义不同**
- Codex P2-5 的 overlap 担忧通过明确 join condition + 不同 PK 解决

### Tranche B 1 表

```python
# ftd_pit (Tranche B, CUSIP-centric, 需 CUSIP→ticker 映射)
op.create_table(
    "ftd_pit",
    sa.Column("cusip", sa.String(16), nullable=False),
    sa.Column("ticker", sa.String(16), nullable=True),  # nullable: CUSIP mapping 可能 fail
    sa.Column("settlement_date", sa.Date, nullable=False),
    sa.Column("knowledge_time", sa.DateTime(timezone=True), nullable=False),
    sa.Column("publication_date", sa.Date, nullable=False),  # SEC 实际发布日 (索引页抓)
    sa.Column("fail_shares", sa.BigInteger, nullable=False),
    sa.Column("close_price", sa.Numeric(12, 4), nullable=True),
    sa.Column("is_equity", sa.Boolean, nullable=False, server_default="true"),  # filter non-equity
    sa.PrimaryKeyConstraint("cusip", "settlement_date"),
)
op.create_index("ix_ftd_ticker_kt", "ftd_pit", ["ticker", "knowledge_time"])
op.create_index("ix_ftd_cusip_kt", "ftd_pit", ["cusip", "knowledge_time"])
```

---

## 📦 任务拆分 (Tasks — 10 Tranche A + 3 Tranche B)

### Tranche A

#### Task 0: DB Schema — Tranche A 5 表

- [ ] **Step 0.1:** 写 migration `alembic/versions/008_add_shorting_analyst_tables_A.py`
- [ ] **Step 0.2:** Test `tests/test_alembic/test_migration_008.py`:
    - upgrade → seed sample rows → verify PK + index → downgrade → re-upgrade
    - 5 tables 都 clean
- [ ] **Step 0.3:** `alembic upgrade head`
- [ ] **Step 0.4:** Commit `feat(week5): Tranche A schema (5 tables)`

---

#### Task 1: FINRA Short Sale Source + backfill

**Files:** `src/data/finra_short_sale.py`, `scripts/backfill_finra_short_sale.py`, tests

**API 细节:**

- URL pattern: `https://cdn.finra.org/equity/regsho/daily/{Prefix}shvol{YYYYMMDD}.txt`
- Prefixes: `CNMS` (Nasdaq NMS), `ADF` (Alternative Display Facility), `BNY` (NYSE)
- Format: pipe-delimited `Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market`
- PIT: knowledge_time = `trade_date` 18:00 ET (FINRA 同日披露)
- Re-fetch policy: 每次下载记录 HTTP `ETag` header, 下次若 ETag 变 → re-parse + upsert (detect updates)

**接口:**

```python
class FINRAShortSaleSource(DataSource):
    source_name = "finra_short_sale"

    def fetch_day(self, trade_date: date, market: Literal["CNMS", "ADF", "BNY"]) -> tuple[pd.DataFrame, str | None]:
        """Download 1 day × 1 market file. Returns (frame, etag)."""

    def fetch_historical(
        self,
        start_date: date,
        end_date: date,
        markets: list[str] = ["CNMS", "ADF", "BNY"],
        session_factory: Callable | None = None,
        force_refetch: bool = False,
    ) -> int:
        """Iterate XNYS sessions. Skip if ETag matches cached (unless force_refetch).
        Upsert to short_sale_volume_daily via ON CONFLICT UPDATE.
        Returns rows inserted/updated.
        """
```

**Tests (6)**:
- 解析 sample pipe-delimited file (fixture)
- 404 response (weekend/holiday) → empty frame, no error
- malformed line → skip + warning
- 429 rate-limit → exponential retry success
- ETag update detection → re-parse
- Missing `ShortExemptVolume` (pre-2014 format) → fallback to None

- [ ] **Step 1.1:** 6 tests
- [ ] **Step 1.2:** Client + ORM upsert
- [ ] **Step 1.3:** Backfill CLI `scripts/backfill_finra_short_sale.py`
- [ ] **Step 1.4:** Smoke: 1 week × 3 markets → verify DB rows
- [ ] **Step 1.5:** Commit `feat(week5): FINRA daily short sale volume source + backfill`

---

#### Task 2: FMP 4 新端点 Adapters

**Files:** `src/data/sources/fmp_{grades,ratings,price_target,earnings_calendar}.py`, tests

**FMP 端点 (Codex P2-4 修订: query-param style per FMP stable docs):**

```python
# fmp_grades.py
class FMPGradesSource(DataSource):
    def fetch_ticker(self, ticker: str) -> pd.DataFrame:
        # GET https://financialmodelingprep.com/stable/grades?symbol={ticker}&apikey=...
        # Returns: [{date, newGrade, previousGrade, gradingCompany, action}]
        # Map to grades_events schema

    def fetch_historical(self, tickers: Sequence[str]) -> int:
        # Bulk loop, insert rows, compute grade_score_change (+1/-1/0)
```

```python
# fmp_ratings.py
class FMPRatingsSource(DataSource):
    def fetch_ticker(self, ticker: str) -> pd.DataFrame:
        # GET /stable/ratings-historical?symbol={ticker}&apikey=...
        # Returns: [{date, rating, ratingScore, ratingRecommendation, dcfRating, peRating, roeRating}]
```

```python
# fmp_price_target.py
# 注意: FMP 有 2 个 price_target 端点, 一新一老, Tranche A 都用:
#
# 1. 新 stable: GET /stable/price-target-consensus?symbol={ticker}
#    - 返 consensus snapshot (current consensus target, high, low, num_analysts)
#    - 用于 consensus_upside 特征 (需当前 consensus)
#
# 2. Legacy v4: GET /api/v4/price-target?symbol={ticker}
#    - 返 per-analyst 历史 target events
#    - 用于 target_price_drift / target_dispersion_proxy / coverage_change_proxy 特征
#    - 这是 legacy endpoint (docs 2024 仍列出), PIT: published_date 23:59 ET
#    - Rate limit: 同 /stable (共享 FMP 账户配额)
#    - Test policy: snapshot sample + 字段变更 warn, 若 end-of-life 兜底到 /stable/price-target-summary
#
# 两个端点都在 Tranche A, 缺一不可. fetch_ticker 返 union frame (is_consensus bool).

class FMPPriceTargetSource(DataSource):
    stable_url = "/stable/price-target-consensus"
    legacy_v4_url = "/api/v4/price-target"  # legacy, documented 2024

    def fetch_ticker(self, ticker: str) -> pd.DataFrame:
        # 1. Fetch consensus snapshot from /stable/price-target-consensus
        # 2. Fetch per-analyst history from /api/v4/price-target
        # 3. Merge into union frame:
        #    - per-analyst rows: is_consensus=False, target_price, analyst_firm, event_date
        #    - consensus row: is_consensus=True, target_price (= consensus), analyst_firm=NULL
        # 4. Each row gets price_target_events schema alignment
```

**Legacy v4 风险缓解**: 若某时 FMP 下线 v4, 特征 `target_price_drift`/`target_dispersion_proxy`/`coverage_change_proxy` 降级 — 自动回退到 consensus summary 的 target_high-target_low dispersion (有 test 覆盖这路径).

```python
# fmp_earnings_calendar.py
class FMPEarningsCalendarSource(DataSource):
    def fetch_range(self, start: date, end: date) -> pd.DataFrame:
        # GET /stable/earnings-calendar?from={start}&to={end}&apikey=...
        # 可能需要 page (>365 天 split)
```

**统一 backfill CLI:**

```bash
uv run python scripts/backfill_fmp_analyst.py \
    --tickers-file data/universe/sp500_tickers.txt \
    --endpoints grades,ratings,price_target,earnings_calendar \
    --start-date 2018-01-01 \
    --end-date 2026-04-23
```

**Tests:** 4 × 4 = 16 (每 adapter 4 个)

- [ ] **Step 2.1:** 16 tests
- [ ] **Step 2.2:** 4 adapters 实现 (query-param style)
- [ ] **Step 2.3:** 统一 backfill CLI
- [ ] **Step 2.4:** Smoke: AAPL × 4 端点 → DB 有记录
- [ ] **Step 2.5:** Commit `feat(week5): FMP 4 新端点 (grades/ratings/price_target/earnings_calendar)`

---

#### Task 3: Shorting 特征 (4 Tranche A)

**Files:** `src/features/shorting.py`, `tests/test_features/test_shorting.py`

```python
# 4 Tranche A features (FINRA-based)

def compute_short_sale_ratio_1d(ticker: str, as_of: date, session_factory=None) -> float:
    """短卖占当日总量 (合 CNMS + ADF + BNY). PIT: knowledge_time <= as_of."""

def compute_short_sale_ratio_5d(ticker: str, as_of: date, session_factory=None) -> float:
    """5-day rolling 平均."""

def compute_short_sale_accel(ticker: str, as_of: date, session_factory=None) -> float:
    """5d MA - 20d MA."""

def compute_abnormal_off_exchange_shorting(ticker: str, as_of: date, session_factory=None) -> float:
    """(ADF_short_ratio) z-score vs 90d 基线.
    ADF = Alternative Display Facility (dark pool / internalizer), 高比例 = 机构暗中做空.
    """
```

- [ ] **Step 3.1:** 8 tests (每特征 2 个: 正常 + 边界/缺失)
- [ ] **Step 3.2:** 实现
- [ ] **Step 3.3:** Commit `feat(week5): shorting 4 特征 (FINRA-based)`

---

#### Task 4: Analyst Proxy 特征 (9)

**Files:** `src/features/analyst_proxy.py`, tests

```python
def compute_net_grade_change(ticker, as_of, horizon_days, session_factory=None) -> int:
    """grade_score_change sum over [as_of-horizon, as_of], horizon in {5, 20, 60}."""

def compute_upgrade_count(ticker, as_of, horizon_days=20, session_factory=None) -> int:
def compute_downgrade_count(ticker, as_of, horizon_days=20, session_factory=None) -> int:

def compute_consensus_upside(ticker, as_of, session_factory=None) -> float:
    """(consensus_target - close) / close.
    consensus_target = 最近 60d price_target_events where is_consensus=True 的 target_price.
    """

def compute_target_price_drift(ticker, as_of, horizon_days=60, session_factory=None) -> float:
    """60d linear regression slope of target price (归一化 by current price)."""

def compute_target_dispersion_proxy(ticker, as_of, session_factory=None) -> float:
    """近 60d per-analyst targets 的 std/mean. 高 = 分歧."""

def compute_coverage_change_proxy(ticker, as_of, horizon_days=60, session_factory=None) -> int:
    """distinct analyst_firm count 变化."""

def compute_financial_health_trend(ticker, as_of, horizon_days=60, session_factory=None) -> float:
    """ratings_events.rating_score 60d 趋势 (current - 60d ago)."""
```

- [ ] **Step 4.1:** 18 tests (每特征 2 个)
- [ ] **Step 4.2:** 实现 9 函数
- [ ] **Step 4.3:** Commit `feat(week5): analyst_proxy 9 特征`

---

#### Task 5: Registry + Config flags

**Files:** `src/features/registry.py`, `src/config/__init__.py`, `configs/research/data_lineage.yaml`

```python
# src/features/registry.py
_SHORTING_FEATURE_METADATA = {
    "short_sale_ratio_1d": "FINRA 1-day short sale ratio (short/total volume).",
    "short_sale_ratio_5d": "FINRA 5-day rolling average.",
    "short_sale_accel": "5d MA - 20d MA short sale ratio.",
    "abnormal_off_exchange_shorting": "ADF short ratio z-score vs 90d baseline.",
}
_ANALYST_PROXY_FEATURE_METADATA = {
    "net_grade_change_5d": "Upgrade count - downgrade count, 5 days.",
    "net_grade_change_20d": "Same, 20 days.",
    "net_grade_change_60d": "Same, 60 days.",
    "upgrade_count": "FMP upgrade events, 20 days.",
    "downgrade_count": "FMP downgrade events, 20 days.",
    "consensus_upside": "(Consensus target - close) / close.",
    "target_price_drift": "60d target price regression slope (normalized).",
    "target_dispersion_proxy": "60d target price std/mean (analyst disagreement).",
    "coverage_change_proxy": "60d num_analysts count change.",
    "financial_health_trend": "60d FMP rating score trend.",
}

# src/config/__init__.py
class Settings(BaseSettings):
    # ...
    ENABLE_SHORTING_FEATURES: bool = False
    ENABLE_ANALYST_PROXY_FEATURES: bool = False
```

- [ ] **Step 5.1:** Tests (3):
    - `test_shorting_registry_default_off`: ENABLE_SHORTING_FEATURES=False → default pipeline 不含
    - `test_analyst_proxy_registry_default_off`: 同
    - `test_feature_registry_count_updated`: 既有 count + 13 (147 → 160)
- [ ] **Step 5.2:** registry + config + data_lineage 更新
- [ ] **Step 5.3:** Commit

---

#### Task 6: Week 5 Gate Verification (Tranche A)

**Files:** `scripts/run_week5_gate_verification.py`, tests

**Gate 规则 (Codex P3-8 修订, source-specific):**

1. **Coverage Gate** (每月 >= 90% SP500 tickers 有记录):
   - FINRA: 每月期望 ~20 trading days × 3 markets 都有数据
   - FMP grades/ratings/price_target: 每 ticker 至少 1 条/季度
   - earnings_calendar: 每 ticker 每季度至少 1 条

2. **Missing Rate Gate** (每特征 missing < 40%)

3. **Lag Rule Gate** (每特征 `knowledge_time >= event/settlement_date`)

4. **Source Integrity Gate** (NEW, Codex P3-8 + round 2 方向校正):
   - FINRA: **历史文件 ETag change rate <= 5%** (re-fetch 时发现 ETag 变化的比例 = file update rate. 健康状态 < 5%, 若 > 5% 说明 FINRA 在改历史 → 警报 + 批量 re-parse)
   - FMP: 每端点 field-population 成功率 >= 95% (低成功率说明 API 格式变化)
   - 整体: adapter HTTP 4xx/5xx 错误率 < 1%

**产出**: `data/reports/week5/gate_summary.json`

- [ ] **Step 6.1:** 4 Gate 分别的 test (pass/fail 分支)
- [ ] **Step 6.2:** 实现
- [ ] **Step 6.3:** Commit

---

#### Task 7: Tranche A 历史 Backfill

**Separate estimates (Codex P2-7 修订):**

| 阶段 | Engineering days | Wall-clock | Operator supervision |
|---|---|---|---|
| Task 0-6 (code + test) | 3 天 | — | 主动 |
| FINRA backfill 2018-2026 | — | 1-2h (supervisor 自动) | 背景监控 |
| FMP 4 endpoint backfill SP500 | — | 1-2h (rate-limited) | 背景监控 |
| Gate 验证 + 调整 | 0.5 天 | — | 主动 |
| **Tranche A 合计** | **~3.5 天** | **~3h wall-clock** | **~4h 主动** |

- [ ] **Step 7.1:** 起 FINRA backfill (supervisor setsid 后台)
- [ ] **Step 7.2:** 起 FMP backfill (supervisor 后台)
- [ ] **Step 7.3:** 等 backfill 完成 (wakeup check)
- [ ] **Step 7.4:** 跑 Gate verification → 若 FAIL 诊断补数据
- [ ] **Step 7.5:** Commit Gate 报告

---

#### Task 8: Tranche A PR + Merge

- [ ] **Step 8.1:** PR body 模板 (引用 plan + Codex review 分数 + Gate 报告 + Tranche B 后续说明)
- [ ] **Step 8.2:** `gh pr create --base main --head feature/s2-v5.1-week5-shorting-analyst`
- [ ] **Step 8.3:** Codex final code-review
- [ ] **Step 8.4:** Merge `gh pr merge --merge --delete-branch`
- [ ] **Step 8.5:** 更新 `IMPLEMENTATION_PLAN.md` Week 5 Tranche A DONE

---

### Tranche B (可选, 独立 PR)

#### Task B1: SEC FTD Source + Tranche B schema

**Files:** `alembic/versions/009_add_ftd_pit.py`, `src/data/sec_ftd.py`, `scripts/backfill_sec_ftd.py`, tests

**CUSIP 映射策略 (Codex P3-9 首级指标):**

- 使用已有 `stocks` 表 (或新建 `cusip_to_ticker_map`) 做 CUSIP → ticker lookup
- **Gate**: 未匹配 CUSIP 比例 < 10% 整体, 对 equity-filtered CUSIP 未匹配 < 2%
- 未匹配 CUSIP **保留原 CUSIP 值 + `ticker=NULL`**, 便于事后补救 (不丢数据)

**SEC 发布日期 (P2-3 修订):**

- 从 https://www.sec.gov/data/foiadocsfailsdatahtm 主页抓 **文件发布日期**
- `publication_date` 列存实际日期
- `knowledge_time = publication_date 18:00 ET` (SEC 通常下午发布)

**Tests (5):** parser + CUSIP map + non-equity filter + dup aggregation + 404 handle

- [ ] **Step B1.1:** 5 tests
- [ ] **Step B1.2:** Migration 009 + client + backfill
- [ ] **Step B1.3:** Smoke (1 period)
- [ ] **Step B1.4:** Commit `feat(week5.B): SEC FTD source + schema + backfill`

#### Task B2: 3 FTD 特征

```python
# src/features/shorting.py 扩展
def compute_ftd_to_float(ticker, as_of, ...) -> float:
    """fail_shares / shares_outstanding (from fundamentals_pit)."""

def compute_ftd_persistence(ticker, as_of, ...) -> int:
    """连续 FTD 出现的 settlement_date 数."""

def compute_ftd_shock(ticker, as_of, ...) -> float:
    """最近 FTD / 90d 平均 - 1."""
```

- [ ] **Step B2.1:** 6 tests
- [ ] **Step B2.2:** 实现
- [ ] **Step B2.3:** Registry 追加 3 + Gate 扩充
- [ ] **Step B2.4:** Commit

#### Task B3: Tranche B PR + Merge

类似 Task 8.

---

## 🎯 Gate (Week 5 Tranche A 准出)

- [ ] 5 表 migration up/down clean
- [ ] FINRA backfill 2018-2026 完成
- [ ] FMP 4 端点 backfill 完成 (SP500)
- [ ] 13 新特征函数 + 单测 pass (共 32 测试 Tranche A)
- [ ] Registry 注册 + default-off 生效
- [ ] 4 Gate (coverage + missing + lag + source-integrity) **全 PASS**
- [ ] PR merged to main
- [ ] `IMPLEMENTATION_PLAN.md` 更新

---

## 风险 & 回滚

**Tranche A 风险:**

1. **FINRA 文件格式**: Pre-2023 历史格式可能不同, parser defensive
2. **FINRA 文件 update**: 历史数据可能被 FINRA 更新 → ETag 检测 + re-fetch policy
3. **FMP rate limit**: Starter tier 750 req/min, SP500 × 4 endpoints × 历史 = 估 2000-5000 calls, 需要分批
4. **FMP API 格式变**: query-param style 是 stable docs, 但仍可能变, test fixture 验证
5. **earnings_calendar timezone**: BMO vs AMC 判定错会泄漏 intraday, PIT 测试必覆盖

**Tranche B 额外风险:**

6. **CUSIP 映射失败**: non-equity / 旧 ticker rename → 加 is_equity filter + 保留未映射行
7. **SEC 文件发布日**: 需从 HTML 索引抓, parser 脆性
8. **SEC ZIP 大小**: 单 ZIP 可能 200 MB, streaming parse

**缓解:** 每条写入 test fixture + defensive parser + ETag re-fetch + CUSIP quality metric in Gate.

**回滚:**
- `alembic downgrade -5` (Tranche A) / `-1` (Tranche B)
- Feature 不注册 = 0 侵入
- 2 config flags default False

---

## 预估工作量

| 阶段 | Engineering days | Wall-clock |
|---|---|---|
| **Tranche A** | **3.5 天** | backfill 3h |
| **Tranche B (optional)** | **1.5 天** | backfill 1h |
| **合计 (A+B)** | **5 天** | 4h |

**Tranche A 可独立 merge, 不依赖 B 完成**. 

---

## 📝 特别说明

### 和 Week 4 的关系
- 完全正交. Week 4 的 trade data / minute-proxy 不影响 Week 5.
- 复用 Week 4 修复的 universe_membership PIT (cross-bug cleanup)

### Default-off 理由
- Week 5 只定义特征 + 采集数据, **不判 alpha adoption**
- Week 7 per-horizon IC screening 才决定入 V5 bundle
- V5 bundle 保持不变, 避免 dependency 问题

### 与 existing 代码的边界 (再次强调)
- `analyst_estimates` (EPS/revenue estimate) 保留不动
- `short_interest` (bi-weekly Polygon) 保留不动
- `earnings_estimates` (历史 EPS actual/estimate) 保留不动
- Week 5 新增的 4 类 FMP 事件 + FINRA 日度 + SEC FTD 都是 net-new 业务, 零改动已有
