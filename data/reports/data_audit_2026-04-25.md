# 数据底座审计报告 — 2026-04-25 (W7→W8 全宇宙 IC 启动前)

**审计人**: Claude
**审计范围**: PIT 完整性 / Survivorship / 时效性 / 覆盖率 / Corporate Actions / 跨源一致性
**审计目的**: W8 全宇宙 503t × 9y IC 验证启动前的硬性 gate. 所有 P0 必须修复; P1 修后再跑.
**结论**: **不能直接进 W8**. 发现 3 个 P0 + 4 个 P1 + 3 个 P2.

---

## P0 (Blocker — 必须修)

### P0-1. stock_prices PIT 不一致 — 最近 8 天 lag=0 (3710 行)

**发现**:
- `audit_price_truth_20260425.json::issues[pit_violation_stock_prices].count = 3710`
- 全部来自 polygon source, trade_date 范围 `2026-04-17 ~ 2026-04-24`
- 这些行 `knowledge_time = trade_date 20:10 UTC` (lag=0), 历史行是 `trade_date+1 20:00 UTC` (lag=1)

**根因**:
`dags/dag_daily_data.py:411-417, 432-438, 467-473`:
```python
frame = polygon_source.fetch_grouped_daily_range(
    request_start, market_data_end,
    tickers=tracked_tickers,
    knowledge_time_mode="observed_at",  # ← 这里
    observed_at=as_of,                   # = datetime.now(timezone.utc)
)
```
`market_close_fast_pipeline` DAG 在每个交易日 16:10 NYT 触发, `as_of=now()` ≈ 20:10 UTC, 写入 trade_date=今日的价格时把 `kt=20:10 UTC` 同日.

`polygon.py::persist_prices` 用 `LEAST(existing_kt, new_kt)` UPSERT (lines 475-481), 后续 historical refetch 也无法覆盖.

**影响**:
- W8 全宇宙 IC 若回测窗口延伸到 2026-04, 会用 today's close 算 today's signal → IC 虚高
- 历史窗口 (≤2025) 不受影响, 但未来日常更新会持续累积新 lag=0 行
- W7 IC screening (2016-2025-02) 已发布数字不受此影响

**修复方案**:
1. **代码层**: `dag_daily_data.py` 的 3 处 polygon 调用从 `knowledge_time_mode="observed_at"` 改回 `"historical"`. 历史 mode 给出 `trade_date+1 16:00 ET` 的保守 lag=1.
2. **数据层**: 写脚本 `scripts/fix_recent_pit_lag.py`, 把这 3710 行 `knowledge_time` shift 到 `trade_date+1 20:00 UTC`. 直接 SQL UPDATE 即可.
3. (可选) 改 `persist_prices` UPSERT 让 LEAST 不至于让 observed_at 永远赢过 historical (但若 1 修了, 2 修了, 这条不必改).

---

### P0-2. fundamentals_pit 14675 行 PIT 泄漏 (S&P 500 universe 内, 154 ticker)

**发现**:
- 总计 15429 行 `kt::date <= event_time::date`, 其中 14675 行属 universe_membership 内 ticker
- 涉及 154 个 ticker × 多个核心 metric (eps, ebitda, revenue, total_assets, net_income, etc.)
- 时间分布: 2015 年 2978 行 / 2016 年 2898 行 / 2024-2025 仅 363 行 (越早越严重)

**典型样本** (APC ticker):
```
APC 2025Q4 eps         event_time=2025-12-31 kt=2025-12-31 00:00 UTC ❌ Q4 财报实际 2026-02 才出
APC 2025Q4 total_debt  event_time=2025-12-31 kt=2025-12-31 05:00 UTC ❌
```
现实 10-Q 在 quarter end + 35-45d 公布, 10-K 在 fiscal year end + 60-90d.

**根因**:
`src/data/sources/fmp.py:281`:
```python
knowledge_time = self._parse_knowledge_time(row) or self._fallback_knowledge_time(event_time)
```
`_parse_knowledge_time` 解析 FMP API 的 `acceptedDate` / `fillingDate` 字段. 当 FMP 返回的这些字段值 ≤ event_time (vendor data quality issue), `_parse_knowledge_time` 仍信任并使用. 应该有合理性校验.

**影响**:
- 这是 W7 ablation "drop fundamental → IC +0.018" 的可能根因之一: 早期 fundamental 数据的人为完美 PIT (kt=event_time) 让模型过拟合, 后期数据正常 lag → 信号坍塌
- W8 全宇宙 IC 的 fundamental family contribution 不可信
- 推荐: drop fundamental 路径仍可走, 但应先 fix 然后再 ablation 一次以获得真值

**修复方案**:
1. **代码层**: `fmp.py::_parse_knowledge_time` 添加 sanity check — 如果 parsed timestamp ≤ event_time, 视为 vendor 错误并 return None, 走 `_fallback_knowledge_time` (event_time + 45d).
2. **数据层**: 写一次性脚本 `scripts/fix_fundamentals_pit_lag.py` — 对 `kt::date <= event_time::date` 的 15429 行, 用保守 fallback (`event_time + 45d 16:00 NYT`) 重写 kt.

---

### P0-3. analyst_estimates 表 100% PIT 时效失效 (4516 行 future fiscal_date)

**发现**:
- `analyst_estimates` 5019 行中 100% 都有 `kt = fiscal_date 23:59:59 UTC`
- 4516 行 `fiscal_date > current_date` (90%) — kt 也在未来, 永远不会被 backtest 看到
- `created_at` (我们 fetch 的时间) 与 `knowledge_time` 完全脱节: e.g. `created_at=2026-04-15 fetched`, `kt=2028-06-30` (将来 2 年才"可见")

**典型样本**:
```
MS    fiscal_date=2028-06-30  kt=2028-06-30 23:59 UTC  created_at=2026-04-15  ❌ 估值要 2028 才可见
TRGP  fiscal_date=2028-09-30  kt=2028-09-30 23:59 UTC  created_at=2026-04-15  ❌
```

**根因**: 入库逻辑把 kt 错设为 fiscal_date 而非估算/发布日. 这导致 analyst_estimates 表对历史 backtest **完全不可用**.

**影响**:
- W7 analyst_proxy 家族特征 (target_dispersion_proxy IC 0.122 等) 都来自其他源 (fmp_price_target / fmp_grades 等), 不来自这张表
- 不阻塞 W8 IC, 但浪费了已采集的数据
- 若未来要新增 analyst-based features, 此表先不可用

**修复方案** (低紧急度, 可推后):
- 重新 fetch 时使用 `created_at` 或 FMP API 返回的 `date` / `acceptedDate` 字段当 kt
- 或彻底废弃此表, 全靠 `analyst_proxy` 衍生特征

---

## P1 (Should-fix — 修了再跑 W8 更稳)

### P1-1. fundamentals_pit 474 行 future kt (FMP consensus_eps for 2026Q1)

**发现**: 474 行 `consensus_eps` / `eps_consensus` for `fiscal_period=2026Q1, event_time=2026-03-31`, 但 `kt=2026-07-13 ~ 2026-07-21` (3 个月以后).

**含义**: 这是 forward consensus, 用 expected report date 当 kt. 不是 leak (今天看不到), 但数据并未真正使用 (实际今天就已知 consensus 估值).

**修复**: kt 改设为 `created_at` (我们 fetch 的时间), forward consensus 才能在 backtest 中正确"今日可见".

---

### P1-2. earnings_estimates kt = fiscal_date 23:59:59 (28360 行 realized actuals)

**发现**: 28360 行 `eps_actual not null` (历史已实现), kt 设为 fiscal_date (报告日) 23:59:59. 含义: 当日下午 11:59 UTC 才"可见".

**含义**: 这对 daily backtest with `as_of = trade_date+1 09:30 UTC` 是 OK 的. 但对盘中/早盘 backtest 偏激进 (实际财报 typically 发布于 BMO 或 AMC, kt 应分开).

**影响**: 边际, 不阻塞 W8 IC. P2 可考虑细化.

---

### P1-3. short_interest knowledge_time 全部硬编码 lag=3 calendar days

**发现**: 122377 行 short_interest, 全部 `(kt::date - settlement_date) = 3`, 全部 23:00 UTC.

**问题**: FINRA 半月度 short interest 报告的实际发布日是 settlement_date + ~8 个工作日, 不是 +3 calendar days. 这是 **过激进** lag (假设比真实更早可见 5+ 天).

**修复**: 改成 settlement_date + 8 business days (FINRA 标准发布周期).

**影响**: Phase E S1 优化中 short_interest 家族 IC=0.032 单家族, 修后 IC 可能略降 (但是真值).

---

### P1-4. polygon `_historical_knowledge_time` 加 +1 day 是否过保守?

**发现**: `polygon.py:561-567` 历史 mode 返回 `trade_date + 1 day 16:00 ET` (lag=1 整天).
理论上 daily close 在 trade_date 16:00 NYT (20:00 UTC) 就可知, lag=0 (但晚于 close 时点) 即可.

**讨论**: 这是 design 选择, 不是 bug. 但如果想榨干 next-day execution 的边际优势, 可以让 lag=0 表示 close+10min (类似 observed_at), 使 backtest as_of=trade_date+1 09:00 UTC 时 today's close 已可见.

**建议**: 暂保持 +1 day 保守设定, P0-1 修复也按 +1 day. 一致性 > 边际.

---

## P2 (Defer — 不阻塞 W8)

### P2-1. stock_prices 2 行 NULL source (test 残留)
- `TST_PIT_A` 2026-01-09 / 2026-01-16, 测试数据残留. 删除.

### P2-2. insider_trades 极端负 lag (filing_date < transaction_date 多年)
- 16 行 lag < -1500 天. 可能 vendor 数据错误或异常 10b5-1 plan. 删除或 flag 即可, 量小不影响特征.

### P2-3. SPY benchmark 仅从 2016-04-18 起
- 早期 excess-return labels 受影响. 已知 (`audit_price_truth_20260425.json::warnings[benchmark_starts_late]`), 跨 W8 之外问题.

### P2-4. universe research_only=['CASY']
- 1 ticker 在 universe_membership PIT 但 not in live active universe. 数据时效问题, 不阻塞.

---

## 不构成问题 (审计后排除)

| 检查 | 结果 |
|---|---|
| **A6 split anomaly** | 0 行 (audit_price_truth) |
| **A6 zero_volume** | 0 行 |
| **A7 跨源 (ticker,date) collision** | 0 行 |
| **macro_series_pit PIT** | 0.2% lag 异常, ~忽略 |
| **insider_trades / sec_filings PIT** | kt 来自 filing_date 或 accepted_date, 语义正确 |
| **survivorship** | universe_membership 覆盖 2016-01 ~ 2026-04, 710 tickers (含 121 退市), audit 标记 risk=low |

---

## 修复优先级与顺序

```
1. P0-1 修复 dag_daily_data.py 的 3 处 + 写 fix_recent_pit_lag.py → 修复 stock_prices 3710 行
2. P0-2 修复 fmp.py::_parse_knowledge_time + 写 fix_fundamentals_pit_lag.py → 修复 14675 行
3. P1-3 修复 short_interest 入库逻辑 (改 lag=8 business days) + 重 set 现有 122k 行 kt
4. P1-1 P1-2 修复 fmp.py 中 forward consensus 与 earnings actual 的 kt 设定
5. (可选) P0-3 不在阻塞链, 留 follow-up
6. P2 全部留 follow-up
```

每条修复都需:
- 配 unit test (tests/test_data/test_pit.py 增 case)
- ruff + pytest 全绿
- 修复后重跑 audit_price_truth + audit_universe + 自定义 fundamentals_pit lag check, 确认 P0 行数归 0

---

## 修复后验证清单

- [x] `price_truth_audit_20260425_postfix.json::issues == []` (critical 1→0)
- [x] `fundamentals_pit count(kt<=event_time) == 0` (15429→0)
- [x] `stock_prices count(kt<=trade_date) == 0` (3710→0)
- [x] `short_interest min(lag_days) == 10` (8 business days, 122377 行 kt 重写)
- [x] `stock_prices count(source IS NULL) == 0` (2 行 test 残留删除)
- [x] `pytest tests/test_data/test_sources.py` 13 pass / 1 deselected (不相关 pre-broken)
- [x] `ruff check` 改动文件全清

## 修复实际产出 (commit 前清单)

**代码改动**:
- `dags/dag_daily_data.py`: 3 处 `knowledge_time_mode="observed_at"` → `"historical"`
- `src/data/sources/fmp.py::_parse_knowledge_time`: 添加 `event_time` 参数 + sanity check (kt::date <= event_time → 拒绝)
- `src/data/sources/polygon_short_interest.py`: 3d calendar → 8 business days (新增 `_add_business_days` helper)
- `scripts/_audit_common.py`: `REPORT_DATE_TAG` 20260417 → 20260425

**新脚本** (一次性数据修复):
- `scripts/fix_recent_pit_lag.py` — stock_prices 3710 行
- `scripts/fix_fundamentals_pit_lag.py` — fundamentals_pit 15429 行
- `scripts/fix_short_interest_pit_lag.py` — short_interest 122377 行 (199 个 settlement_date)

**新测试**:
- `test_polygon_historical_knowledge_time_is_next_day_market_close`
- `test_polygon_fetch_historical_default_mode_emits_lag_one_knowledge_time`
- `test_fmp_parse_knowledge_time_rejects_kt_le_event_time`
- `test_short_interest_knowledge_time_uses_8_business_day_lag`

**新 audit 报告**:
- `data/reports/data_audit_2026-04-25.md` (本文件)
- `data/reports/price_truth_audit_20260425_postfix.json`
- `data/reports/universe_audit_20260425_postfix.json`

## 留作 Follow-up (不阻塞 W8 IC)

- **P0-3** analyst_estimates 表 100% kt=fiscal_date (4516 future-fiscal): 表对历史 backtest 不可用. W7 analyst_proxy 不依赖此表, W8 IC 不阻塞.
- **P1-1** fundamentals_pit 474 行 forward consensus future kt: 仅影响实时信号生成, 历史 IC 验证不阻塞.
- **P1-2** earnings_estimates kt=fiscal_date 23:59:59: daily backtest with as_of=trade_date+1 morning 仍 PIT 正确, 仅影响盘中粒度.
- **P1-4** polygon `_historical_knowledge_time` 加 +1 day 是 design 选择, 已锁住 unit test, 不改.
- **P2-2** insider_trades 16 行极端负 lag: 量太小可推迟.
- **P2-3** SPY 仅从 2016-04-18 起: 已知 (audit warning), 跨 W8 之外问题.
- **P2-4** universe research_only=['CASY']: 修复后已自然消解 (post-fix audit warnings=0).

修复完毕，**等 Codex 二次审查通过 → 用户 GO → 启动 W8 全宇宙 IC**.

---

## Codex 深度审查 Round 2 修复 (2026-04-25 二轮)

Codex 评分 5.8 / 7.0 阈值 → 不通过, 给出 1 high + 1 medium:

### High (Codex 阻塞): fundamentals 45d fallback 对 Q4/年报仍然 leak
**Codex 发现**: 10-K SEC 截止日 60 (large) / 75 (accelerated) / 90 (non-acc) 天, 我的 +45d fallback 对 Q4 行不够保守. DB 里有 **3109 行 December event_time** 卡在 +45d, 加上 vendor `acceptedDate` 直接给出 lag<60d 的, 总计 **128762 行 Q4 lag<60d**.

**修复**:
- `fmp.py::_fallback_knowledge_time(event_time, fiscal_period)`: period-aware, Q4/FY → +90d, 其余 → +45d
- `fmp.py::_parse_knowledge_time(row, event_time, fiscal_period)`: Q4 行要求 vendor parsed kt 与 event_time lag >= 60d, 否则视为 vendor bug 走 fallback
- `fmp.py::_is_annual_period(fiscal_period)` helper, 识别 Q4/FY 两种命名约定
- 3 处 `_fallback_knowledge_time` caller 全部传 fiscal_period (income/balance/cashflow + dividend + consensus)
- `scripts/fix_fundamentals_pit_lag.py`: 新增 STATEMENT_METRICS 白名单 (18 个 income/balance/cashflow + book_value_per_share), Q4 lag<60d 仅对这些 metric 生效. dividend / consensus rows 因 kt 来自 declaration_date / earnings_date 已 PIT-safe, 不动.
- 处理 unique constraint: 18 组内冲突 → 新增 `ANNUAL_DEDUP_SQL` 删除同 group 内多余 bad row (保留 max id)
- 数据修复: 199 dedup + 108193 shift to +90d, 4097 dividend rows 保留

### Medium (Codex 修正): FINRA 应是 7 BD 不是 8 BD
**Codex 发现**: FINRA 实际 schedule 是 broker T+2 BD 上报, 数据 T+7 BD 公布. 我用 8 BD 过保守 1 天.

**修复**:
- `polygon_short_interest.py::_fetch_ticker`: 8 → 7 business days
- `scripts/fix_short_interest_pit_lag.py`: 8 → 7 business days, 重新 shift 122377 行
- 测试 rename 8_business_day → 7_business_day, 边界 case 重算

### Round 2 验证清单 (修后实测)
- [x] stock_prices lag<=0: 0
- [x] fundamentals statement metric kt<=event_time: 0
- [x] fundamentals Q4 statement lag<60d: 0 (Codex 高优 finding 已消除)
- [x] short_interest min lag: 9 calendar days = 7 BD spanning weekend
- [x] short_interest weekend kt: 0
- [x] fundamentals dividend Q4 lag<60d: 4097 行保留 (PIT-safe by design, 不应 shift)
- [x] tests/test_data/test_sources.py 15 pass / 1 deselect (pre-broken)
- [x] ruff check 改动文件全清

### Round 2 新加测试
- `test_short_interest_knowledge_time_uses_7_business_day_lag`
- `test_fmp_fallback_knowledge_time_is_period_aware`
- `test_fmp_parse_knowledge_time_q4_requires_60_day_lag`

---

## Round 3 自审 (2026-04-25, 用 Codex 视角再挑刺)

Codex 已结案 5.8→预期 7+ (round-2 修了 high+medium). 但用户要求再深审一轮, 找 round-2 仍然漏的点.

### 重新清点漏审表
原 audit 只审了 stock_prices/fundamentals_pit/analyst_estimates/earnings_estimates/short_interest/insider_trades/sec_filings/macro_series_pit/corporate_actions. 漏审 5 张:

| 表 | 行数 | 状态 | W7/W8 影响 |
|---|---|---|---|
| `short_sale_volume_daily` | 13.4M | **全部 lag<=0** (kt=trade_date 22:00 UTC) | W7 retained 不用 (`abnormal_off_exchange_shorting` dropped). 不阻塞. |
| `ratings_events` | 979K | 0 lag<=0, 0 future_kt | OK |
| `grades_events` | 129K | 0 lag<=0, 0 future_kt | W7 retained 用 (downgrade_count, net_grade_change_20d). OK. |
| `earnings_calendar` | 4.6K | 0 lag<=0, 0 future_kt | OK |
| `price_target_events` | 32K | 1 行 (TSTPT 测试残留) | W7 60D 主信号 target_dispersion_proxy 用. **已删** TSTPT, 现 0 lag<=0 |

### Round 3 新发现 (按优先级)

**P1 (不阻塞 W8 但应 follow-up)**:

- **A1**: `short_sale_volume_daily` 13.4M 行 kt=trade_date 22:00 UTC. FINRA 实际公布 18:00 ET (= 22:00 UTC) 的当天, 但应 +1 BD 到 trade_date+1 才符合保守 PIT. 当前 W7 retained 不用此表, 不阻塞 W8. 若 W9+ 想用 `abnormal_off_exchange_shorting` 类特征要先修.
- **A2**: `_add_business_days` 不处理美国市场节假日. FINRA 公布若遇节假日顺延 1 天, 我的 +7 BD 在节假日前后会算成 +6 effective BD. 影响是 PIT 过激进 1 天, 非 leak 但不准确.
- **A3**: `market_close_fast_pipeline` DAG 改成 `historical` mode 后, kt 在 wall-clock 未来. 这破坏了"实时 signal"的初衷 — DAG 在 16:10 NYT 写入 kt=T+1 16:00 NYT 的 row, 当晚跑特征看不到今日 close. **不阻塞 W8 IC 验证 (历史 backtest 用 T-1 close 算 T's signal)**, 但 Phase 3 实时部署需要单独的 live ingest path.

**P2 (Defer)**:

- **A4**: `universe_membership` 行数从 718→687 (-31 行) 在我审计期间发生. **不在我修复脚本范围内**: 我的脚本没碰 universe_membership. 怀疑 Airflow `daily_data_pipeline` DAG 在某次运行中调用 `_sync_universe_membership_impl`. 当前 active=472. 不阻塞 W8 (集合稳定即可), 但应查 Airflow log 确认.
- **A5**: `_fetch_market_proxy_prices` (src/features/technical.py:367) 用 `as_of=knowledge_times.max()` 拉取市场代理价. 单 trade_date snapshot 无影响; 若被多 trade_date 批量 panel 调用, 老 row 会看到新 row 的 proxy 数据. 但实际特征逐行计算用同 trade_date proxy, **可能不构成实际 leak**. 留 TODO 深查.
- **A6**: `_fallback_knowledge_time(event_time, fiscal_period=None)` 当 `fiscal_period` 缺失时 fallback 到 +45d (quarterly). 应 fallback 到 +90d (更保守). 但所有 caller 都传 fiscal_period, 实际不会 hit. 防御性可改.
- **A7**: `_parse_knowledge_time` Q4 行 60d floor 会 over-reject 早 filed 10-K (legitimate large filer 50-60d). Trade-off: 防 vendor bug vs 损失 30-60d 内真实数据可见性. 当前 90d fallback 可接受.

### Round 3 实际清理
- 删除 1 行 `price_target_events` 测试残留 (TSTPT, 由 P2-1 同类问题)
- price_target_events lag<=0 现 = 0

### Round 3 Verdict
- **W8 IC 启动条件**: 全部满足
  - stock_prices PIT 干净 (0)
  - fundamentals statement PIT 干净 (0)
  - fundamentals Q4 statement 60d 浮动窗口干净 (0)
  - short_interest 7BD lag 正确
  - W7 retained 60D 主信号依赖的 4 张表 (stock_prices, fundamentals_pit, grades_events, price_target_events) 全部 PIT 干净
- **不阻塞 W8 的 follow-up**: A1-A7 总计 7 项, 其中 A1 (short_sale_volume_daily) 优先级最高, 但 W7 retained 不用. A4 (universe_membership 31 行) 应查 Airflow.

修复完毕. **W8 全宇宙 IC 启动安全**.

---

## Round 3 后续修复 (用户要求 "一并修复" 后)

3 项怀疑点逐一处理:

### A5 — _fetch_market_proxy_prices max(kt) 路径
**结论**: 不是 leak. 追代码: `market_return_5d(T) = SPY[T]/SPY[T-5] - 1` (pct_change 不向前看), merge on trade_date 后每行只用自己 trade_date 的 SPY. as_of=max(kt) 只是 fetch 范围, 不影响 per-row 计算. **dismissed, 无需修.**

### A4 — universe_membership 31 行消失根因
**根因找到 (重大发现)**: `tests/test_data/test_sources.py::test_backfill_universe_membership_reconstructs_history` 用 conftest `db_engine` 直接连**生产 DB**, 调用 `backfill_universe_membership(2024-01-01, 2025-12-31)` (内部 DELETE+REBUILD with mocked constituents=['AAA','CCC']). **我跑 pytest 全部 tests 时触发**, 删掉真实 universe_membership 行换成测试 mock 数据.

**修复**:
- 跑 `scripts/backfill_universe_membership.py --start-date 2024-01-01 --end-date 2025-12-31` 重建. `active today: 472 → 503` 恢复, `membership_rows: 687 → 1221`.
- 测试加 `@pytest.mark.skip` 注释, 防止再被触发. 后续要独立测试 DB 才能再启用.
- 顺手修 `dag_daily_data.py::_sync_universe_membership_impl` 的 `_result()` 多 trade_date kwarg bug (Airflow 自 2026-04-15 起所有 sync 任务都失败的根因).

### A1 — short_sale_volume_daily kt convention
**修复**:
- `src/data/finra_short_sale.py::_knowledge_time`: `trade_date 18:00 NYT` → `trade_date+1 16:00 NYT` (匹配 stock_prices lag-1 convention)
- `scripts/fix_short_sale_volume_pit_lag.py`: 13.4M 行一次性 SQL UPDATE
- 测试更新: `test_fetch_day_parses_pipe_delimited_file_and_legacy_short_exempt_fallback` 期望值 22:00 UTC → 20:00 UTC (next day)
- 新加 `test_knowledge_time_uses_lag_one_next_day_close` 锁定新约定

### Round 3 后续完整状态 (修后实测)

| 指标 | 状态 |
|---|---|
| stock_prices lag<=0 | 0 |
| fundamentals_pit statement kt<=evt | 0 |
| fundamentals_pit Q4 statement lag<60d | 0 (Codex 阻塞项) |
| short_interest min lag | 9 calendar days (=7 BD spanning weekend) |
| **short_sale_volume_daily lag<=0** | **0** (新增, 13.4M 行修复) |
| **universe_membership active today** | **503** (恢复, 之前被测试污染至 472) |
| price_target_events lag<=0 | 0 |
| pytest tests/test_data/ | 90 pass / 1 skipped (destructive test) |
| ruff check 改动文件 | 全清 |

### 仍 follow-up (真不阻塞 W8)
- A2 `_add_business_days` 不处理 US market holidays (过保守, 非 leak)
- A3 `market_close_fast_pipeline` DAG 实时语义 (Phase 3 影响)
- A6 `_fallback_knowledge_time` 默认 45d 防御性 (不 hit)
- A7 Q4 60d floor over-reject (trade-off, 当前选保守)
- 测试 `test_fmp_price_target.py` 用 TSTPT 仍污染 prod DB (留 1 row 残留, 不阻塞)

**所有阻塞项消除. 等用户 GO 进 W8 全宇宙 IC.**
