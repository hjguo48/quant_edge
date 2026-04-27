# W12 Automation Final Plan — 2026-04-27

**起草**: Claude (designer) + Codex (deep review)
**目标**: 完全自动化灰度测试链路, 8 周 hands-off
**Path**: Y+ on Windows Task Scheduler (NOT Airflow, NOT WSL systemd)
**状态**: Ready to execute

---

## 1. 整条链路架构

```
[Windows Task Scheduler on host]
        │                     │
        │ weekly Sat 06:15    │ daily 09:00
        ▼                     ▼
QuantEdge-W12-Greyscale     QuantEdge-W12-Watchdog
        │                     │
        ▼                     ▼
wsl.exe → /home/jiahao/quant_edge/scripts/run_w12_greyscale_once.sh
        │
        ├── preflight: DB freshness loop (12 × 10min retry)
        ├── run_greyscale_live.py (full universe, --dry-run paper)
        ├── run_greyscale_monitor.py (4-week required)
        ├── write last_success.json / last_failure.json
        └── send email alert on red/yellow

        │
        ▼
data/reports/greyscale/week_*.json
        │
        ▼
FastAPI /api reads from disk via GreyscaleReader (snapshot reload)
        │
        ▼
Frontend GreyscaleMonitor 页面 (Vite dev :4173)
```

**Key**: 前后端服务**常驻** (systemd user 或 Windows at-logon), wrapper 写 report 后**自动** visible — 不需重启服务.

---

## 2. 4 个组件详细规格

### 2A. Wrapper — `scripts/run_w12_greyscale_once.sh`

职责: 一次性周度 run, idempotent, 安全可重入.

执行流程:
1. 加载 `.venv`
2. 获 lock (避免重复运行)
3. Preflight: DB freshness loop (12 × 10min)
4. Run `run_greyscale_live.py --bundle-path ... --dry-run`
5. Run `run_greyscale_monitor.py --required-weeks 4 ...`
6. Parse output → write `last_success.json` (含 layer1-4 pass + gate_status)
7. On any stage fail → write `last_failure.json` + send RED alert
8. On red layer / FAIL gate → send RED alert (但仍认为是成功 cycle)

关键参数:
- Bundle: `data/models/bundles/w12_60d_ridge_swbuf_v1/bundle.json`
- Reference FM: `data/features/walkforward_v9full9y_fm_60d.parquet`
- Report dir: `data/reports/greyscale/`
- Lock: `data/reports/greyscale/.w12_greyscale.lock`
- Logs: `data/reports/greyscale/logs/greyscale_<RUN_ID>.log`

Retry policy:
- Preflight: 12 × 10min, 最多 120min
- Run失败: wrapper 内部 0 retry, scheduler 1 retry 可接受
- Monitor 失败: 标记 failure + alert, 但保留 weekly report

### 2B. Email alert — `scripts/send_w12_email_alert.py`

Gmail SMTP + **App Password** (NOT account password).

Config: `~/.config/quantedge/w12_alert.env` (chmod 600)
```
W12_ALERT_SMTP_HOST=smtp.gmail.com
W12_ALERT_SMTP_PORT=465
W12_ALERT_FROM=hjguo48@gmail.com
W12_ALERT_TO=hjguo48@gmail.com
W12_ALERT_APP_PASSWORD=xxxx xxxx xxxx xxxx
```

Args: `--severity {RED,YELLOW,TEST}` `--subject` `--body`

**推荐 email 不是 webhook 不是 local file** (用户不会主动看 file).

### 2C. Watchdog — `scripts/run_w12_watchdog.py`

每日 09:00 跑.

检查项:
1. `last_success.json` 存在 + 不老化 (> 8 天 → RED)
2. `last_failure.json` 是否新于 last success
3. DB 最新 PIT trade date 不老化 (> 3 天 → RED)
4. `/api/health` 返回 200 (失败 1 次 YELLOW, 重复 RED)
5. Frontend `/` 200 (失败 1 次 YELLOW, 重复 RED)
6. Disk free space (< 10GB YELLOW, < 5GB RED)

任何 RED → email alert.

### 2D. Windows Task Scheduler

**Weekly task** `QuantEdge-W12-Greyscale-Weekly`:
- Trigger: Saturday 06:15 Asia/Shanghai (= Friday 18:15 ET DST)
- Action:
  ```
  Program: C:\Windows\System32\wsl.exe
  Args: -d <Distro> -u jiahao -- bash -lc "cd /home/jiahao/quant_edge && ./scripts/run_w12_greyscale_once.sh"
  ```
- Settings:
  - Run whether user logged on or not
  - Run with highest privileges
  - Wake computer to run this task
  - If missed, run as soon as possible
  - Stop if runs longer than 3 hours
  - Restart on failure: 15min × 2 retries

**Daily watchdog task** `QuantEdge-W12-Watchdog-Daily`:
- Trigger: every day 09:00 local
- Action:
  ```
  Program: C:\Windows\System32\wsl.exe
  Args: -d <Distro> -u jiahao -- bash -lc "cd /home/jiahao/quant_edge && source .venv/bin/activate && python scripts/run_w12_watchdog.py"
  ```

---

## 3. 后端 + 前端常驻

### Backend (FastAPI uvicorn)

Service file: `~/.config/systemd/user/quantedge-api.service` (如 systemd 稳定):
```ini
[Unit]
Description=QuantEdge FastAPI backend
After=network-online.target

[Service]
WorkingDirectory=/home/jiahao/quant_edge
Environment=PYTHONPATH=/home/jiahao/quant_edge
ExecStart=/home/jiahao/quant_edge/.venv/bin/python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
Restart=on-failure
RestartSec=10

[Install]
WantedBy=default.target
```

或 Windows Task Scheduler "At logon" 任务调用 wsl.exe 启动.

GreyscaleReader 已支持 disk snapshot reload — **新 weekly report 落地无需重启**.

### Frontend (Vite dev server)

Command:
```bash
cd /home/jiahao/quant_edge/frontend
npm run dev -- --host 0.0.0.0 --port 4173
```

Vite 已配置 `/api` proxy → `http://127.0.0.1:8000`.

**Caveat**: Vite dev server 不是 production grade. 8 周内部 greyscale 用 OK, 不当生产 frontend.

User 浏览器访问: `http://localhost:4173/` → GreyscaleMonitor page.

---

## 4. 第一次启动顺序 (Step 0-10)

### Step 0: Pause Airflow weekly DAGs
```bash
docker exec quant_edge-airflow-1 airflow dags pause weekly_signal_pipeline
docker exec quant_edge-airflow-1 airflow dags pause weekly_rebalance_pipeline
```

### Step 1: 写 3 个 scripts
- `scripts/run_w12_greyscale_once.sh` (chmod +x)
- `scripts/send_w12_email_alert.py`
- `scripts/run_w12_watchdog.py`

### Step 2: 配 Gmail App Password
1. 用户去 Google Account → Security → 2-Step Verification → App passwords
2. 生成 16-char App Password
3. 写到 `~/.config/quantedge/w12_alert.env` (chmod 600)

### Step 3: 启 FastAPI backend
Manual first:
```bash
source .venv/bin/activate
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Verify: `curl http://127.0.0.1:8000/api/health` → 200 + `{"status":"ok"}`

### Step 4: 启 Frontend
Manual first:
```bash
cd frontend
npm run dev -- --host 0.0.0.0 --port 4173
```

Verify: 浏览器打开 `http://localhost:4173/` → 看到 Dashboard

### Step 5: 测 alert channel
```bash
source .venv/bin/activate
python scripts/send_w12_email_alert.py \
  --severity TEST \
  --subject "[QuantEdge W12] alert channel test" \
  --body "If you see this, email alerts work."
```

Verify: hjguo48@gmail.com 收到邮件

### Step 6: Supervised wrapper run (FULL universe)
```bash
cd /home/jiahao/quant_edge
./scripts/run_w12_greyscale_once.sh
```

Verify (~5min):
- New `data/reports/greyscale/week_NN.json` 写出
- `data/reports/greyscale/g4_gate_summary.json` 写出
- `data/reports/greyscale/last_success.json` 写出
- 没有 `last_failure.json` newer than success
- 没有意外 email (除非真有 red)

### Step 7: Verify backend sees new data
```bash
curl http://127.0.0.1:8000/api/predictions/latest
```

Verify: 返回 latest signal_date matches 新 weekly report

### Step 8: Verify frontend 看到新数据
浏览器 `http://localhost:4173/` → Greyscale Monitor page

Verify: 最新 week visible, charts 渲染

### Step 9: Manual watchdog run
```bash
source .venv/bin/activate
python scripts/run_w12_watchdog.py
```

Verify: exit 0, 没 alert sent

### Step 10: 注册 Windows Task Scheduler 任务
- Create `QuantEdge-W12-Greyscale-Weekly` (周六 06:15)
- Create `QuantEdge-W12-Watchdog-Daily` (每天 09:00)
- (可选) `QuantEdge-API-Backend` at logon
- (可选) `QuantEdge-Frontend-Vite` at logon

进入 hands-off 阶段.

---

## 5. 8-week hands-off ops

### 自动跑的
**Weekly (周六 06:15)**:
- preflight + greyscale + monitor
- 写 weekly report + heartbeat
- email 仅在 red 触发

**Daily (每天 09:00)**:
- watchdog 检查 heartbeat / API / frontend / disk
- email 仅在异常触发

### 用户做的
- ❌ **不**需每周手动启动
- ❌ **不**需登 Airflow Web UI
- ✅ 看 email alert (RED level 必须 action)
- ✅ 周末可选浏览器看 dashboard

### 4 周 / 8 周 review

`run_greyscale_monitor.py` 自动跑, 写 `g4_gate_summary.json`. 状态 = PENDING (week 1-3) / PASS / FAIL.

Week 4 后用户手动决定:
- PASS → 继续 paper 4 周 + 准备 W14 real-money
- FAIL → STOP, root cause review

---

## 6. Failure handling 矩阵

| Layer | Failure | Retry | Alert | Auto action | Human action |
|---|---|---:|---|---|---|
| Preflight | DB unreachable / PIT 老 | 12 × 10m | RED | no run | 查上游数据 |
| Greyscale run | nonzero exit | 0 wrapper / 1-2 scheduler | RED | no success heartbeat | 看 log, 手动重跑 |
| Monitor | nonzero exit | 0 | RED | report 存在 summary 缺 | 手动 monitor |
| Strategy health | layer fail / gate FAIL | none | RED/YELLOW | 仍标 success cycle | per guardrails review |
| Heartbeat write | filesystem fail | none | RED | fail run | 修磁盘 |
| Email alert | SMTP fail | none | local log only | continue run | 修 App Password |
| Backend | uvicorn down | watchdog daily | YELLOW→RED | none | restart service |
| Frontend | Vite down | watchdog daily | YELLOW→RED | none | restart service |
| Host/WSL2 | reboot/sleep | TS catch-up | RED if >8d stale | run on next avail | check power policy |
| Disk | low space | none | YELLOW/RED | none | clean space |

---

## 7. 监控 policy

### ALERT 级 (立即 action)
- `last_success.json` > 8 天 stale
- 最新 run failed
- DB freshness stale
- 任何 RED layer1/3 or gate FAIL
- 连续 2 天 API/frontend down

### INFO 级 (周末浏览)
- top scores
- holdings + weights
- turnover
- gate status PENDING / matured weeks count
- 最新 signal date

### Monthly review (week 4 + 8)
- rolling live IC vs backtest 0.0699
- positive weeks count
- turnover drift trend
- missingness / PSI alerts
- go/no-go for real money

---

## 8. 关键风险 + 缓解

| Risk | Mitigation |
|---|---|
| 上游 Airflow `daily_data_pipeline` flaky | watchdog freshness alert; W13 重写 daily |
| WSL2 sleep / restart | Windows TS (NOT WSL systemd) + wake computer |
| API key 过期 (Polygon/FMP/FRED) | preflight 抓下游影响; calendar reminder 月度 verify |
| UI silent dies | watchdog daily curl; 重复失败 RED |
| Vite dev 不是 production | 接受 8 周内部 OK; W13 hardening |

---

## 9. Honest assessment

**8 周可行性**: 中-高

| Component | 信心 |
|---|---|
| Core greyscale runner | 高 (full-universe dry-run PASS) |
| UI/observability | 中 (Vite dev OK 但 not hardened) |
| 8 周无人 intervention | 中 (WSL2 sleep + 上游 Airflow + API key) |

**必须有的**:
- Email alert
- Daily watchdog
- 不依赖 Airflow weekly DAGs

**省略以上任意一个 = 不可行**.

---

## 10. 立即行动

按 Step 0-10 执行. 先 0-1 (pause Airflow + 写 scripts), 再 2 (Gmail App Password), 然后 3-9 supervised, 最后 10 注册 TS.

完成后**真正进入 8 周 hands-off greyscale**.
