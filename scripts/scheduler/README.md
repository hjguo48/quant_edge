# Windows Task Scheduler XML Configs

4 个 task XML 用于让 W12 灰度自动化在 Windows 上跑：

| File | Trigger | Action |
|---|---|---|
| `QuantEdge-W12-Greyscale-Weekly.xml` | 周六 06:15 (= 美东周五 18:15 DST) | 跑 wrapper |
| `QuantEdge-W12-Watchdog-Daily.xml` | 每天 09:00 | 跑 watchdog |
| `QuantEdge-API-AtLogon.xml` | 用户登录 | 启 FastAPI 后端 |
| `QuantEdge-Frontend-AtLogon.xml` | 用户登录 | 启 Vite 前端 |

## Deprecation Note: Frontend AtLogon

`QuantEdge-Frontend-AtLogon.xml` is deprecated after PR #27. The production frontend is
served by FastAPI from `frontend/dist` on the same port as `/api/*` (`8000`), so the
separate Vite dev-server task on port `5173` should be disabled or deleted on deployed
machines. The XML is kept only as a legacy local-development reference.

## 导入步骤 (在 Windows 上做)

### 1. 找出 WSL distro 名

PowerShell:
```powershell
wsl -l -v
```

输出类似:
```
NAME            STATE   VERSION
* Ubuntu-24.04  Running 2
```

记下名字 (e.g., `Ubuntu-24.04`).

### 2. 替换 XML 中的 `<YOUR_DISTRO>` 占位符

每个 XML 内 `<Arguments>` 行有 `&lt;YOUR_DISTRO&gt;` 占位符。
用记事本打开每个 XML，把 `<YOUR_DISTRO>` 替换成你的 distro 名 (e.g., `Ubuntu-24.04`)。

### 3. 在 Task Scheduler 导入

1. Win+R → `taskschd.msc`
2. 右侧 **Action → Import Task...**
3. 选择 XML 文件
4. **下一步** 直接确认 (默认配置 OK)
5. 重复对每个 XML

### 4. 验证

Task Scheduler Library 应看到 4 个 tasks:
- QuantEdge-W12-Greyscale-Weekly (Saturday 06:15 trigger)
- QuantEdge-W12-Watchdog-Daily (daily 09:00 trigger)
- QuantEdge-API-AtLogon (At log on trigger)
- QuantEdge-Frontend-AtLogon (At log on trigger)

### 5. 测试触发

右键任意 task → **Run**, 立即触发不等 schedule.

WSL2 应启动 + 任务跑通.

## 备注

- **不要忘了 Step 2**: Gmail App Password 配 `~/.config/quantedge/w12_alert.env`
  否则 watchdog 触发 alert 时邮件发不出.
- **测试 Email 通道** 一次:
  ```
  source .venv/bin/activate
  python scripts/send_w12_email_alert.py --severity TEST --subject test --body "first test"
  ```
- **第一次 Schedule 触发**: 5/2 周六 06:15 (= 5/1 周五 18:15 ET) — Week 1 灰度.
