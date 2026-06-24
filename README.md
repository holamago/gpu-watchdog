# gpu-watchdog

Session-scoped GPU training watchdog. It checks training heartbeat files, GPU
idleness, starts a CUDA keepalive when needed, stops keepalive when training
resumes, and sends Slack alerts.

## Install

Clone one copy per GPU session or user workspace. Each clone should keep its own
`.env`, logs, and runtime state.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install a CUDA-compatible PyTorch wheel if the default `torch` package does not
match the server.

## Session Env

Create one `.env` per GPU session. Do not share `GPU_WATCHDOG_NAME` across
sessions.

The env file can live inside this clone as `.env`, or outside the repo as a
shared session env. When `gpu-watchdog` and training repos sit side by side,
placing it one level above both repos is usually simplest.

```bash
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/XXXXX/XXXXX/XXXXX
GPU_WATCHDOG_NAME=server-a-gpu0-session-001
GPU_WATCHDOG_LOG_DIR=/data/s2s_en/workspace/nayoung/gpu-watchdog/logs
```

Default runtime paths are scoped from `GPU_WATCHDOG_NAME`:

- State: `/tmp/gpu-watchdog/<GPU_WATCHDOG_NAME>/state.json`
- Watchdog log: `<clone>/$GPU_WATCHDOG_LOG_DIR/<GPU_WATCHDOG_NAME>-watchdog.log`
- Keepalive log: `<clone>/$GPU_WATCHDOG_LOG_DIR/<GPU_WATCHDOG_NAME>-keepalive.log`

Path-unsafe characters in `GPU_WATCHDOG_NAME` are normalized before creating
files.

## Run

Start in the background with `nohup`:

```bash
./scripts/run/watchdog.sh --nohup --env-file /path/to/session/.env
```

Run in the foreground for debugging:

```bash
./scripts/run/watchdog.sh --foreground --env-file /path/to/session/.env
```

Run a single check:

```bash
python3 main.py --config config.yaml --once
```

The watchdog uses `/tmp/gpu-watchdog/<GPU_WATCHDOG_NAME>/watchdog.lock`;
starting the same session twice exits with an "already running" message.

## Configuration

Edit `config.yaml` for:

- `training_jobs.heartbeat_path`: JSON heartbeat file refreshed by training
- `training_jobs.heartbeat_timeout_seconds`: max heartbeat age before a job is stopped
- `training_jobs.*_marker_path`: marker paths. `${ENV_VAR}` placeholders are supported.
- `keepalive.command`: CUDA keepalive workload and device
- `slack.alert_timezone`: alert/log timezone, currently `America/Toronto`
- `slack.keepalive_failure_alert_interval_seconds`: repeated failure alert throttle

Job names and marker paths are managed in `config.yaml`, not in the shared
session env.
For heartbeat and completion/failure marker files, see `docs/training-markers.md`.
For a Moshi fine-tune example, see `docs/examples/moshi-finetune.md`.

## Logs

- `watchdog.log`: watchdog decisions, GPU reads, Slack errors, keepalive start failures
- `keepalive.log`: CUDA keepalive worker output, PyTorch/CUDA errors, OOMs

When a training heartbeat becomes alive, the watchdog stops the recorded
keepalive process so training does not share GPU resources with it.

Common checks:

```bash
tail -f "$GPU_WATCHDOG_LOG_DIR/$GPU_WATCHDOG_NAME-watchdog.log"
tail -f "$GPU_WATCHDOG_LOG_DIR/$GPU_WATCHDOG_NAME-keepalive.log"
```

## Alerts

Slack alerts include the `Session` field from `GPU_WATCHDOG_NAME`. This is the
primary identifier when hostnames are identical across sessions.

Webhook smoke test:

```bash
python - <<'PY'
from gpu_watchdog.config import SlackConfig
from gpu_watchdog.slack import SlackNotifier

ok = SlackNotifier(SlackConfig()).send("[GPU Watchdog] Slack test message")
print("sent:", ok)
PY
```

`sent: True` means the webhook request succeeded.

## Tests

```bash
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/..."
./scripts/tests/run_alert_tests.sh
```

Local throttle test, no Slack webhook required:

```bash
./scripts/tests/test_keepalive_failure_throttle.sh
```

Tests use temporary configs under `/tmp` and do not modify `config.yaml`.