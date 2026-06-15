# gpu-watchdog

Python watchdog for GPU training jobs. It monitors one user/session, tracks one
or more training jobs in that session, checks GPU utilization, starts a
keepalive job when needed, and sends Slack alerts for important events.

## Features

- Tracks multiple training jobs independently by command-line patterns.
- Reads GPU utilization with `nvidia-smi`.
- Starts a keepalive job when no configured training job is running.
- Starts a keepalive job when training is alive but the GPU stays idle for too long.
- Sends Slack alerts for per-job training completion, unexpected training stops, and keepalive startup failures.
- Persists runtime state in `state.json` so the watchdog can continue after restart.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

PyTorch may need a CUDA-specific wheel for your server. It is fine to install `torch` separately according to your CUDA environment.

## Configuration

The default configuration lives in `config.yaml`.

```yaml
check_interval_seconds: 60
idle_threshold_minutes: 300
state_path: state.json
session_name:
session_name_env_var: GPU_WATCHDOG_NAME

training_jobs:
  - name: default-training
    patterns:
      - "accelerate launch"
      - "train.py"
    success_marker_path:
    failure_marker_path:

gpu:
  nvidia_smi_path: nvidia-smi
  idle_utilization_threshold: 1

keepalive:
  command:
    - python
    - -m
    - keepalive.gpu_keepalive
    - --device
    - cuda:0
    - --matrix-size
    - "4096"
    - --work-seconds
    - "0.2"
    - --sleep-seconds
    - "2"
```

Use an environment variable for the Slack webhook instead of committing the URL to the config file.

```bash
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/..."
```

Use `GPU_WATCHDOG_NAME` to make Slack alerts identify the session or user. This
is especially useful when multiple cloud sessions send alerts to the same Slack
channel.

```bash
export GPU_WATCHDOG_NAME="user-a-session-001"
```

## Multiple Training Jobs

Run one watchdog per user/session. Inside that session, configure every training
job that should be tracked independently under `training_jobs`.

Each job needs:

- `name`: Unique name used in Slack alerts and state tracking.
- `patterns`: Command-line fragments that identify that training process.
- `success_marker_path`: File written by that job when training finishes successfully.
- `failure_marker_path`: File written by that job when training fails.

Example for one session with two GPUs and two independent training processes:

```yaml
session_name: user-a-session-001
state_path: /tmp/gpu-watchdog/user-a-session-001/state.json

training_jobs:
  - name: train-gpu0
    patterns:
      - "train.py --run-name train-gpu0"
    success_marker_path: /tmp/gpu-watchdog/user-a-session-001/train-gpu0/success
    failure_marker_path: /tmp/gpu-watchdog/user-a-session-001/train-gpu0/failed

  - name: train-gpu1
    patterns:
      - "train.py --run-name train-gpu1"
    success_marker_path: /tmp/gpu-watchdog/user-a-session-001/train-gpu1/success
    failure_marker_path: /tmp/gpu-watchdog/user-a-session-001/train-gpu1/failed

keepalive:
  log_path: /tmp/gpu-watchdog/user-a-session-001/keepalive.log
```

Start the training jobs with command lines that include the configured pattern:

```bash
CUDA_VISIBLE_DEVICES=0 python3 train.py --run-name train-gpu0
CUDA_VISIBLE_DEVICES=1 python3 train.py --run-name train-gpu1
```

The watchdog sends completion/failure alerts per job. Keepalive is still decided
for the whole session: it starts when none of the configured jobs is running, or
when the session's visible GPUs stay idle longer than `idle_threshold_minutes`.

## Run

Run one check and exit:

```bash
python main.py --once
```

Run continuously:

```bash
python main.py
```

Run the keepalive job directly:

```bash
python -m keepalive.gpu_keepalive --device cuda:0 --matrix-size 4096 --work-seconds 0.2 --sleep-seconds 2
```

For environments without `systemd`, such as some containers, WSL sessions, or
test shells, use `nohup`:

```bash
nohup python3 main.py --config config.yaml > watchdog.log 2>&1 &
```

Check the process and logs:

```bash
ps aux | grep '[m]ain.py'
tail -f watchdog.log
```

Stop the watchdog:

```bash
pkill -f 'python3 main.py --config config.yaml'
```

For production on a host booted with `systemd`, use the service file below.

## Training Exit Classification

The watchdog observes external processes by command-line pattern, so it cannot
read each training process exit code directly. To distinguish successful
completion from an unexpected stop, each training job must write marker files
and `config.yaml` must point to those files.

Without marker paths, or if neither marker exists when a previously running job
disappears, the watchdog treats that job as an unexpected stop.

For this job config:

```yaml
training_jobs:
  - name: train-gpu0
    patterns:
      - "train.py --run-name train-gpu0"
    success_marker_path: /tmp/gpu-watchdog/user-a-session-001/train-gpu0/success
    failure_marker_path: /tmp/gpu-watchdog/user-a-session-001/train-gpu0/failed
```

wrap the training entry point like this:

```python
from pathlib import Path

job_dir = Path("/tmp/gpu-watchdog/user-a-session-001/train-gpu0")
success_marker = job_dir / "success"
failure_marker = job_dir / "failed"

job_dir.mkdir(parents=True, exist_ok=True)
success_marker.unlink(missing_ok=True)
failure_marker.unlink(missing_ok=True)

try:
    train()
except Exception:
    failure_marker.touch()
    raise
else:
    success_marker.touch()
```

If you run two jobs in one session, repeat the same wrapper with a different
job directory for each job. The marker paths must match that job's
`success_marker_path` and `failure_marker_path` exactly.

## Slack Test

First confirm that the webhook itself works:

```bash
python - <<'PY'
from gpu_watchdog.config import SlackConfig
from gpu_watchdog.slack import SlackNotifier

ok = SlackNotifier(SlackConfig()).send("[GPU Watchdog] Slack test message")
print("sent:", ok)
PY
```

`sent: True` means the webhook request succeeded.

## Alert Scenario Tests

The `scripts/` directory contains runnable alert tests. They use temporary configs under `/tmp` and do not modify the real `config.yaml`.

```bash
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/..."
```

Run individual scenarios:

```bash
./scripts/test_training_completed_alert.sh
./scripts/test_training_failed_alert.sh
./scripts/test_keepalive_failed_alert.sh
./scripts/test_gpu_reclaim_risk_alert.sh
```

Run all scenarios:

```bash
./scripts/run_alert_tests.sh
```

Expected Slack alerts:

- `✅ [GPU Watchdog] Training Completed`
- `⚠️ [GPU Watchdog] Training Stopped`
- `🚨 [GPU Watchdog] Keepalive Failed`
- `🚨 [GPU Watchdog] GPU Reclaim Risk Detected`

## Keepalive Tuning

The keepalive job repeatedly performs CUDA matrix multiplication for `--work-seconds`, then sleeps for `--sleep-seconds`.

Start conservatively:

```yaml
- --matrix-size
- "4096"
- --work-seconds
- "0.2"
- --sleep-seconds
- "2"
```

If the GPU is still not recognized as active by your resource monitor, increase `--work-seconds` gradually. If utilization is too high, decrease `--work-seconds` or increase `--sleep-seconds`.

## systemd

An example unit file is available at `systemd/gpu-watchdog.service`. Use this
only on a host where `systemd` is running as PID 1.

If `systemctl` prints this error, the current environment does not support
`systemd`; use the `nohup` command above instead.

```bash
System has not been booted with systemd as init system (PID 1). Can't operate.
Failed to connect to bus: Host is down
```

```bash
sudo cp systemd/gpu-watchdog.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now gpu-watchdog.service
```

Check service status and logs:

```bash
systemctl status gpu-watchdog.service
journalctl -u gpu-watchdog.service -f
```
