# gpu-watchdog

Python watchdog for GPU training jobs. It monitors a training process, checks GPU utilization, starts a keepalive job when needed, and sends Slack alerts for important events.

## Features

- Detects training processes by command-line patterns such as `accelerate launch` or `train.py`.
- Reads GPU utilization with `nvidia-smi`.
- Starts a keepalive job when the training process disappears.
- Starts a keepalive job when training is alive but the GPU stays idle for too long.
- Sends Slack alerts for training completion, unexpected training stops, and keepalive startup failures.
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

training:
  patterns:
    - "accelerate launch"
    - "train.py"

gpu:
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

When the watchdog only observes an external process by PID pattern, it cannot read that process exit code directly. To distinguish successful completion from an unexpected stop, have your training script write marker files and configure them in `config.yaml`.

```yaml
training:
  success_marker_path: /tmp/training-success
  failure_marker_path: /tmp/training-failed
```

Example:

```python
from pathlib import Path

try:
    train()
except Exception:
    Path("/tmp/training-failed").touch()
    raise
else:
    Path("/tmp/training-success").touch()
```

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
```

Run all scenarios:

```bash
./scripts/run_alert_tests.sh
```

Expected Slack alerts:

- `Training completed successfully.`
- `Training process stopped unexpectedly.`
- `Keepalive job failed.`

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
