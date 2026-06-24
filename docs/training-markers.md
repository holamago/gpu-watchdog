# Training Markers

The watchdog does not inspect training process command lines. Each training job
is tracked by files written by the training code:

- `heartbeat`: JSON file refreshed periodically while training is running
- `success`: touched when training finishes successfully
- `failed`: touched when training exits with an error

## Configure Jobs

Set one marker directory per training job:

```yaml
training_jobs:
  - name: moshi-exp-a
    heartbeat_path: /tmp/gpu-watchdog/${GPU_WATCHDOG_NAME}/moshi-exp-a/heartbeat
    heartbeat_timeout_seconds: 180
    success_marker_path: /tmp/gpu-watchdog/${GPU_WATCHDOG_NAME}/moshi-exp-a/success
    failure_marker_path: /tmp/gpu-watchdog/${GPU_WATCHDOG_NAME}/moshi-exp-a/failed
```

`config.yaml` supports `${ENV_VAR}` placeholders. Missing environment variables
fail fast when the watchdog starts.

Alive and final status rules:

- `success_marker_path` or `failure_marker_path` exists: training is stopped
- `heartbeat_path` exists and is newer than `heartbeat_timeout_seconds`: training is alive
- heartbeat is missing or stale: training is stopped
- `failure_marker_path` exists: stopped status is `failed`
- `success_marker_path` exists: stopped status is `completed`
- neither terminal marker exists: stopped status is `stopped`

Terminal markers are deduped by file mtime, so a fast failure still alerts even
if the watchdog did not observe the job while the heartbeat was alive.

Heartbeat JSON example:

```json
{
  "timestamp": 1750709823,
  "epoch": 3,
  "step": 12400,
  "status": "training"
}
```

The watchdog uses the heartbeat file mtime for alive detection. The JSON payload
is for operators, logs, and future alert context. Write heartbeat with an atomic
temporary-file replace so the watchdog never observes a partially written file.

## Write Markers

Wrap the training entry point so it refreshes heartbeat while training runs and
writes one terminal marker before exiting:

```python
import json
import os
import time
from pathlib import Path
import threading

job_dir = Path("/tmp/gpu-watchdog/server-a-gpu0-session-001/moshi-exp-a")
heartbeat_marker = job_dir / "heartbeat"
success_marker = job_dir / "success"
failure_marker = job_dir / "failed"

job_dir.mkdir(parents=True, exist_ok=True)
success_marker.unlink(missing_ok=True)
failure_marker.unlink(missing_ok=True)


def write_heartbeat(epoch=None, step=None, status="training") -> None:
    payload = {
        "timestamp": int(time.time()),
        "epoch": epoch,
        "step": step,
        "status": status,
    }
    tmp_marker = heartbeat_marker.with_name(
        f"{heartbeat_marker.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    tmp_marker.write_text(json.dumps(payload), encoding="utf-8")
    tmp_marker.replace(heartbeat_marker)


write_heartbeat()

stop_event = threading.Event()


def refresh_heartbeat() -> None:
    while not stop_event.wait(30):
        write_heartbeat(status="training")


heartbeat_thread = threading.Thread(target=refresh_heartbeat, daemon=True)
heartbeat_thread.start()

try:
    train()
except Exception:
    failure_marker.touch()
    raise
else:
    success_marker.touch()
finally:
    stop_event.set()
    heartbeat_thread.join(timeout=5)
```

Use a separate marker directory per training job. The paths in the wrapper must
match `heartbeat_path`, `success_marker_path`, and `failure_marker_path` in
`config.yaml`.

## Single Job

For one active job per session:

```yaml
training_jobs:
  - name: moshi-exp-a
    heartbeat_path: /tmp/gpu-watchdog/${GPU_WATCHDOG_NAME}/moshi-exp-a/heartbeat
    heartbeat_timeout_seconds: 180
    success_marker_path: /tmp/gpu-watchdog/${GPU_WATCHDOG_NAME}/moshi-exp-a/success
    failure_marker_path: /tmp/gpu-watchdog/${GPU_WATCHDOG_NAME}/moshi-exp-a/failed
```

Source the session env, then pass the matching marker directory to training:

```bash
set -a
source /path/to/session.env
set +a

GPU_WATCHDOG_MARKER_DIR=/tmp/gpu-watchdog/${GPU_WATCHDOG_NAME}/moshi-exp-a \
CUDA_VISIBLE_DEVICES=0 python train.py --config configs/moshi-exp-a.yaml
```

## Multiple Jobs

For multiple jobs in one session, configure each job explicitly:

```yaml
training_jobs:
  - name: moshi-exp-a
    heartbeat_path: /tmp/gpu-watchdog/${GPU_WATCHDOG_NAME}/moshi-exp-a/heartbeat
    heartbeat_timeout_seconds: 180
    success_marker_path: /tmp/gpu-watchdog/${GPU_WATCHDOG_NAME}/moshi-exp-a/success
    failure_marker_path: /tmp/gpu-watchdog/${GPU_WATCHDOG_NAME}/moshi-exp-a/failed

  - name: moshi-exp-b
    heartbeat_path: /tmp/gpu-watchdog/${GPU_WATCHDOG_NAME}/moshi-exp-b/heartbeat
    heartbeat_timeout_seconds: 180
    success_marker_path: /tmp/gpu-watchdog/${GPU_WATCHDOG_NAME}/moshi-exp-b/success
    failure_marker_path: /tmp/gpu-watchdog/${GPU_WATCHDOG_NAME}/moshi-exp-b/failed
```

Pass a different marker directory per training command:

```bash
set -a
source /path/to/session.env
set +a

GPU_WATCHDOG_MARKER_DIR=/tmp/gpu-watchdog/${GPU_WATCHDOG_NAME}/moshi-exp-a \
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 train.py --config configs/moshi-exp-a.yaml

GPU_WATCHDOG_MARKER_DIR=/tmp/gpu-watchdog/${GPU_WATCHDOG_NAME}/moshi-exp-b \
CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 train.py --config configs/moshi-exp-b.yaml
```

`CUDA_VISIBLE_DEVICES` is still how you choose the GPU for training, but the
watchdog no longer uses it for job detection.
