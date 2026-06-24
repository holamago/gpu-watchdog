# Moshi Fine-Tune Example

This example shows how to apply watchdog heartbeat and terminal markers to
`moshi-finetune`.

## Training Code Change

Edit `moshi-finetune/train.py`. The file already imports `os` and `Path`; add
`json`, `threading`, and `time`.

Before:

```python
def train(config: str):
    args: TrainArgs = TrainArgs.load(config, drop_extra_fields=False)
    set_logger(logging.INFO)

    with ExitStack() as exit_stack:
        _train(args, exit_stack)
    logger.info("Closed everything!")
```

After:

```python
def _watchdog_marker_paths(run_dir: str) -> tuple[Path, Path, Path]:
    marker_dir = Path(
        os.environ.get("GPU_WATCHDOG_MARKER_DIR", Path(run_dir) / "gpu-watchdog")
    )
    return marker_dir / "heartbeat", marker_dir / "success", marker_dir / "failed"


def _prepare_watchdog_markers(
    heartbeat_marker: Path,
    success_marker: Path,
    failure_marker: Path,
) -> None:
    success_marker.parent.mkdir(parents=True, exist_ok=True)
    success_marker.unlink(missing_ok=True)
    failure_marker.unlink(missing_ok=True)
    _write_watchdog_heartbeat(heartbeat_marker)


def _touch_watchdog_marker(marker_path: Path, status: str) -> None:
    try:
        marker_path.touch()
    except Exception:
        logger.exception(
            "Failed to write GPU watchdog %s marker: %s",
            status,
            marker_path,
        )
        raise


def _write_watchdog_heartbeat(
    heartbeat_marker: Path,
    epoch: int | None = None,
    step: int | None = None,
    status: str = "training",
) -> None:
    payload = {
        "timestamp": int(time.time()),
        "epoch": epoch,
        "step": step,
        "status": status,
    }
    tmp_marker = heartbeat_marker.with_name(
        f"{heartbeat_marker.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    try:
        heartbeat_marker.parent.mkdir(parents=True, exist_ok=True)
        tmp_marker.write_text(json.dumps(payload), encoding="utf-8")
        tmp_marker.replace(heartbeat_marker)
    except Exception:
        logger.exception(
            "Failed to write GPU watchdog heartbeat marker: %s",
            heartbeat_marker,
        )
        raise


def _start_watchdog_heartbeat(
    heartbeat_marker: Path,
    progress: dict[str, int | None],
    progress_lock: threading.Lock,
    interval_seconds: int = 30,
) -> tuple[threading.Event, threading.Thread]:
    stop_event = threading.Event()

    def refresh_heartbeat() -> None:
        while not stop_event.wait(interval_seconds):
            with progress_lock:
                epoch = progress.get("epoch")
                step = progress.get("step")
            _write_watchdog_heartbeat(heartbeat_marker, epoch=epoch, step=step)

    thread = threading.Thread(
        target=refresh_heartbeat,
        name="gpu-watchdog-heartbeat",
        daemon=True,
    )
    thread.start()
    return stop_event, thread


def _update_watchdog_progress(
    progress: dict[str, int | None],
    progress_lock: threading.Lock,
    step: int,
    epoch: int | None = None,
) -> None:
    with progress_lock:
        progress["epoch"] = epoch
        progress["step"] = step


def train(config: str):
    args: TrainArgs = TrainArgs.load(config, drop_extra_fields=False)
    set_logger(logging.INFO)
    heartbeat_marker, success_marker, failure_marker = _watchdog_marker_paths(
        args.run_dir
    )
    _prepare_watchdog_markers(heartbeat_marker, success_marker, failure_marker)
    watchdog_progress = {"epoch": None, "step": None}
    watchdog_progress_lock = threading.Lock()
    heartbeat_stop_event, heartbeat_thread = _start_watchdog_heartbeat(
        heartbeat_marker,
        watchdog_progress,
        watchdog_progress_lock,
    )

    try:
        with ExitStack() as exit_stack:
            _train(args, exit_stack, watchdog_progress, watchdog_progress_lock)
    except Exception:
        _touch_watchdog_marker(failure_marker, "failure")
        raise
    else:
        _touch_watchdog_marker(success_marker, "success")
    finally:
        heartbeat_stop_event.set()
        heartbeat_thread.join(timeout=5)

    logger.info("Closed everything!")
```

Then update progress inside the training loop after `state.start_step()`:

```python
while state.step < args.max_steps:
    state.start_step()
    _update_watchdog_progress(
        watchdog_progress,
        watchdog_progress_lock,
        step=state.step,
    )
```

With `torchrun`, every rank may attempt to write the same marker file. The
heartbeat write uses temporary-file replace, and the watchdog treats terminal
markers as higher priority than heartbeat freshness.
If marker creation fails because of a bad path, permissions, or disk issues,
the training log includes the marker path and traceback.

## Run Moshi Fine-Tune

Define Moshi jobs in `gpu-watchdog/config.yaml`. If your Moshi repo includes the
`run_with_watchdog.sh` wrapper, use it to load the session env and set the
matching marker directory:

```bash
./run_with_watchdog.sh \
  --job-name moshi-exp-a \
  --cuda 0 \
  --nproc-per-node 1 \
  --master-port 29511 \
  example/moshi_7B.yaml
```

For multiple Moshi jobs in one session, use a different marker directory per
job name:

```bash
./run_with_watchdog.sh \
  --job-name moshi-exp-a \
  --cuda 0 \
  --nproc-per-node 1 \
  --master-port 29511 \
  configs/moshi-exp-a.yaml

./run_with_watchdog.sh \
  --job-name moshi-exp-b \
  --cuda 1 \
  --nproc-per-node 1 \
  --master-port 29512 \
  configs/moshi-exp-b.yaml
```
