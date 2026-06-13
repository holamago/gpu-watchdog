from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess

import psutil

from gpu_watchdog.config import KeepaliveConfig


@dataclass(frozen=True)
class KeepaliveResult:
    success: bool
    pid: int | None = None
    reason: str | None = None


def is_process_running(pid: int | None) -> bool:
    if pid is None:
        return False

    try:
        process = psutil.Process(pid)
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def start_keepalive(config: KeepaliveConfig, existing_pid: int | None) -> KeepaliveResult:
    if is_process_running(existing_pid):
        return KeepaliveResult(success=True, pid=existing_pid)

    log_path = Path(config.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        log_handle = log_path.open("a", encoding="utf-8")
        try:
            process = subprocess.Popen(
                config.command,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                text=True,
            )
        finally:
            log_handle.close()
    except OSError as exc:
        return KeepaliveResult(success=False, reason=str(exc))

    try:
        return_code = process.wait(timeout=config.start_grace_seconds)
    except subprocess.TimeoutExpired:
        return KeepaliveResult(success=True, pid=process.pid)

    return KeepaliveResult(
        success=False,
        reason=f"Keepalive exited during startup with code {return_code}",
    )

