# Copyright (c) 2026- MAGO

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging
import subprocess

import psutil

from gpu_watchdog.config import KeepaliveConfig


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class KeepaliveResult:
    """
    Result of attempting to start or reuse a keepalive process.
    """

    success: bool
    pid: int | None = None
    reason: str | None = None


def is_process_running(pid: int | None) -> bool:
    """
    Check whether a PID points to a live, non-zombie process.
    """
    if pid is None:
        return False

    try:
        process = psutil.Process(pid)
        if not process.is_running():
            return False
        if process.status() == psutil.STATUS_ZOMBIE:
            # Reap dead child keepalives so PID checks do not leave zombies behind.
            _reap_zombie_process(process)
            return False
        return True
    except psutil.NoSuchProcess:
        return False


def stop_keepalive(pid: int | None, timeout_seconds: float = 5.0) -> bool:
    """
    Stop a recorded keepalive process when training resumes.

    Args:
        pid: Keepalive PID to stop.
        timeout_seconds: Seconds to wait before killing the process.

    Returns:
        True when there is no longer a live process for the PID.
    """
    if pid is None:
        return True

    try:
        process = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return True

    if not process.is_running():
        return True

    if process.status() == psutil.STATUS_ZOMBIE:
        _reap_zombie_process(process)
        return True

    process.terminate()
    try:
        process.wait(timeout=timeout_seconds)
        return True
    except psutil.TimeoutExpired:
        LOGGER.warning("Keepalive did not stop after SIGTERM; killing pid=%s", pid)
        process.kill()
        try:
            process.wait(timeout=timeout_seconds)
        except psutil.TimeoutExpired:
            LOGGER.error("Failed to stop keepalive pid=%s", pid)
            return False

    return True


def _reap_zombie_process(process: psutil.Process) -> None:
    """
    Reap a zombie process when it is a child of the current watchdog.
    """
    try:
        process.wait(timeout=0)
    except (psutil.NoSuchProcess, psutil.TimeoutExpired, ChildProcessError):
        return


def start_keepalive(config: KeepaliveConfig, existing_pid: int | None) -> KeepaliveResult:
    """
    Start keepalive unless the recorded keepalive process is still running.

    Args:
        config: Keepalive command and startup settings.
        existing_pid: Previously recorded keepalive PID.

    Returns:
        A result with either the running PID or a startup failure reason.
    """
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
        # A keepalive that survives startup is treated as managed by its PID.
        return KeepaliveResult(success=True, pid=process.pid)

    return KeepaliveResult(
        success=False,
        reason=f"Keepalive exited during startup with code {return_code}",
    )

