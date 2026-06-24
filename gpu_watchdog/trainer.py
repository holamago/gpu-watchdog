# Copyright (c) 2026- MAGO

from __future__ import annotations

from pathlib import Path
import time


def is_training_alive(
    heartbeat_path: str,
    heartbeat_timeout_seconds: int,
    success_marker_path: str | None = None,
    failure_marker_path: str | None = None,
) -> bool:
    """
    Check whether a training heartbeat was updated recently.

    Args:
        heartbeat_path: Heartbeat JSON file refreshed by the training process.
        heartbeat_timeout_seconds: Maximum allowed heartbeat age.
        success_marker_path: Marker written by a successful training job.
        failure_marker_path: Marker written by a failed training job.

    Returns:
        True when the heartbeat exists and is fresh.
    """
    if _marker_exists(success_marker_path) or _marker_exists(failure_marker_path):
        return False

    heartbeat = Path(heartbeat_path)
    try:
        heartbeat_mtime = heartbeat.stat().st_mtime
    except FileNotFoundError:
        return False

    heartbeat_age_seconds = time.time() - heartbeat_mtime
    if heartbeat_age_seconds > heartbeat_timeout_seconds:
        return False

    return True


def classify_finished_training(
    success_marker_path: str | None,
    failure_marker_path: str | None,
) -> str:
    """
    Classify a stopped training job using optional marker files.

    Args:
        success_marker_path: Marker written by a successful training job.
        failure_marker_path: Marker written by a failed training job.

    Returns:
        "completed", "failed", or "stopped".
    """
    if failure_marker_path and Path(failure_marker_path).exists():
        return "failed"

    if success_marker_path and Path(success_marker_path).exists():
        return "completed"

    return "stopped"


def terminal_marker_signature(
    success_marker_path: str | None,
    failure_marker_path: str | None,
) -> str | None:
    """
    Return a stable signature for the current terminal marker, if any.
    """
    if failure_marker_path:
        signature = _marker_signature("failed", failure_marker_path)
        if signature:
            return signature

    if success_marker_path:
        signature = _marker_signature("completed", success_marker_path)
        if signature:
            return signature

    return None


def _marker_signature(status: str, marker_path: str) -> str | None:
    """
    Build a dedupe key from marker status, path, and mtime.
    """
    try:
        marker_stat = Path(marker_path).stat()
    except FileNotFoundError:
        return None

    return f"{status}:{marker_path}:{marker_stat.st_mtime_ns}"


def _marker_exists(marker_path: str | None) -> bool:
    """
    Return whether an optional marker path exists.
    """
    return bool(marker_path and Path(marker_path).exists())

