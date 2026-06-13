from __future__ import annotations

from pathlib import Path
import os

import psutil


def is_training_alive(patterns: list[str]) -> bool:
    current_pid = os.getpid()

    for process in psutil.process_iter(["pid", "cmdline"]):
        try:
            if process.info["pid"] == current_pid:
                continue

            cmdline = process.info.get("cmdline") or []
            command = " ".join(cmdline)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue

        if command and any(pattern in command for pattern in patterns):
            return True

    return False


def classify_finished_training(
    success_marker_path: str | None,
    failure_marker_path: str | None,
) -> str:
    if failure_marker_path and Path(failure_marker_path).exists():
        return "failed"

    if success_marker_path and Path(success_marker_path).exists():
        return "completed"

    return "stopped"

