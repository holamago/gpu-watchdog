# Copyright (c) 2026- MAGO

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import json
import logging


LOGGER = logging.getLogger(__name__)


@dataclass
class WatchdogState:
    """
    Persisted state used to compare the current check with the previous check.
    """

    idle_seconds: int = 0
    training_jobs: dict[str, bool | None] = field(default_factory=dict)
    training_job_terminal_markers: dict[str, str] = field(default_factory=dict)
    keepalive_pid: int | None = None
    last_event: str | None = None
    last_keepalive_failure_reason: str | None = None
    # Persisted so repeated keepalive failures do not spam Slack after restarts.
    last_keepalive_failure_alert_epoch: float | None = None


def load_state(path: str | Path) -> WatchdogState:
    """
    Load watchdog state from disk.

    Args:
        path: Path to the JSON state file.

    Returns:
        A state object. Missing or invalid JSON files are treated as empty state.
    """
    state_path = Path(path)
    if not state_path.exists():
        return WatchdogState()

    try:
        with state_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        LOGGER.warning(
            "State file is invalid JSON (%s); resetting to defaults: %s",
            state_path,
            exc,
        )
        return WatchdogState()

    if not isinstance(data, dict):
        raise ValueError("State file must contain a JSON object")

    training_jobs = data.get("training_jobs")
    if not isinstance(training_jobs, dict):
        legacy_last_training_alive = data.get("last_training_alive")
        training_jobs = (
            {"training": legacy_last_training_alive}
            if legacy_last_training_alive is not None
            else {}
        )
    terminal_markers = data.get("training_job_terminal_markers")
    if not isinstance(terminal_markers, dict):
        terminal_markers = {}

    return WatchdogState(
        idle_seconds=int(data.get("idle_seconds", 0)),
        training_jobs=training_jobs,
        training_job_terminal_markers=terminal_markers,
        keepalive_pid=data.get("keepalive_pid"),
        last_event=data.get("last_event"),
        last_keepalive_failure_reason=data.get("last_keepalive_failure_reason"),
        last_keepalive_failure_alert_epoch=_optional_float(
            data.get("last_keepalive_failure_alert_epoch")
        ),
    )


def _optional_float(value: object) -> float | None:
    """
    Convert a persisted optional numeric value without failing state loading.
    """
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def save_state(path: str | Path, state: WatchdogState) -> None:
    """
    Save watchdog state to disk as stable, human-readable JSON.
    """
    state_path = Path(path)
    state_path.parent.mkdir(parents=True, exist_ok=True)

    with state_path.open("w", encoding="utf-8") as handle:
        json.dump(asdict(state), handle, indent=2, sort_keys=True)
        handle.write("\n")

