from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import json


@dataclass
class WatchdogState:
    idle_seconds: int = 0
    training_jobs: dict[str, bool | None] = field(default_factory=dict)
    keepalive_pid: int | None = None
    last_event: str | None = None
    last_keepalive_failure_reason: str | None = None


def load_state(path: str | Path) -> WatchdogState:
    state_path = Path(path)
    if not state_path.exists():
        return WatchdogState()

    with state_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

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

    return WatchdogState(
        idle_seconds=int(data.get("idle_seconds", 0)),
        training_jobs=training_jobs,
        keepalive_pid=data.get("keepalive_pid"),
        last_event=data.get("last_event"),
        last_keepalive_failure_reason=data.get("last_keepalive_failure_reason"),
    )


def save_state(path: str | Path, state: WatchdogState) -> None:
    state_path = Path(path)
    state_path.parent.mkdir(parents=True, exist_ok=True)

    with state_path.open("w", encoding="utf-8") as handle:
        json.dump(asdict(state), handle, indent=2, sort_keys=True)
        handle.write("\n")

