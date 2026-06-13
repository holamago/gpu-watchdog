from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json


@dataclass
class WatchdogState:
    idle_seconds: int = 0
    last_training_alive: bool | None = None
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

    return WatchdogState(
        idle_seconds=int(data.get("idle_seconds", 0)),
        last_training_alive=data.get("last_training_alive"),
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

