from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class TrainingConfig:
    patterns: list[str] = field(default_factory=lambda: ["accelerate launch"])
    success_marker_path: str | None = None
    failure_marker_path: str | None = None


@dataclass(frozen=True)
class GpuConfig:
    nvidia_smi_path: str = "nvidia-smi"
    idle_utilization_threshold: int = 1


@dataclass(frozen=True)
class KeepaliveConfig:
    command: list[str] = field(
        default_factory=lambda: ["python", "-m", "keepalive.gpu_keepalive"]
    )
    log_path: str = "keepalive.log"
    start_grace_seconds: int = 5


@dataclass(frozen=True)
class SlackConfig:
    webhook_url: str | None = None
    webhook_env_var: str = "SLACK_WEBHOOK_URL"
    request_timeout_seconds: int = 10
    notify_on_idle_keepalive: bool = False
    notify_on_training_completed: bool = False


@dataclass(frozen=True)
class WatchdogConfig:
    check_interval_seconds: int = 60
    idle_threshold_minutes: int = 300
    state_path: str = "state.json"
    training: TrainingConfig = field(default_factory=TrainingConfig)
    gpu: GpuConfig = field(default_factory=GpuConfig)
    keepalive: KeepaliveConfig = field(default_factory=KeepaliveConfig)
    slack: SlackConfig = field(default_factory=SlackConfig)


def load_config(path: str | Path) -> WatchdogConfig:
    config_path = Path(path)
    raw = _load_yaml(config_path)

    return WatchdogConfig(
        check_interval_seconds=int(raw.get("check_interval_seconds", 60)),
        idle_threshold_minutes=int(raw.get("idle_threshold_minutes", 300)),
        state_path=str(raw.get("state_path", "state.json")),
        training=TrainingConfig(**raw.get("training", {})),
        gpu=GpuConfig(**raw.get("gpu", {})),
        keepalive=KeepaliveConfig(**raw.get("keepalive", {})),
        slack=SlackConfig(**raw.get("slack", {})),
    )


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if not isinstance(data, dict):
        raise ValueError("Config file must contain a YAML object at the top level")

    return data

