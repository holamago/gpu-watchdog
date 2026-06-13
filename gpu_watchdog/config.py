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
class TrainingJobConfig:
    name: str
    patterns: list[str]
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
    session_name: str | None = None
    session_name_env_var: str = "GPU_WATCHDOG_NAME"
    check_interval_seconds: int = 60
    idle_threshold_minutes: int = 300
    state_path: str = "state.json"
    training: TrainingConfig = field(default_factory=TrainingConfig)
    training_jobs: list[TrainingJobConfig] = field(default_factory=list)
    gpu: GpuConfig = field(default_factory=GpuConfig)
    keepalive: KeepaliveConfig = field(default_factory=KeepaliveConfig)
    slack: SlackConfig = field(default_factory=SlackConfig)


def load_config(path: str | Path) -> WatchdogConfig:
    config_path = Path(path)
    raw = _load_yaml(config_path)

    return WatchdogConfig(
        session_name=raw.get("session_name"),
        session_name_env_var=str(raw.get("session_name_env_var", "GPU_WATCHDOG_NAME")),
        check_interval_seconds=int(raw.get("check_interval_seconds", 60)),
        idle_threshold_minutes=int(raw.get("idle_threshold_minutes", 300)),
        state_path=str(raw.get("state_path", "state.json")),
        training=TrainingConfig(**raw.get("training", {})),
        training_jobs=_load_training_jobs(raw),
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


def _load_training_jobs(raw: dict[str, Any]) -> list[TrainingJobConfig]:
    raw_jobs = raw.get("training_jobs")
    if raw_jobs is None:
        training = TrainingConfig(**raw.get("training", {}))
        return [
            TrainingJobConfig(
                name="training",
                patterns=training.patterns,
                success_marker_path=training.success_marker_path,
                failure_marker_path=training.failure_marker_path,
            )
        ]

    if not isinstance(raw_jobs, list):
        raise ValueError("training_jobs must be a YAML list")
    if not raw_jobs:
        raise ValueError("training_jobs must contain at least one job")

    jobs: list[TrainingJobConfig] = []
    seen_names: set[str] = set()
    for index, raw_job in enumerate(raw_jobs, start=1):
        if not isinstance(raw_job, dict):
            raise ValueError("Each training_jobs entry must be a YAML object")

        name = str(raw_job.get("name") or "").strip()
        if not name:
            raise ValueError(f"training_jobs[{index}] must define a non-empty name")
        if name in seen_names:
            raise ValueError(f"Duplicate training job name: {name}")

        patterns = raw_job.get("patterns")
        if not isinstance(patterns, list) or not all(
            isinstance(pattern, str) and pattern for pattern in patterns
        ):
            raise ValueError(f"training job {name} must define non-empty patterns")

        jobs.append(
            TrainingJobConfig(
                name=name,
                patterns=patterns,
                success_marker_path=raw_job.get("success_marker_path"),
                failure_marker_path=raw_job.get("failure_marker_path"),
            )
        )
        seen_names.add(name)

    return jobs

