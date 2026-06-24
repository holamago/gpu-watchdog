# Copyright (c) 2026- MAGO

from __future__ import annotations

from dataclasses import dataclass, field, replace
import os
from pathlib import Path
import re
from typing import Any

import yaml


ENV_VAR_PATTERN = re.compile(
    r"\$(?:\{(?P<braced>[A-Za-z_][A-Za-z0-9_]*)\}|(?P<plain>[A-Za-z_][A-Za-z0-9_]*))"
)


@dataclass(frozen=True)
class TrainingJobConfig:
    """
    Configuration for one training job tracked by heartbeat and markers.
    """

    name: str
    heartbeat_path: str
    heartbeat_timeout_seconds: int = 180
    success_marker_path: str | None = None
    failure_marker_path: str | None = None


@dataclass(frozen=True)
class GpuConfig:
    """
    GPU utilization query settings.
    """

    nvidia_smi_path: str = "nvidia-smi"
    idle_utilization_threshold: int = 1


@dataclass(frozen=True)
class KeepaliveConfig:
    """
    Command and startup settings for the keepalive subprocess.
    """

    command: list[str] = field(
        default_factory=lambda: ["python", "-m", "keepalive.gpu_keepalive"]
    )
    log_path: str = "keepalive.log"
    start_grace_seconds: int = 5


@dataclass(frozen=True)
class SlackConfig:
    """
    Slack webhook and alert formatting settings.
    """

    webhook_url: str | None = None
    webhook_env_var: str = "SLACK_WEBHOOK_URL"
    request_timeout_seconds: int = 10
    alert_timezone: str | None = None
    keepalive_failure_alert_interval_seconds: int = 3600
    notify_on_idle_keepalive: bool = False
    notify_on_training_completed: bool = False


@dataclass(frozen=True)
class WatchdogConfig:
    """
    Top-level watchdog configuration loaded from YAML.
    """

    session_name: str | None = None
    session_name_env_var: str = "GPU_WATCHDOG_NAME"
    check_interval_seconds: int = 60
    idle_threshold_minutes: int = 300
    state_path: str = "state.json"
    training_jobs: list[TrainingJobConfig] = field(default_factory=list)
    gpu: GpuConfig = field(default_factory=GpuConfig)
    keepalive: KeepaliveConfig = field(default_factory=KeepaliveConfig)
    slack: SlackConfig = field(default_factory=SlackConfig)


def load_config(path: str | Path) -> WatchdogConfig:
    """
    Load watchdog configuration and apply session-scoped runtime defaults.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        The fully resolved watchdog configuration.
    """
    config_path = Path(path)
    raw = _load_yaml(config_path)

    config = WatchdogConfig(
        session_name=expand_env_vars(raw.get("session_name")),
        session_name_env_var=expand_env_vars(
            str(raw.get("session_name_env_var", "GPU_WATCHDOG_NAME"))
        ),
        check_interval_seconds=int(raw.get("check_interval_seconds", 60)),
        idle_threshold_minutes=int(raw.get("idle_threshold_minutes", 300)),
        state_path=expand_env_vars(str(raw.get("state_path", "state.json"))),
        training_jobs=_load_training_jobs(raw),
        gpu=_load_gpu(raw),
        keepalive=_load_keepalive(raw),
        slack=_load_slack(raw),
    )

    return _scope_default_runtime_paths(config, config_path.parent)


def _load_yaml(path: Path) -> dict[str, Any]:
    """
    Read a YAML object from disk.
    """
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if not isinstance(data, dict):
        raise ValueError("Config file must contain a YAML object at the top level")

    return data


def expand_env_vars(value: str | None) -> str | None:
    """
    Expand ${ENV_VAR} or $ENV_VAR placeholders in config strings.

    Args:
        value: Optional config string.

    Returns:
        The expanded string, or None when the input is None.

    Raises:
        ValueError: If the string references an unset environment variable.
    """
    if value is None:
        return None

    missing_vars = sorted(
        {
            match.group("braced") or match.group("plain")
            for match in ENV_VAR_PATTERN.finditer(value)
            if os.getenv(match.group("braced") or match.group("plain")) is None
        }
    )
    if missing_vars:
        missing = ", ".join(missing_vars)
        raise ValueError(f"Missing environment variable(s) in config: {missing}")

    return os.path.expandvars(value)


def _expand_string_list(values: list[str]) -> list[str]:
    """
    Expand environment variables in a list of config strings.
    """
    return [expand_env_vars(value) or "" for value in values]


def _load_gpu(raw: dict[str, Any]) -> GpuConfig:
    """
    Load GPU config with env expansion.
    """
    gpu = raw.get("gpu", {})
    if not isinstance(gpu, dict):
        raise ValueError("gpu must be a YAML object")

    return GpuConfig(
        nvidia_smi_path=expand_env_vars(str(gpu.get("nvidia_smi_path", "nvidia-smi")))
        or "nvidia-smi",
        idle_utilization_threshold=int(gpu.get("idle_utilization_threshold", 1)),
    )


def _load_keepalive(raw: dict[str, Any]) -> KeepaliveConfig:
    """
    Load keepalive config with env expansion.
    """
    keepalive = raw.get("keepalive", {})
    if not isinstance(keepalive, dict):
        raise ValueError("keepalive must be a YAML object")

    command = keepalive.get("command")
    return KeepaliveConfig(
        command=_expand_string_list(command)
        if isinstance(command, list)
        else KeepaliveConfig().command,
        log_path=expand_env_vars(str(keepalive.get("log_path", "keepalive.log")))
        or "keepalive.log",
        start_grace_seconds=int(keepalive.get("start_grace_seconds", 5)),
    )


def _load_slack(raw: dict[str, Any]) -> SlackConfig:
    """
    Load Slack config with env expansion.
    """
    slack = raw.get("slack", {})
    if not isinstance(slack, dict):
        raise ValueError("slack must be a YAML object")

    return SlackConfig(
        webhook_url=expand_env_vars(slack.get("webhook_url")),
        webhook_env_var=expand_env_vars(
            str(slack.get("webhook_env_var", "SLACK_WEBHOOK_URL"))
        )
        or "SLACK_WEBHOOK_URL",
        request_timeout_seconds=int(slack.get("request_timeout_seconds", 10)),
        alert_timezone=expand_env_vars(slack.get("alert_timezone")),
        keepalive_failure_alert_interval_seconds=int(
            slack.get("keepalive_failure_alert_interval_seconds", 3600)
        ),
        notify_on_idle_keepalive=bool(slack.get("notify_on_idle_keepalive", False)),
        notify_on_training_completed=bool(
            slack.get("notify_on_training_completed", False)
        ),
    )


def _scope_default_runtime_paths(
    config: WatchdogConfig,
    config_dir: Path,
) -> WatchdogConfig:
    """
    Move default runtime files into per-session locations.
    """
    session_name = config.session_name or os.getenv(config.session_name_env_var)
    if not session_name:
        return config

    session_dir = path_safe_session_name(session_name)
    state_path = config.state_path
    keepalive = config.keepalive

    # Keep runtime state out of the shared repo unless the operator chose a path.
    if state_path == "state.json":
        state_path = str(Path("/tmp/gpu-watchdog") / session_dir / "state.json")

    log_dir = os.getenv("GPU_WATCHDOG_LOG_DIR")
    # Keep logs persistent and per-session when a shared log directory is provided.
    if keepalive.log_path == "keepalive.log" and log_dir:
        keepalive = replace(
            keepalive,
            log_path=str(Path(log_dir) / f"{session_dir}-keepalive.log"),
        )
    elif keepalive.log_path == "keepalive.log":
        keepalive = replace(
            keepalive,
            log_path=str(config_dir / "logs" / session_dir / "keepalive.log"),
        )

    return replace(config, state_path=state_path, keepalive=keepalive)


def path_safe_session_name(session_name: str) -> str:
    """
    Convert a user-provided session name into a filesystem-safe slug.
    """
    # Session names are used in paths, so normalize anything shell/user-provided.
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "-", session_name.strip())
    return sanitized.strip(".-") or "unnamed-session"


def _load_training_jobs(raw: dict[str, Any]) -> list[TrainingJobConfig]:
    """
    Load the training job list.
    """
    raw_jobs = raw.get("training_jobs")
    if raw_jobs is None:
        raise ValueError("training_jobs must be defined")

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

        heartbeat_path = expand_env_vars(raw_job.get("heartbeat_path"))
        if not heartbeat_path:
            raise ValueError(f"training job {name} must define heartbeat_path")
        heartbeat_timeout_seconds = int(raw_job.get("heartbeat_timeout_seconds", 180))
        if heartbeat_timeout_seconds <= 0:
            raise ValueError(
                f"training job {name} must define a positive heartbeat_timeout_seconds"
            )

        jobs.append(
            TrainingJobConfig(
                name=name,
                heartbeat_path=heartbeat_path,
                heartbeat_timeout_seconds=heartbeat_timeout_seconds,
                success_marker_path=expand_env_vars(
                    raw_job.get("success_marker_path")
                ),
                failure_marker_path=expand_env_vars(
                    raw_job.get("failure_marker_path")
                ),
            )
        )
        seen_names.add(name)

    return jobs

