# Copyright (c) 2026- MAGO

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import logging
import os
import time
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from gpu_watchdog.config import TrainingJobConfig, WatchdogConfig
from gpu_watchdog.gpu import get_max_gpu_utilization
from gpu_watchdog.keepalive import is_process_running, start_keepalive, stop_keepalive
from gpu_watchdog.slack import (
    SlackNotifier,
    format_alert,
    format_alert_blocks,
    format_alert_body,
)
from gpu_watchdog.state import WatchdogState, load_state, save_state
from gpu_watchdog.trainer import (
    classify_finished_training,
    is_training_alive,
    terminal_marker_signature,
)


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingStopEvent:
    """
    Training stop event passed to downstream alert decisions.
    """

    job_name: str
    status: str


@dataclass
class Watchdog:
    """
    Main watchdog loop that monitors training, GPU idleness, and keepalive.
    """

    config: WatchdogConfig
    notifier: SlackNotifier

    def check_once(self) -> WatchdogState:
        """
        Run one watchdog check and persist the resulting state.
        """
        state = load_state(self.config.state_path)
        training_jobs_alive = self._read_training_jobs_alive()
        training_alive = any(training_jobs_alive.values())
        max_gpu_utilization = self._read_gpu_utilization()

        self._update_idle_seconds(state, max_gpu_utilization)
        training_stop_events = self._notify_training_transitions(state, training_jobs_alive)
        self._stop_keepalive_if_training_resumed(state, training_alive)
        self._ensure_keepalive_if_needed(
            state,
            training_alive,
            max_gpu_utilization,
            training_stop_events,
        )

        if state.keepalive_pid and not is_process_running(state.keepalive_pid):
            LOGGER.info("Recorded keepalive PID is no longer running: %s", state.keepalive_pid)
            state.keepalive_pid = None

        state.training_jobs = training_jobs_alive
        save_state(self.config.state_path, state)
        return state

    def run_forever(self) -> None:
        """
        Run watchdog checks until the process is stopped.
        """
        while True:
            try:
                self.check_once()
            except Exception:
                LOGGER.exception("Watchdog check failed")

            time.sleep(self.config.check_interval_seconds)

    def _read_gpu_utilization(self) -> int | None:
        """
        Read the maximum visible GPU utilization.
        """
        try:
            utilization = get_max_gpu_utilization(self.config.gpu.nvidia_smi_path)
        except Exception as exc:
            LOGGER.warning("Failed to read GPU utilization: %s", exc)
            return None

        LOGGER.info("Max GPU utilization: %s%%", utilization)
        return utilization

    def _read_training_jobs_alive(self) -> dict[str, bool]:
        """
        Read alive status for each configured training job.
        """
        jobs_alive: dict[str, bool] = {}
        for job in self.config.training_jobs:
            alive = is_training_alive(
                job.heartbeat_path,
                job.heartbeat_timeout_seconds,
                job.success_marker_path,
                job.failure_marker_path,
            )
            jobs_alive[job.name] = alive
            LOGGER.info(
                "Training job status: name=%s heartbeat_path=%s alive=%s",
                job.name,
                job.heartbeat_path,
                alive,
            )

        return jobs_alive

    def _stop_keepalive_if_training_resumed(
        self,
        state: WatchdogState,
        training_alive: bool,
    ) -> None:
        """
        Stop keepalive once a real training heartbeat is alive again.
        """
        if not training_alive or state.keepalive_pid is None:
            return

        keepalive_pid = state.keepalive_pid
        if stop_keepalive(keepalive_pid):
            LOGGER.info(
                "Stopped keepalive because training is alive again. pid=%s",
                keepalive_pid,
            )
            state.keepalive_pid = None
            state.last_event = "keepalive_stopped:training_alive"
            return

        LOGGER.warning(
            "Keepalive is still running after stop attempt. pid=%s",
            keepalive_pid,
        )

    def _update_idle_seconds(
        self,
        state: WatchdogState,
        max_gpu_utilization: int | None,
    ) -> None:
        """
        Update the accumulated idle duration based on GPU utilization.
        """
        if (
            max_gpu_utilization is not None
            and max_gpu_utilization < self.config.gpu.idle_utilization_threshold
        ):
            state.idle_seconds += self.config.check_interval_seconds
            return

        state.idle_seconds = 0

    def _notify_training_transitions(
        self,
        state: WatchdogState,
        training_jobs_alive: dict[str, bool],
    ) -> list[TrainingStopEvent]:
        """
        Detect jobs that transitioned from alive to stopped.
        """
        stop_events: list[TrainingStopEvent] = []
        for job in self.config.training_jobs:
            training_alive = training_jobs_alive.get(job.name, False)
            marker_stop_event = self._notify_terminal_marker_if_new(state, job)
            if marker_stop_event:
                stop_events.append(marker_stop_event)
                continue

            last_training_alive = state.training_jobs.get(job.name)
            if last_training_alive is not True or training_alive:
                continue

            stop_events.append(self._notify_training_job_finished(state, job))

        return stop_events

    def _notify_terminal_marker_if_new(
        self,
        state: WatchdogState,
        job: TrainingJobConfig,
    ) -> TrainingStopEvent | None:
        """
        Alert when a success/failed marker appears even if alive state was missed.
        """
        marker_signature = terminal_marker_signature(
            job.success_marker_path,
            job.failure_marker_path,
        )
        if marker_signature is None:
            return None

        if state.training_job_terminal_markers.get(job.name) == marker_signature:
            return None

        state.training_job_terminal_markers[job.name] = marker_signature
        return self._notify_training_job_finished(state, job)

    def _notify_training_job_finished(
        self, state: WatchdogState, job: TrainingJobConfig
    ) -> TrainingStopEvent:
        """
        Classify a stopped training job and send the appropriate alert.
        """
        finished_status = classify_finished_training(
            job.success_marker_path,
            job.failure_marker_path,
        )
        if finished_status == "completed":
            state.last_event = f"training_completed:{job.name}"
            if self.config.slack.notify_on_training_completed:
                self._send_training_completed_alert(job.name)
            return TrainingStopEvent(job.name, finished_status)

        state.last_event = f"training_{finished_status}:{job.name}"
        self._send_training_stopped_alert(job.name, finished_status)
        return TrainingStopEvent(job.name, finished_status)

    def _ensure_keepalive_if_needed(
        self,
        state: WatchdogState,
        training_alive: bool,
        max_gpu_utilization: int | None,
        training_stop_events: list[TrainingStopEvent],
    ) -> None:
        """
        Start or reuse keepalive when the session needs GPU protection.
        """
        reason = self._keepalive_reason(training_alive, max_gpu_utilization, state)
        if reason is None:
            return

        if reason == "training_idle" and self.config.slack.notify_on_idle_keepalive:
            idle_minutes = state.idle_seconds // 60
            self._send_alert(
                "⚠️ [GPU Watchdog] GPU Idle Protection Started",
                fields=[("Idle Minutes", str(idle_minutes))],
                sections=[
                    ("Action", "Keepalive started; monitor training status."),
                ],
            )

        result = start_keepalive(self.config.keepalive, state.keepalive_pid)
        if result.success:
            state.keepalive_pid = result.pid
            state.last_keepalive_failure_reason = None
            state.last_event = f"keepalive_started:{reason}"
            LOGGER.info("Keepalive is running. pid=%s reason=%s", result.pid, reason)
            return

        state.keepalive_pid = None
        state.last_event = "keepalive_failed"
        failure_reason = result.reason or "unknown error"
        LOGGER.error("Keepalive failed: %s", failure_reason)

        now = time.time()
        # Failure reasons can flap; rate-limit alerts by session instead of reason.
        if self._should_send_keepalive_failure_alert(state, now, failure_reason):
            if reason == "training_not_alive" and training_stop_events:
                self._send_gpu_reclaim_risk_alert(failure_reason, training_stop_events)
            else:
                self._send_keepalive_failed_alert(failure_reason, training_alive)
            state.last_keepalive_failure_alert_epoch = now
        else:
            LOGGER.info(
                "Skipping keepalive failure alert; last alert was sent less than %s seconds ago",
                self.config.slack.keepalive_failure_alert_interval_seconds,
            )

        state.last_keepalive_failure_reason = failure_reason

    def _should_send_keepalive_failure_alert(
        self,
        state: WatchdogState,
        now: float,
        failure_reason: str,
    ) -> bool:
        """
        Decide whether a keepalive failure should page Slack now.
        """
        interval_seconds = self.config.slack.keepalive_failure_alert_interval_seconds
        if interval_seconds <= 0:
            return state.last_keepalive_failure_reason != failure_reason

        last_alert_epoch = state.last_keepalive_failure_alert_epoch
        if last_alert_epoch is None:
            return True

        return now - last_alert_epoch >= interval_seconds

    def _keepalive_reason(
        self,
        training_alive: bool,
        max_gpu_utilization: int | None,
        state: WatchdogState,
    ) -> str | None:
        """
        Return the reason keepalive should run, or None when it is not needed.
        """
        if not training_alive:
            return "training_not_alive"

        if max_gpu_utilization is None:
            return None

        idle_threshold_seconds = self.config.idle_threshold_minutes * 60
        if state.idle_seconds >= idle_threshold_seconds:
            return "training_idle"

        return None

    def _send_training_completed_alert(self, job_name: str) -> None:
        """
        Send a training completion alert.
        """
        self._send_alert(
            "✅ [GPU Watchdog] Training Completed",
            fields=[("Job", job_name)],
        )

    def _send_training_stopped_alert(self, job_name: str, status: str) -> None:
        """
        Send an alert for stopped or failed training.
        """
        self._send_alert(
            "⚠️ [GPU Watchdog] Training Stopped",
            fields=[("Training", job_name), ("Status", status)],
            sections=[
                (
                    "Action",
                    "Check logs and confirm whether this stop was expected.",
                ),
            ],
        )

    def _send_keepalive_failed_alert(
        self,
        failure_reason: str,
        training_alive: bool,
    ) -> None:
        """
        Send an alert when keepalive could not start.
        """
        action = (
            "Check training status and restart keepalive."
            if training_alive
            else "Check GPU allocation and restart keepalive if resources are still needed."
        )
        self._send_alert(
            "🚨 [GPU Watchdog] Keepalive Failed",
            sections=[
                ("Reason", failure_reason),
                ("Action", action),
            ],
        )

    def _send_gpu_reclaim_risk_alert(
        self,
        failure_reason: str,
        training_stop_events: list[TrainingStopEvent],
    ) -> None:
        """
        Send a high-priority alert when training stopped and keepalive failed.
        """
        training_names = ", ".join(event.job_name for event in training_stop_events)
        self._send_alert(
            "🚨 [GPU Watchdog] GPU Reclaim Risk Detected",
            fields=[("Training", training_names)],
            sections=[
                ("Reason", failure_reason),
                (
                    "Action",
                    "Check training status and restart keepalive immediately.",
                ),
            ],
        )

    def _send_alert(
        self,
        title: str,
        fields: list[tuple[str, str]] | None = None,
        sections: list[tuple[str, str]] | None = None,
    ) -> None:
        """
        Format and send a Slack alert.
        """
        timestamp = self._alert_timestamp()
        session_name = self.config.session_name or os.getenv(
            self.config.session_name_env_var
        )
        alert_fields = [*(fields or []), ("Time", timestamp)]
        alert_sections = sections or []
        body = format_alert_body(alert_fields, alert_sections)
        message = format_alert(title, body, session_name)
        blocks = format_alert_blocks(title, alert_fields, alert_sections, session_name)

        try:
            sent = self.notifier.send(message, blocks=blocks)
        except Exception:
            LOGGER.exception("Failed to send Slack alert")
            return

        if not sent:
            LOGGER.info("Slack webhook is not configured; alert was logged only: %s", title)

    def _alert_timestamp(self) -> str:
        """
        Return the alert timestamp in the configured timezone.
        """
        timezone_name = self.config.slack.alert_timezone
        if timezone_name:
            try:
                timezone = ZoneInfo(timezone_name)
            except ZoneInfoNotFoundError:
                # Bad timezone config should not block critical Slack alerts.
                LOGGER.warning(
                    "Invalid Slack alert timezone configured: %s; using local timezone",
                    timezone_name,
                )
            else:
                return datetime.now(timezone).strftime("%Y-%m-%d %H:%M:%S %Z")

        return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")

