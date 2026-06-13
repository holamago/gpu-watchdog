from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import logging
import os
import time

from gpu_watchdog.config import TrainingJobConfig, WatchdogConfig
from gpu_watchdog.gpu import get_max_gpu_utilization
from gpu_watchdog.keepalive import is_process_running, start_keepalive
from gpu_watchdog.slack import SlackNotifier, format_alert
from gpu_watchdog.state import WatchdogState, load_state, save_state
from gpu_watchdog.trainer import classify_finished_training, is_training_alive


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingStopEvent:
    job_name: str
    status: str


@dataclass
class Watchdog:
    config: WatchdogConfig
    notifier: SlackNotifier

    def check_once(self) -> WatchdogState:
        state = load_state(self.config.state_path)
        training_jobs_alive = self._read_training_jobs_alive()
        training_alive = any(training_jobs_alive.values())
        max_gpu_utilization = self._read_gpu_utilization()

        self._update_idle_seconds(state, max_gpu_utilization)
        training_stop_events = self._notify_training_transitions(state, training_jobs_alive)
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
        while True:
            try:
                self.check_once()
            except Exception:
                LOGGER.exception("Watchdog check failed")

            time.sleep(self.config.check_interval_seconds)

    def _read_gpu_utilization(self) -> int | None:
        try:
            utilization = get_max_gpu_utilization(self.config.gpu.nvidia_smi_path)
        except Exception as exc:
            LOGGER.warning("Failed to read GPU utilization: %s", exc)
            return None

        LOGGER.info("Max GPU utilization: %s%%", utilization)
        return utilization

    def _read_training_jobs_alive(self) -> dict[str, bool]:
        jobs_alive: dict[str, bool] = {}
        for job in self.config.training_jobs:
            alive = is_training_alive(job.patterns)
            jobs_alive[job.name] = alive
            LOGGER.info("Training job status: name=%s alive=%s", job.name, alive)

        return jobs_alive

    def _update_idle_seconds(
        self,
        state: WatchdogState,
        max_gpu_utilization: int | None,
    ) -> None:
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
        stop_events: list[TrainingStopEvent] = []
        for job in self.config.training_jobs:
            training_alive = training_jobs_alive.get(job.name, False)
            last_training_alive = state.training_jobs.get(job.name)
            if last_training_alive is not True or training_alive:
                continue

            stop_events.append(self._notify_training_job_finished(state, job))

        return stop_events

    def _notify_training_job_finished(
        self, state: WatchdogState, job: TrainingJobConfig
    ) -> TrainingStopEvent:
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
        reason = self._keepalive_reason(training_alive, max_gpu_utilization, state)
        if reason is None:
            return

        if reason == "training_idle" and self.config.slack.notify_on_idle_keepalive:
            idle_minutes = state.idle_seconds // 60
            self._send_alert(
                "⚠️ [GPU Watchdog] GPU Idle Protection Started",
                (
                    "Summary:\n"
                    "Training GPU utilization is below the idle threshold.\n\n"
                    "Impact:\n"
                    "Keepalive will run to reduce idle reclaim risk.\n\n"
                    "Action:\n"
                    f"Monitor training status. Idle minutes: {idle_minutes}"
                ),
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

        if state.last_keepalive_failure_reason != failure_reason:
            if reason == "training_not_alive" and training_stop_events:
                self._send_gpu_reclaim_risk_alert(failure_reason, training_stop_events)
            else:
                self._send_keepalive_failed_alert(failure_reason, training_alive)
            state.last_keepalive_failure_reason = failure_reason

    def _keepalive_reason(
        self,
        training_alive: bool,
        max_gpu_utilization: int | None,
        state: WatchdogState,
    ) -> str | None:
        if not training_alive:
            return "training_not_alive"

        if max_gpu_utilization is None:
            return None

        idle_threshold_seconds = self.config.idle_threshold_minutes * 60
        if state.idle_seconds >= idle_threshold_seconds:
            return "training_idle"

        return None

    def _send_training_completed_alert(self, job_name: str) -> None:
        self._send_alert(
            "✅ [GPU Watchdog] Training Completed",
            (
                f"Job: {job_name}\n\n"
                "Training finished successfully."
            ),
        )

    def _send_training_stopped_alert(self, job_name: str, status: str) -> None:
        self._send_alert(
            "⚠️ [GPU Watchdog] Training Stopped",
            (
                f"Training: {job_name}\n\n"
                "Summary:\n"
                f"Training stopped with status: {status}.\n\n"
                "Impact:\n"
                "The training process is no longer running.\n\n"
                "Action:\n"
                "Check training logs and confirm whether this stop was expected."
            ),
        )

    def _send_keepalive_failed_alert(
        self,
        failure_reason: str,
        training_alive: bool,
    ) -> None:
        impact = (
            "GPU idle protection may not be active."
            if training_alive
            else "GPU resources may be reclaimed."
        )
        action = (
            "Check training status and restart keepalive."
            if training_alive
            else "Check GPU allocation and restart keepalive if resources are still needed."
        )
        self._send_alert(
            "⚠️ [GPU Watchdog] Keepalive Failed",
            (
                "Summary:\n"
                "Keepalive failed to start.\n\n"
                "Impact:\n"
                f"{impact}\n\n"
                "Reason:\n"
                f"• {failure_reason}\n\n"
                "Action:\n"
                f"{action}"
            ),
        )

    def _send_gpu_reclaim_risk_alert(
        self,
        failure_reason: str,
        training_stop_events: list[TrainingStopEvent],
    ) -> None:
        training_names = ", ".join(event.job_name for event in training_stop_events)
        self._send_alert(
            "🚨 [GPU Watchdog] GPU Reclaim Risk Detected",
            (
                f"Training: {training_names}\n\n"
                "Summary:\n"
                "Training stopped and keepalive failed.\n\n"
                "Impact:\n"
                "GPU resources may be reclaimed.\n\n"
                "Reason:\n"
                f"• {failure_reason}\n\n"
                "Action:\n"
                "Immediate action required. Check training status and restart keepalive."
            ),
        )

    def _send_alert(self, title: str, body: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        session_name = self.config.session_name or os.getenv(
            self.config.session_name_env_var
        )
        message = format_alert(title, f"{body}\nTime: {timestamp}", session_name)

        try:
            sent = self.notifier.send(message)
        except Exception:
            LOGGER.exception("Failed to send Slack alert")
            return

        if not sent:
            LOGGER.info("Slack webhook is not configured; alert was logged only: %s", title)

