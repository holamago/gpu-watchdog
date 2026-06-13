from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
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
        self._notify_training_transitions(state, training_jobs_alive)
        self._ensure_keepalive_if_needed(state, training_alive, max_gpu_utilization)

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
    ) -> None:
        for job in self.config.training_jobs:
            training_alive = training_jobs_alive.get(job.name, False)
            last_training_alive = state.training_jobs.get(job.name)
            if last_training_alive is not True or training_alive:
                continue

            self._notify_training_job_finished(state, job)

    def _notify_training_job_finished(
        self, state: WatchdogState, job: TrainingJobConfig
    ) -> None:
        finished_status = classify_finished_training(
            job.success_marker_path,
            job.failure_marker_path,
        )
        if finished_status == "completed":
            state.last_event = f"training_completed:{job.name}"
            if self.config.slack.notify_on_training_completed:
                self._send_alert(
                    "Training completed successfully.",
                    f"Training job: {job.name}",
                )
            return

        state.last_event = f"training_{finished_status}:{job.name}"
        self._send_alert(
            "Training process stopped unexpectedly.",
            f"Training job: {job.name}\nStatus: {finished_status}",
        )

    def _ensure_keepalive_if_needed(
        self,
        state: WatchdogState,
        training_alive: bool,
        max_gpu_utilization: int | None,
    ) -> None:
        reason = self._keepalive_reason(training_alive, max_gpu_utilization, state)
        if reason is None:
            return

        if reason == "training_idle" and self.config.slack.notify_on_idle_keepalive:
            idle_minutes = state.idle_seconds // 60
            self._send_alert(
                "GPU idle threshold reached during training.",
                f"Idle minutes: {idle_minutes}\nKeepalive job will be started.",
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
            self._send_alert(
                "Keepalive job failed.",
                f"Reason: {failure_reason}\nImmediate action required.",
            )
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

    def _send_alert(self, title: str, body: str) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        session_name = self.config.session_name or os.getenv(
            self.config.session_name_env_var
        )
        message = format_alert(title, f"Time: {timestamp}\n{body}", session_name)

        try:
            sent = self.notifier.send(message)
        except Exception:
            LOGGER.exception("Failed to send Slack alert")
            return

        if not sent:
            LOGGER.info("Slack webhook is not configured; alert was logged only: %s", title)

