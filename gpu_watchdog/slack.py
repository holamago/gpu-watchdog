from __future__ import annotations

from dataclasses import dataclass
import os
import socket

import requests

from gpu_watchdog.config import SlackConfig


@dataclass(frozen=True)
class SlackNotifier:
    config: SlackConfig

    def send(self, text: str) -> bool:
        webhook_url = self.config.webhook_url or os.getenv(self.config.webhook_env_var)
        if not webhook_url:
            return False

        response = requests.post(
            webhook_url,
            json={"text": text},
            timeout=self.config.request_timeout_seconds,
        )
        response.raise_for_status()
        return True


def format_alert(title: str, body: str) -> str:
    return (
        f"[GPU Watchdog] {title}\n\n"
        f"Host: {socket.gethostname()}\n"
        f"{body}"
    )

