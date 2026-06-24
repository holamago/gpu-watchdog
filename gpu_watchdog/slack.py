# Copyright (c) 2026- MAGO

from __future__ import annotations

from dataclasses import dataclass
import os
import socket
from typing import Any

import requests

from gpu_watchdog.config import SlackConfig

SlackBlock = dict[str, Any]


@dataclass(frozen=True)
class SlackNotifier:
    """
    Small Slack webhook client used by the watchdog alerts.
    """

    config: SlackConfig

    def send(self, text: str, blocks: list[SlackBlock] | None = None) -> bool:
        """
        Send a Slack message.

        Args:
            text: Plain-text fallback message.
            blocks: Optional Slack Block Kit payload.

        Returns:
            True when a webhook was configured and the request succeeded.
        """
        webhook_url = self.config.webhook_url or os.getenv(self.config.webhook_env_var)
        if not webhook_url:
            return False

        payload: dict[str, Any] = {"text": text}
        if blocks:
            payload["blocks"] = blocks

        response = requests.post(
            webhook_url,
            json=payload,
            timeout=self.config.request_timeout_seconds,
        )
        response.raise_for_status()
        return True


def format_alert_body(
    fields: list[tuple[str, str]],
    sections: list[tuple[str, str]],
) -> str:
    """
    Build a plain-text Slack alert body from field and section pairs.
    """
    field_lines = [f"*{label}:* {value}" for label, value in fields]
    section_lines = [f"*{label}:* {value}" for label, value in sections]
    spacer = [""] if field_lines and section_lines else []
    return "\n".join([*field_lines, *spacer, *section_lines]).strip()


def format_alert(title: str, body: str, session_name: str | None = None) -> str:
    """
    Build the plain-text fallback alert message.
    """
    session_line = f"*Session:* {session_name}\n" if session_name else ""
    return (
        f"*{title}*\n\n"
        f"*Host:* {socket.gethostname()}\n"
        f"{session_line}"
        f"{body}"
    )


def format_alert_blocks(
    title: str,
    fields: list[tuple[str, str]],
    sections: list[tuple[str, str]],
    session_name: str | None = None,
) -> list[SlackBlock]:
    """
    Build Slack Block Kit content for better alert readability.
    """
    metadata = [("Host", socket.gethostname())]
    if session_name:
        metadata.append(("Session", session_name))
    metadata.extend(fields)

    blocks: list[SlackBlock] = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": title,
                "emoji": True,
            },
        },
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*{label}:*\n{value}"}
                for label, value in metadata
            ],
        },
    ]

    if sections:
        section_text = "\n".join(f"*{label}:* {value}" for label, value in sections)
        blocks.append(
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": section_text,
                },
            }
        )

    return blocks

