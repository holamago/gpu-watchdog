# Copyright (c) 2026- MAGO

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import logging
import os
from pathlib import Path
import time
from typing import Iterator

try:
    from dotenv import load_dotenv
except ImportError:
    # Allow environments that export variables directly to run without python-dotenv.
    def load_dotenv() -> bool:
        """
        No-op fallback used when python-dotenv is not installed.
        """
        return False


from gpu_watchdog.config import load_config
from gpu_watchdog.config import path_safe_session_name
from gpu_watchdog.monitor import Watchdog
from gpu_watchdog.slack import SlackNotifier


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the watchdog process.
    """
    parser = argparse.ArgumentParser(description="Monitor training and keep GPUs alive.")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to the watchdog YAML config file.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single watchdog check and exit.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Python logging level.",
    )
    return parser.parse_args()


def configure_process_timezone(timezone_name: str | None) -> None:
    """
    Configure process-local timezone for Python logging timestamps.

    Args:
        timezone_name: IANA timezone name such as "America/Toronto".
    """
    if not timezone_name or not hasattr(time, "tzset"):
        return

    os.environ["TZ"] = timezone_name
    time.tzset()


def lock_path_for_session(session_name: str | None) -> Path:
    """
    Return the lock path used to prevent duplicate watchdogs per session.
    """
    if not session_name:
        return Path("/tmp/gpu-watchdog/default/watchdog.lock")

    return Path("/tmp/gpu-watchdog") / path_safe_session_name(session_name) / "watchdog.lock"


@contextmanager
def single_instance_lock(lock_path: Path) -> Iterator[None]:
    """
    Acquire an exclusive non-blocking lock for the long-running watchdog.

    Args:
        lock_path: Path to the session lock file.

    Raises:
        SystemExit: If another watchdog already holds the session lock.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as handle:
        try:
            # Prevent duplicate watchdogs in the same session from racing on state.
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"gpu-watchdog is already running for this session: {lock_path}")

        handle.seek(0)
        handle.truncate()
        handle.write(f"{os.getpid()}\n")
        handle.flush()

        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def main() -> None:
    """
    Load configuration and run the watchdog once or forever.
    """
    load_dotenv()

    args = parse_args()
    config = load_config(args.config)
    configure_process_timezone(config.slack.alert_timezone)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    watchdog = Watchdog(config=config, notifier=SlackNotifier(config.slack))

    if args.once:
        watchdog.check_once()
        return

    session_name = config.session_name or os.getenv(config.session_name_env_var)
    with single_instance_lock(lock_path_for_session(session_name)):
        watchdog.run_forever()


if __name__ == "__main__":
    main()

