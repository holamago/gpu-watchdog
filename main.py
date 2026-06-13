from __future__ import annotations

import argparse
import logging

from gpu_watchdog.config import load_config
from gpu_watchdog.monitor import Watchdog
from gpu_watchdog.slack import SlackNotifier


def parse_args() -> argparse.Namespace:
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


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = load_config(args.config)
    watchdog = Watchdog(config=config, notifier=SlackNotifier(config.slack))

    if args.once:
        watchdog.check_once()
        return

    watchdog.run_forever()


if __name__ == "__main__":
    main()

