from __future__ import annotations

import argparse
import logging
import time

import torch


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Keep CUDA GPUs minimally active.")
    parser.add_argument("--device", default="cuda:0", help="CUDA device to use.")
    parser.add_argument(
        "--matrix-size",
        type=int,
        default=1024,
        help="Square matrix size used for the keepalive matmul.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=10.0,
        help="Seconds to sleep between GPU operations.",
    )
    parser.add_argument(
        "--work-seconds",
        type=float,
        default=1.0,
        help="Seconds to continuously run GPU operations before sleeping.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Python logging level.",
    )
    return parser.parse_args()


def run(device: str, matrix_size: int, sleep_seconds: float, work_seconds: float) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if matrix_size <= 0:
        raise ValueError("matrix_size must be greater than 0")
    if work_seconds <= 0:
        raise ValueError("work_seconds must be greater than 0")
    if sleep_seconds < 0:
        raise ValueError("sleep_seconds must be greater than or equal to 0")

    LOGGER.info(
        (
            "Starting GPU keepalive on %s. "
            "matrix_size=%s work_seconds=%s sleep_seconds=%s"
        ),
        device,
        matrix_size,
        work_seconds,
        sleep_seconds,
    )

    left = torch.randn((matrix_size, matrix_size), device=device)
    right = torch.randn((matrix_size, matrix_size), device=device)

    while True:
        deadline = time.monotonic() + work_seconds
        iterations = 0

        with torch.no_grad():
            while time.monotonic() < deadline:
                result = torch.matmul(left, right)
                torch.cuda.synchronize(device)
                iterations += 1

        del result
        LOGGER.debug("Completed %s keepalive matmul iterations", iterations)
        time.sleep(sleep_seconds)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    run(args.device, args.matrix_size, args.sleep_seconds, args.work_seconds)


if __name__ == "__main__":
    main()

