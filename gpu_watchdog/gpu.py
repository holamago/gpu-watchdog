from __future__ import annotations

import logging
import subprocess


LOGGER = logging.getLogger(__name__)


def get_gpu_utilizations(nvidia_smi_path: str = "nvidia-smi") -> list[int]:
    output = subprocess.check_output(
        [
            nvidia_smi_path,
            "--query-gpu=utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        stderr=subprocess.STDOUT,
        text=True,
    )

    values: list[int] = []
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        try:
            values.append(int(stripped))
        except ValueError:
            LOGGER.warning("Skipping non-numeric GPU utilization value: %s", stripped)

    if not values:
        raise RuntimeError("nvidia-smi returned no GPU utilization values")

    return values


def get_max_gpu_utilization(nvidia_smi_path: str = "nvidia-smi") -> int:
    return max(get_gpu_utilizations(nvidia_smi_path))

