# Copyright (c) 2026- MAGO

from __future__ import annotations

import logging
import subprocess


LOGGER = logging.getLogger(__name__)
UNAVAILABLE_UTILIZATION_VALUES = {"[Not Found]", "N/A", "Not Supported"}


def get_gpu_utilizations(nvidia_smi_path: str = "nvidia-smi") -> list[int]:
    """
    Read per-GPU utilization percentages from nvidia-smi.

    Args:
        nvidia_smi_path: Path or command name for nvidia-smi.

    Returns:
        A list of utilization percentages.
    """
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
            if stripped in UNAVAILABLE_UTILIZATION_VALUES:
                LOGGER.debug(
                    "Skipping unavailable GPU utilization value: %s",
                    stripped,
                )
            else:
                LOGGER.warning(
                    "Skipping unexpected GPU utilization value: %s",
                    stripped,
                )

    if not values:
        raise RuntimeError("nvidia-smi returned no GPU utilization values")

    return values


def get_max_gpu_utilization(nvidia_smi_path: str = "nvidia-smi") -> int:
    """
    Return the highest visible GPU utilization percentage.
    """
    return max(get_gpu_utilizations(nvidia_smi_path))

