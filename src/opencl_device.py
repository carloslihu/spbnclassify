"""Selection of the OpenCL device used by pybnesian for the CKDE computations.

This module must not import pybnesian (or anything that imports it), because the device has to be chosen before
pybnesian is imported.
"""

import os
import subprocess


def select_most_available_gpu() -> str | None:
    """Exposes only the NVIDIA GPU with the most free memory (ties broken by lower utilization), as ``tests/conftest.py``.

    pybnesian's OpenCL backend always uses the first visible GPU and fails with CL_INVALID_PROGRAM (-44) when that GPU is
    saturated by other jobs. A CUDA_VISIBLE_DEVICES already set in the environment takes precedence, and nothing is
    changed without nvidia-smi.

    Returns:
        str | None: The value of CUDA_VISIBLE_DEVICES, if any.
    """
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        return os.environ["CUDA_VISIBLE_DEVICES"]
    try:
        output = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=uuid,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    gpus = []
    for line in output.strip().splitlines():
        uuid, free, utilization = (field.strip() for field in line.split(","))
        try:
            gpus.append((int(free), -int(utilization), uuid))
        except ValueError:  # "[N/A]" fields
            continue
    if gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = max(gpus)[2]
    return os.environ.get("CUDA_VISIBLE_DEVICES")


def opencl_device_name() -> str:
    """Name of the first visible GPU (the OpenCL device of pybnesian), or "cpu" if there is none."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        command = ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]
        if visible:
            command.insert(1, f"--id={visible.split(',')[0]}")
        output = subprocess.run(
            command, capture_output=True, text=True, check=True, timeout=10
        ).stdout
        return output.strip().splitlines()[0]
    except (OSError, subprocess.SubprocessError, IndexError):
        return "cpu"
