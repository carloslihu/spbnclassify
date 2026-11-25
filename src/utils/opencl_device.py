"""Selection of the OpenCL device used by pybnesian for the CKDE computations.

This module must not import pybnesian (or anything that imports it), because the device has to be chosen before
pybnesian is imported.
"""

import os
import subprocess

# CUDA_VISIBLE_DEVICES before ``select_most_available_gpu`` restricts it to a single GPU, i.e., the GPUs that the user
# made available to the experiments (None means all of them)
ORIGINAL_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES")


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


def available_gpus() -> list[str]:
    """GPUs that the experiments may use: those of the original CUDA_VISIBLE_DEVICES if it was set, otherwise every GPU
    listed by nvidia-smi (by UUID). Empty without GPUs."""
    if ORIGINAL_VISIBLE_DEVICES is not None:
        return [g.strip() for g in ORIGINAL_VISIBLE_DEVICES.split(",") if g.strip()]
    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [line.strip() for line in output.splitlines() if line.strip()]


def worker_slots(
    n_workers: int, gpus: list[str], cores: list[int] | None
) -> list[tuple[str | None, int | None]]:
    """(GPU, CPU core) of each parallel worker: the GPUs are assigned round-robin, so with at most one worker per GPU each
    worker has its own device, and each worker is pinned to its own core (``cores=None`` disables the pinning)."""
    return [
        (
            gpus[i % len(gpus)] if gpus else None,
            cores[i % len(cores)] if cores else None,
        )
        for i in range(n_workers)
    ]


def assign_worker_device(slots) -> None:
    """Initializer of a parallel worker: takes a (GPU, core) slot from the ``slots`` queue, exposes only that GPU to
    OpenCL and pins the process to that core.

    It must run before the first CKDE computation: pybnesian creates its OpenCL context lazily, when a CKDE is first
    evaluated, so the device can still be chosen after pybnesian has been imported (e.g., when the spawned worker
    re-imports the main module).
    """
    gpu, core = slots.get()
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    if core is not None and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {core})
