# Solution to import helper script by this answer https://stackoverflow.com/a/33515264
import os
import subprocess
import sys


def _select_most_available_gpu() -> None:
    """Expose only the NVIDIA GPU with the most free memory (ties broken by lower utilization).

    pybnesian's OpenCL backend (CKDE) always uses the first visible GPU and has no API to choose another; when that GPU
    is saturated by other jobs it fails with CL_INVALID_PROGRAM (-44). This must run before pybnesian is imported. The
    xdist workers inherit the variable, so all of them use the same GPU. A CUDA_VISIBLE_DEVICES already set in the shell
    (e.g. `CUDA_VISIBLE_DEVICES=3 pytest`) takes precedence; without nvidia-smi nothing is changed.
    """
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        return
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
        return
    gpus = []
    for line in output.strip().splitlines():
        uuid, free, utilization = (field.strip() for field in line.split(","))
        try:
            gpus.append((int(free), -int(utilization), uuid))
        except ValueError:  # "[N/A]" fields
            continue
    if gpus:
        # The UUID identifies the GPU unambiguously, whatever the device ordering of CUDA/OpenCL
        os.environ["CUDA_VISIBLE_DEVICES"] = max(gpus)[2]


_select_most_available_gpu()

# Add the rutile-ai directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Add the 'tests' directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "tests")))
