import time
from contextlib import contextmanager

try:
    import pynvml  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "NVML bindings are required. Install with `pip install nvidia-ml-py`."
    ) from exc


@contextmanager
def gpu_energy_logger(device_index: int = 0):
    """Yield a dict that is populated with energy stats after the block."""
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)

    start_time = time.time()
    start_energy_mj = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
    summary = {"energy_joules": None, "duration_seconds": None, "device_index": device_index}

    try:
        yield summary
    finally:
        end_time = time.time()
        end_energy_mj = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
        pynvml.nvmlShutdown()

        duration_s = max(end_time - start_time, 0.0)
        energy_j = max((end_energy_mj - start_energy_mj) / 1000.0, 0.0)
        summary["energy_joules"] = energy_j
        summary["duration_seconds"] = duration_s
