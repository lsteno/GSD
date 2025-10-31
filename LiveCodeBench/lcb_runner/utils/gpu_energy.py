import os
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

try:  # pragma: no cover - NVML is optional at runtime
    import pynvml  # type: ignore
except ImportError:  # pragma: no cover - NVML not installed
    pynvml = None

_NVML_ERROR = getattr(pynvml, "NVMLError", Exception)


def _resolve_device_index(default: int = 0) -> int:
    """Resolve the GPU index to monitor from CUDA_VISIBLE_DEVICES if available."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible:
        return default
    first_device = visible.split(",")[0].strip()
    if not first_device:
        return default
    try:
        return int(first_device)
    except ValueError:
        return default


@contextmanager
def gpu_energy_logger(device_index: Optional[int] = None) -> Iterator[Dict[str, Any]]:
    """Yield a dict that is populated with GPU energy stats after the block.

    The context gracefully degrades when NVML is unavailable, recording `None`
    values for the metrics and capturing an error message when applicable.
    """

    resolved_index = _resolve_device_index(0) if device_index is None else device_index
    summary: Dict[str, Any] = {
        "device_index": resolved_index,
        "energy_joules": None,
        "duration_seconds": None,
        "nvml_available": pynvml is not None,
        "error": None,
    }

    if pynvml is None:
        yield summary
        return

    initialized = False
    handle = None
    start_energy_mj = 0.0
    start_time = time.time()

    try:
        pynvml.nvmlInit()
        initialized = True
        handle = pynvml.nvmlDeviceGetHandleByIndex(resolved_index)
        start_energy_mj = float(pynvml.nvmlDeviceGetTotalEnergyConsumption(handle))
        yield summary
    except _NVML_ERROR as exc:  # pragma: no cover - hardware/runtime dependent
        summary["error"] = str(exc)
        yield summary
        return
    except Exception as exc:  # pragma: no cover - defensive fallback
        summary["error"] = str(exc)
        yield summary
        return
    finally:
        end_time = time.time()
        if handle is not None:
            try:
                end_energy_mj = float(
                    pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
                )
            except _NVML_ERROR as exc:  # pragma: no cover - hardware/runtime dependent
                summary["error"] = str(exc)
                end_energy_mj = start_energy_mj
        else:
            end_energy_mj = start_energy_mj

        if initialized:
            try:
                pynvml.nvmlShutdown()
            except _NVML_ERROR:
                pass

        duration_s = max(end_time - start_time, 0.0)
        energy_j = max((end_energy_mj - start_energy_mj) / 1000.0, 0.0)

        summary["duration_seconds"] = duration_s
        summary["energy_joules"] = energy_j
