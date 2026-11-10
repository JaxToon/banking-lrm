"""Runtime selection, seeding, determinism settings and environment metadata.

Call `configure_determinism()` BEFORE anything initializes CUDA (it sets
CUBLAS_WORKSPACE_CONFIG, which cuBLAS reads once at initialization), then
`select_device()`, then `seed_everything()`.

Determinism notes:
- `torch.use_deterministic_algorithms(True)` makes PyTorch raise on operations that
  have no deterministic implementation instead of silently running them.
- cuDNN autotuning and TF32 are disabled; computation stays FP32.
- Phase 3 models avoid fused scaled-dot-product attention kernels and atomic scatter
  operations, which are the usual sources of CUDA nondeterminism.
- Determinism holds for one device type, driver, library build and (on CPU) one
  `torch.get_num_threads()` value: measured CPU weights differ between 1, 4 and 10
  threads (different reduction order), so the thread count is recorded. Results are NOT
  claimed to be bitwise identical across CPU and GPU, or across GPU models/drivers.

Usage (prints the runtime report):
    python -m bri.seq.runtime [--device auto|cpu|cuda]
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import random
import sys
from dataclasses import asdict, dataclass

import numpy as np

CUBLAS_WORKSPACE = ":4096:8"
DEVICE_CHOICES = ("auto", "cpu", "cuda")


class RuntimeConfigurationError(RuntimeError):
    pass


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeConfigurationError(
            "PyTorch is not installed; install the optional `seq` extra "
            "(see requirements-seq-cpu.txt / requirements-seq-cuda.txt)") from exc
    return torch


def configure_determinism() -> dict:
    """Apply deterministic settings; must run before CUDA is initialized. Returns them."""
    torch = _torch()
    preset = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if torch.cuda.is_initialized() and preset not in (":4096:8", ":16:8"):
        raise RuntimeConfigurationError(
            "CUDA was initialized before configure_determinism(); CUBLAS_WORKSPACE_CONFIG can no "
            "longer take effect. Call configure_determinism() first.")
    if preset is None:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return determinism_settings()


def determinism_settings() -> dict:
    torch = _torch()
    return {
        "use_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "precision": "fp32",
        "cpu_threads": torch.get_num_threads(),
        "scope": ("deterministic per device type, driver, library build and CPU thread count "
                  "(CPU results change with torch.get_num_threads()); not cross-device bitwise"),
    }


@dataclass(frozen=True)
class DeviceChoice:
    requested: str
    device: str  # "cpu" or "cuda:0"
    fallback_reason: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def cuda_usable() -> tuple[bool, str | None]:
    """(usable, reason_if_not). Checks availability AND that the build supports the GPU."""
    torch = _torch()
    if not torch.cuda.is_available():
        return False, "torch.cuda.is_available() is False (no GPU, no driver, or a CPU-only build)"
    major, minor = torch.cuda.get_device_capability(0)
    arch = f"sm_{major}{minor}"
    archs = torch.cuda.get_arch_list()
    if arch not in archs and f"compute_{major}{minor}" not in archs:
        return False, f"this PyTorch build has no kernels for {arch} (built for {archs})"
    return True, None


def select_device(requested: str = "auto") -> DeviceChoice:
    """'auto' and 'cuda' use CUDA when usable and fall back to CPU (with a reason) otherwise."""
    if requested not in DEVICE_CHOICES:
        raise ValueError(f"device must be one of {DEVICE_CHOICES}")
    if requested == "cpu":
        return DeviceChoice(requested, "cpu")
    ok, reason = cuda_usable()
    if ok:
        return DeviceChoice(requested, "cuda:0")
    return DeviceChoice(requested, "cpu", reason)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, PyTorch CPU and all CUDA devices."""
    torch = _torch()
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)  # also seeds every CUDA device
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.manual_seed_all(seed)


def device_metadata(device: str) -> dict:
    torch = _torch()
    meta = {"device": device, "torch_cuda_build": torch.version.cuda, "cudnn": None}
    if device.startswith("cuda"):
        idx = torch.device(device).index or 0
        props = torch.cuda.get_device_properties(idx)
        free, total = torch.cuda.mem_get_info(idx)
        meta.update({
            "name": props.name,
            "compute_capability": f"{props.major}.{props.minor}",
            "total_memory_mib": round(props.total_memory / 2**20),
            "free_memory_mib_at_start": round(free / 2**20),
            "arch_list": torch.cuda.get_arch_list(),
            "cudnn": torch.backends.cudnn.version(),
        })
    else:
        meta.update({"name": platform.processor() or platform.machine(),
                     "logical_cpus": os.cpu_count(), "torch_threads": torch.get_num_threads()})
    return meta


def reset_peak_memory(device: str) -> None:
    torch = _torch()
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(torch.device(device))


def peak_memory_mib(device: str) -> float | None:
    """Peak allocated GPU memory since the last reset; None on CPU (not tracked)."""
    torch = _torch()
    if not device.startswith("cuda"):
        return None
    return round(torch.cuda.max_memory_allocated(torch.device(device)) / 2**20, 3)


def environment_metadata() -> dict:
    torch = _torch()
    import pandas

    try:
        import optuna
        optuna_version = optuna.__version__
    except ImportError:  # pragma: no cover
        optuna_version = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pandas.__version__,
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "optuna": optuna_version,
    }


def runtime_report(requested: str = "auto") -> dict:
    determinism = configure_determinism()
    choice = select_device(requested)
    return {"device_choice": choice.to_dict(), "device": device_metadata(choice.device),
            "determinism": determinism, "environment": environment_metadata()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Report the Phase 3 runtime (device, determinism, versions).")
    parser.add_argument("--device", choices=DEVICE_CHOICES, default="auto")
    args = parser.parse_args(argv)
    try:
        report = runtime_report(args.device)
    except RuntimeConfigurationError as exc:
        print(f"RUNTIME ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())


def peak_rss_mib() -> float | None:
    """Peak resident memory of THIS PROCESS (all data, not just the model), if measurable."""
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            class PMC(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

            pmc = PMC()
            pmc.cb = ctypes.sizeof(PMC)
            k32 = ctypes.WinDLL("kernel32")
            psapi = ctypes.WinDLL("psapi")
            k32.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
            if not psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
                return None
            return round(pmc.PeakWorkingSetSize / 2**20, 1)
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(rss / (2**20 if sys.platform == "darwin" else 2**10), 1)
    except Exception:  # pragma: no cover - platform specific
        return None


class DeviceMismatchError(RuntimeConfigurationError):
    """An experiment registered for one device would run on another."""


def require_device(choice: DeviceChoice, expected: str) -> None:
    """Refuse instead of silently mixing devices (e.g. a CUDA experiment falling back to CPU)."""
    if choice.device != expected:
        raise DeviceMismatchError(
            f"experiment requires device {expected!r} but would run on {choice.device!r}"
            + (f" ({choice.fallback_reason})" if choice.fallback_reason else "")
            + "; results from different devices are separate experiments and are never merged")


def runtime_fingerprint(device: str) -> dict:
    """Device + runtime configuration that results depend on, and its canonical sha256.

    Part of every Phase 3 experiment fingerprint: a run whose runtime fingerprint
    differs (other device, GPU model, CUDA/cuDNN/torch build, thread count, or
    determinism settings) is a different experiment.
    """
    import hashlib

    meta = device_metadata(device)
    det = determinism_settings()
    env = environment_metadata()
    core = {
        "device": device,
        "device_name": meta.get("name"),
        "compute_capability": meta.get("compute_capability"),
        "torch": env["torch"], "torch_cuda_build": env["torch_cuda_build"], "cudnn": meta.get("cudnn"),
        "cpu_threads": det["cpu_threads"],
        "determinism": {k: det[k] for k in ("use_deterministic_algorithms", "cublas_workspace_config",
                                             "cudnn_deterministic", "cudnn_benchmark", "matmul_allow_tf32",
                                             "cudnn_allow_tf32", "precision")},
    }
    digest = hashlib.sha256(json.dumps(core, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"runtime": core, "runtime_sha256": digest, "environment": env}
