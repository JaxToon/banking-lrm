"""Phase 3 runtime: device selection, CPU fallback, seeding, determinism, metadata.

Skipped entirely when the optional `seq` extra (torch) is not installed. CUDA-only
checks are skipped when no usable GPU is present, so no GPU is ever required.
"""
import json
import os
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")

from bri.seq import runtime as R  # noqa: E402

HAS_CUDA = R.cuda_usable()[0]


def _run(code: str, env_extra: dict | None = None) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k != "CUBLAS_WORKSPACE_CONFIG"}
    env.update(env_extra or {})
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)


def test_cpu_request_never_uses_cuda():
    choice = R.select_device("cpu")
    assert choice.device == "cpu" and choice.fallback_reason is None


def test_cuda_unavailable_falls_back_to_cpu_with_a_reason(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    for requested in ("auto", "cuda"):
        choice = R.select_device(requested)
        assert choice.device == "cpu"
        assert "is_available() is False" in choice.fallback_reason


def test_build_without_kernels_for_the_gpu_falls_back(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda i=0: (99, 0))
    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: ["sm_80"])
    choice = R.select_device("auto")
    assert choice.device == "cpu" and "no kernels for sm_990" in choice.fallback_reason


def test_invalid_device_is_rejected():
    with pytest.raises(ValueError):
        R.select_device("tpu")


def test_seed_everything_is_reproducible():
    R.seed_everything(123)
    a = (torch.rand(5), R.np.random.rand(3), R.random.random())
    R.seed_everything(123)
    b = (torch.rand(5), R.np.random.rand(3), R.random.random())
    assert torch.equal(a[0], b[0]) and (a[1] == b[1]).all() and a[2] == b[2]


def test_cpu_metadata_and_environment_versions():
    meta = R.device_metadata("cpu")
    assert meta["device"] == "cpu" and meta["logical_cpus"] >= 1
    assert R.peak_memory_mib("cpu") is None
    env = R.environment_metadata()
    assert env["torch"].startswith("2.11.0")
    assert env["optuna"] == "4.9.0"


def test_determinism_settings_are_applied_and_recorded_in_a_fresh_process():
    r = _run("import json; from bri.seq import runtime as R; print(json.dumps(R.configure_determinism()))")
    assert r.returncode == 0, r.stderr
    s = json.loads(r.stdout)
    assert s["use_deterministic_algorithms"] is True
    assert s["cublas_workspace_config"] == ":4096:8"
    assert s["cudnn_benchmark"] is False and s["cudnn_deterministic"] is True
    assert s["matmul_allow_tf32"] is False and s["precision"] == "fp32"


def test_runtime_cli_reports_device_determinism_and_versions():
    r = subprocess.run([sys.executable, "-m", "bri.seq.runtime", "--device", "cpu"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    rep = json.loads(r.stdout)
    assert rep["device_choice"]["device"] == "cpu"
    assert rep["determinism"]["use_deterministic_algorithms"] is True
    assert {"python", "torch", "optuna", "numpy", "pandas"} <= set(rep["environment"])


@pytest.mark.cuda
@pytest.mark.skipif(not HAS_CUDA, reason="no usable CUDA GPU")
def test_configuring_after_cuda_initialization_is_refused():
    r = _run("import torch; torch.zeros(1, device='cuda'); from bri.seq import runtime as R; "
             "R.configure_determinism()")
    assert r.returncode != 0 and "initialized before configure_determinism" in r.stderr


@pytest.mark.cuda
@pytest.mark.skipif(not HAS_CUDA, reason="no usable CUDA GPU")
def test_cuda_smoke_metadata_and_deterministic_repeat():
    code = (
        "import json, torch; from bri.seq import runtime as R\n"
        "R.configure_determinism(); c = R.select_device('cuda'); R.reset_peak_memory(c.device)\n"
        "out = []\n"
        "for _ in range(2):\n"
        "    R.seed_everything(7); x = torch.randn(256, 64, device=c.device); w = torch.randn(64, 32, device=c.device)\n"
        "    y = torch.softmax(x @ w, dim=-1); y.sum().backward() if y.requires_grad else None; out.append(y.cpu())\n"
        "print(json.dumps({'device': c.device, 'equal': bool(torch.equal(out[0], out[1])),\n"
        "                  'meta': R.device_metadata(c.device), 'peak': R.peak_memory_mib(c.device)}))\n")
    r = _run(code)
    assert r.returncode == 0, r.stderr
    res = json.loads(r.stdout)
    assert res["device"] == "cuda:0" and res["equal"] is True
    major, minor = res["meta"]["compute_capability"].split(".")
    assert f"sm_{major}{minor}" in res["meta"]["arch_list"]
    assert res["meta"]["total_memory_mib"] > 0 and res["peak"] > 0
