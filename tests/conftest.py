import os
import shutil
from pathlib import Path

# Deterministic cuBLAS must be configured before anything initializes CUDA in this
# test process (some test modules probe the GPU at import). Setting it here mirrors
# what bri.seq.runtime.configure_determinism() does first thing in a real run; tests
# that check the refusal path remove it in their own subprocess.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import pytest

from bri.data.generate_synthetic import write_dataset


@pytest.fixture(scope="session")
def dataset_dir(tmp_path_factory) -> Path:
    """A small valid dataset generated once per test session (never committed)."""
    path = tmp_path_factory.mktemp("synthetic")
    write_dataset(path, customers=60, days=14, seed=42)
    return path


@pytest.fixture
def dataset_copy(dataset_dir, tmp_path) -> Path:
    """A writable copy of the valid dataset for corruption tests."""
    dest = tmp_path / "data"
    shutil.copytree(dataset_dir, dest)
    return dest


@pytest.fixture(scope="session")
def processed_dir(dataset_dir, tmp_path_factory) -> Path:
    """Phase 2 processed outputs built once from the session dataset."""
    from bri.prep.build import build

    path = tmp_path_factory.mktemp("processed")
    build(dataset_dir, path, val_days=3, test_days=3, min_history=1)
    return path
