import shutil
from pathlib import Path

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
