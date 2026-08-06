"""Shared fixtures.

Nearly every kernel test has to run on both dispatch keys, so `device` is a
parametrized fixture rather than a loop inside each test: pytest then reports one
result per device (`... [cpu]`, `... [cuda]`) and skips the CUDA half cleanly on a
machine without a GPU, instead of silently testing less than you think.

Anything that writes to disk must take pytest's `tmp_path` — nothing in the suite
may leave files in the repo.
"""

import pytest
import torch

HAS_CUDA = torch.cuda.is_available()

requires_cuda = pytest.mark.skipif(not HAS_CUDA, reason="no CUDA device available")


@pytest.fixture(params=["cpu", pytest.param("cuda", marks=requires_cuda)])
def device(request) -> str:
    """Each dependent test runs once per available device."""
    return request.param


@pytest.fixture
def torch_device(device) -> torch.device:
    return torch.device(device)
