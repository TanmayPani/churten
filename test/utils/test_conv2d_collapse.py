"""`_collapse_kernels`: the valid-conv kernel schedule that walks an S×S grid to 1×1.

The training study that compares the collapse head against global-average-pool on
held-out accuracy is `bench_conv2d_collapse.py` — it reports numbers rather than
asserting them, so it is a bench, not a test.
"""

import pytest

from torchstrap.utils.nn.archs import _collapse_kernels


@pytest.mark.parametrize(
    "size,n_layers,expected",
    [
        # The reduction is distributed evenly across the layers: 9 -> 7 -> 5 -> 3 -> 1.
        (9, 4, [(3, 3)] * 4),
        (9, 2, [(5, 5)] * 2),
        (9, 8, [(2, 2)] * 8),  # 9 -> 8 -> ... -> 1
    ],
)
def test_lands_on_one_by_one(size, n_layers, expected):
    assert _collapse_kernels(size, n_layers) == expected


def test_rejects_non_positive_size():
    with pytest.raises(ValueError):
        _collapse_kernels(0, 4)
