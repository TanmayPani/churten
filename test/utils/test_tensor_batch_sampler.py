"""Equivalence + speed for the rewritten ``TensorBatchSampler.__iter__``.

The new ``__iter__`` yields contiguous ``narrow`` views; this asserts the batches
are byte-identical to the old SequentialSampler/BatchSampler/advanced-index path
across drop_last and divisibility.

The speed claim lives in ``bench()``, which is not a test — it is a plain
function so it stays out of collection; call it from a REPL or a bench script.
"""
import time

import pytest
import torch
from torch.utils.data import BatchSampler, SequentialSampler

from torchstrap.utils.data import TensorBatchSampler


def _old_iter(source, batch_size, batch_dim, drop_last):
    """The pre-rewrite reference: list-of-ints batches + advanced indexing."""
    batch_sampler = BatchSampler(
        SequentialSampler(torch.arange(source.shape[batch_dim]).to("meta")),
        batch_size=batch_size,
        drop_last=drop_last,
    )
    for batch_indices in batch_sampler:
        index_slice = (slice(None),) * batch_dim
        yield source[*index_slice, batch_indices, ...]


@pytest.mark.parametrize("batch_dim", [0, 1])
@pytest.mark.parametrize("drop_last", [False, True])
@pytest.mark.parametrize("n", [1000, 1003], ids=["divisible", "remainder"])
def test_matches_advanced_indexing(n, batch_dim, drop_last,
                                   num_replicas=4, batch_size=128):
    if batch_dim == 1:
        source = torch.stack([torch.randperm(n) for _ in range(num_replicas)])
    else:
        source = torch.randperm(n)

    sampler = TensorBatchSampler(
        source, batch_size=batch_size, batch_dim=batch_dim, drop_last=drop_last
    )
    new_batches = list(sampler)
    old_batches = list(_old_iter(source, batch_size, batch_dim, drop_last))

    assert len(new_batches) == len(old_batches) == len(sampler), (
        len(new_batches), len(old_batches), len(sampler),
    )
    for a, b in zip(new_batches, old_batches):
        assert torch.equal(a, b), "batch mismatch"



def bench(num_replicas=64, n=1_500_000, batch_size=512, iters=3):
    source = torch.stack([torch.randperm(n) for _ in range(num_replicas)])
    sampler = TensorBatchSampler(source, batch_size=batch_size, batch_dim=1)

    def drain(it):
        s = 0
        for b in it:
            s += b.shape[1]
        return s

    # warmup
    drain(iter(sampler))
    drain(_old_iter(source, batch_size, 1, False))

    t0 = time.perf_counter()
    for _ in range(iters):
        drain(iter(sampler))
    t_new = (time.perf_counter() - t0) / iters

    t0 = time.perf_counter()
    for _ in range(iters):
        drain(_old_iter(source, batch_size, 1, False))
    t_old = (time.perf_counter() - t0) / iters

    print(
        f"\n  one full pass over (N={num_replicas}, n={n}) @ bs={batch_size}: "
        f"new={t_new*1e3:.1f} ms  old={t_old*1e3:.1f} ms  "
        f"speedup={t_old/t_new:.1f}x"
    )
