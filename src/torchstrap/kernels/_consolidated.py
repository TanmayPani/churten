"""Shared helpers for the fused operators' `register_vmap` rules.

Only the three reshaping helpers live here. Each operator keeps its **own**
explicit `register_vmap` function with a real, named argument list — the mapping
from `in_dims[i]` to an argument is the part a reader needs to check against the
schema, and hiding it behind a generic loop would make it unreviewable.

The shared premise: treat the outer vmap dim AS the replica dim. The deferred
composition (R > 1 inside vmap) is not handled — callers pass R == 1 single-replica
state inside the vmap'd function, so the consolidated buffers arrive as `(B, 1, T)`
and the hyperparameters as `(B, 1)`, and the singleton inner dim is squeezed to give
the kernels a regular `(R=B, T)` layout.

`movedim` for a bdim other than 0 yields a **non-contiguous** view. Both C++ kernels
handle that with a strided path, since the dispatcher goes straight to them and
there is no wrapper left to bail out to a reference implementation (see
`test_inplace_adam_vmap.case_strided`, the only coverage of either strided path).
"""

import torch

__all__ = ["lead_squeeze", "hyp", "hyp_mask"]


def lead_squeeze(t: torch.Tensor, d: int | None) -> torch.Tensor:
    """Move the batch dim to the front and drop the singleton replica dim."""
    if d is None:
        return t
    t = t.movedim(d, 0)
    if t.dim() >= 2 and t.shape[1] == 1:
        t = t.squeeze(1)
    return t


def hyp(t: torch.Tensor, d: int | None, batch_size: int) -> torch.Tensor:
    """`lead_squeeze` for an `(R,)` hyperparameter, broadcasting an unbatched scalar."""
    if d is not None:
        return lead_squeeze(t, d)
    return t.expand(batch_size) if t.dim() == 0 else t


def hyp_mask(t: torch.Tensor, d: int | None, batch_size: int) -> torch.Tensor:
    """`hyp` for the `(R,)` active mask, which the kernels read as bool."""
    return hyp(t, d, batch_size).to(dtype=torch.bool)
