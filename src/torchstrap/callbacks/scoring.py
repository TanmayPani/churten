"""Epoch-level score aggregation."""

import torch
from torch import Tensor

from beartype.typing import Sequence


__all__ = ["EpochScore"]


class EpochScore:
    """Aggregate per-batch ``(N,)`` scores into one ``(N,)`` epoch score.

    A plain callable: collect each batch's per-replica score during the epoch
    (e.g. ``batch_losses.append(loss.detach())``), then call this once at epoch
    end. With ``batch_sizes`` the batches are weighted by their (per-replica)
    sample count; without it they are averaged equally. The result is a ``(N,)``
    tensor on the same device as the inputs — feed it straight to ``Checkpoint``,
    ``EarlyStopping``, or ``LRScheduler``.

    Stateless: best-so-far tracking lives in the consumers, not here.
    """

    def __call__(
        self,
        batch_values: Sequence[Tensor] | Tensor,
        batch_sizes: Sequence[Tensor] | Sequence[int] | Tensor | None = None,
    ) -> Tensor:
        scores = (
            batch_values
            if isinstance(batch_values, Tensor)
            else torch.stack(list(batch_values))
        )  # (B, N)
        if batch_sizes is None:
            return scores.mean(dim=0)

        # `batch_sizes` weights each batch. Two layouts are accepted:
        #   * shared per-batch scalars -> `(B,)`/`(B,1)`, broadcast over replicas
        #     (e.g. plain sample counts), or
        #   * per-replica masses -> `(B, N)` (e.g. each batch's per-replica `sum(w)`),
        #     which yields the EXACT per-replica weighted mean
        #     `sum_b sum_i(w*score) / sum_b sum_i(w)` when the per-batch scores are
        #     themselves per-replica weighted means.
        if isinstance(batch_sizes, Tensor):
            weights = batch_sizes
        else:
            items = list(batch_sizes)
            weights = (
                torch.stack(items)
                if items and isinstance(items[0], Tensor)
                else torch.as_tensor(items, dtype=scores.dtype, device=scores.device)
            )
        weights = weights.reshape(scores.shape[0], -1).to(scores.dtype)  # (B,1) or (B,N)
        return (scores * weights).sum(dim=0) / weights.sum(dim=0)
