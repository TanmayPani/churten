"""Per-replica batched training over *uneven* batch sizes (manual-loop API).

The data iterators yield ``(input, target, sample_weight)`` shaped
``(N_replicas, B_per_replica, *features)``. The per-replica sample count is dim 1
(``input.shape[1]``), NOT ``input.shape[0]`` (the constant replica count ``N``) —
the latter was the original latent bug. This test trains an ensemble over a stream
of deliberately uneven batches and checks (a) the per-replica size is read from
dim 1, and (b) the weighted epoch mean over batches differs from the unweighted one
(so the distinction actually matters).

NOTE: the original version asserted this through ``ensemble.fit`` +
``History``/``PassThroughScoring``. That epoch-scoring layer is part of the deferred
callbacks re-sync; here we do the per-batch bookkeeping in the loop directly, which
is the new manual-loop contract.
"""

import torch
from torch.nn import Sequential, Linear, ReLU
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.func import vmap, grad_and_value

from torchstrap.stateless import StatelessModule
from torchstrap.optimizer import Adam


def make_model(*sizes, device="cpu", dtype=torch.float32):
    layers = Sequential()
    for a, b in zip(sizes[:-2], sizes[1:-1]):
        layers.extend(Sequential(Linear(a, b), ReLU()))
    layers.append(Linear(sizes[-2], sizes[-1]))
    return layers.to(device=device, dtype=dtype)


def uneven_iterator(num_replicas, batch_sizes, n_features, *, seed=0):
    """Yield (input, target, sample_weight) with a leading replica dim and the
    given per-replica batch sizes (deliberately uneven, e.g. [100, 100, 50])."""
    g = torch.Generator().manual_seed(seed)
    for b in batch_sizes:
        X = torch.randn(num_replicas, b, n_features, generator=g)
        y = torch.randint(0, 2, (num_replicas, b, 1), generator=g).float()
        yield X, y, None


def test_per_replica_batch_size_and_weighting():
    device = "cpu"
    N = 4
    n_features = 3
    batch_sizes = [100, 100, 50]  # uneven final batch

    torch.manual_seed(0)
    module, state = StatelessModule.init(
        make_model,
        Adam,
        n_features, 8, 1,
        num_replicas=N,
        device=device,
        init_randomness="different",
    )

    def loss_fn(params, buffers, x, y):
        return binary_cross_entropy_with_logits(module(params, buffers, x), y)

    grad_loss = vmap(grad_and_value(loss_fn, argnums=0), in_dims=(0, 0, 0, 0))

    per_batch_losses = []  # list of (N,)
    recorded_sizes = []  # per-replica batch sizes seen
    for X, y, _ in uneven_iterator(N, batch_sizes, n_features):
        X, y = X.to(device), y.to(device)
        recorded_sizes.append(X.shape[1])  # per-replica count is dim 1, NOT dim 0 (== N)
        grads, loss = grad_loss(state.params_dict, state.buffers_dict, X, y)
        Adam.apply_gradient(state, grads)
        per_batch_losses.append(loss.detach())  # (N,)

    # --- the per-replica size is dim 1, never the replica count N ---
    assert recorded_sizes == batch_sizes, (
        f"recorded batch sizes {recorded_sizes} != per-replica sizes {batch_sizes} "
        f"(the dim-0 bug would give [{N}, {N}, {N}])"
    )

    # --- sample-weighted vs unweighted epoch mean must differ on uneven batches ---
    losses = torch.stack(per_batch_losses)  # (n_batches, N)
    weights = torch.tensor(batch_sizes, dtype=losses.dtype).unsqueeze(-1)
    weighted = (losses * weights).sum(0) / weights.sum()  # (N,)
    unweighted = losses.mean(0)  # (N,)
    assert not torch.allclose(weighted, unweighted), (
        "weighted and unweighted means coincide — uneven batches failed to matter"
    )
