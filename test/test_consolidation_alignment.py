"""The consolidated `(N, T)` width is padded up to `_CONSOLIDATION_ALIGNMENT`.

Why this exists: the fused kernels take ATen's vectorized `load_store` path only when
`n % kILP == 0` with `n = T - chunk_offset`. Every chunk offset is a multiple of
`kChunkSize`, so that condition is `T % 4 == 0` for *every* chunk of *every* row — a
`T` that is not a multiple of 4 puts the whole ensemble on the ragged
element-by-element path, not just the last chunk. Both shipped examples were in that
state (spirals `T=264705`, cnn `T=37827`).

The padding is only safe if the pad lanes are invisible to the offset tables and stay
exactly zero, which is what most of this file checks. If someone removes the padding,
`test_width_is_padded` fails; if someone lets a gradient reach the pad, so does
`test_pad_lanes_stay_zero_after_steps`.
"""

import torch
from torch.func import grad_and_value, vmap
from torch.nn import Linear, ReLU, Sequential

from torchstrap.optimizer import Adam
from torchstrap.state import _CONSOLIDATION_ALIGNMENT
from torchstrap.stateless import StatelessModule

# Linear(4, 8): 32 + 8, Linear(8, 1): 8 + 1  ->  49, which is 1 (mod 4). The point of
# the fixture is that the raw width is deliberately NOT already aligned.
RAW_T = 49


def make_model(device="cpu"):
    return Sequential(Linear(4, 8), ReLU(), Linear(8, 1)).to(device)


class BufferedModel(torch.nn.Module):
    """Two parameters and a 5-element buffer — both consolidations land unaligned."""

    def __init__(self, device="cpu"):
        super().__init__()
        self.lin = Linear(4, 8, device=device)
        self.register_buffer("scale", torch.ones(5, device=device))

    def forward(self, x):
        return self.lin(x) * self.scale[0]


def _init(factory, n=3, **kwargs):
    return StatelessModule.init(
        factory,
        Adam,
        num_replicas=n,
        init_randomness="different",
        optimizer_kwargs=dict(lr=1e-2, weight_decay=0.0),
        **kwargs,
    )


def _raw_width(meta):
    """Total per-replica numel actually covered by the offset table."""
    return sum(numel for _, _, _, numel in meta)


def test_width_is_padded(device):
    _, state = _init(make_model, device=device)

    assert _raw_width(state.param_meta) == RAW_T
    assert RAW_T % _CONSOLIDATION_ALIGNMENT != 0, "fixture no longer tests padding"

    padded = state.flat_params.shape[1]
    assert padded % _CONSOLIDATION_ALIGNMENT == 0
    assert padded == RAW_T + (-RAW_T % _CONSOLIDATION_ALIGNMENT)


def test_optimizer_and_grad_buffers_match_padded_width(device):
    """Every `(N, T)` buffer the kernel indexes must share the one padded width."""
    _, state = _init(make_model, device=device)
    T = state.flat_params.shape[1]

    assert state.flat_grads.shape[1] == T
    assert state.optimizer_state["exp_avgs"].shape[1] == T
    assert state.optimizer_state["exp_avg_sqs"].shape[1] == T


def test_param_views_round_trip(device):
    """The offset table never covers the pad, so the named views are unchanged."""
    module, state = _init(make_model, device=device)
    params = state.params_dict

    assert set(params) == {name for name, _, _, _ in state.param_meta}
    for name, shape, off, numel in state.param_meta:
        assert params[name].shape == shape
        assert params[name].numel() == state.batch_size[0] * numel
        assert off + numel <= RAW_T

    # Aliased, not copied: writing a view writes the consolidated buffer.
    name, shape, off, numel = state.param_meta[0]
    params[name].fill_(7.0)
    assert torch.equal(
        state.flat_params[:, off : off + numel],
        torch.full((state.batch_size[0], numel), 7.0, device=state.flat_params.device),
    )


def test_empty_buffers_stay_zero_width(device):
    """`(N, 0)` must stay `(N, 0)` — an empty consolidation is deliberately not padded."""
    _, state = _init(make_model, device=device)

    assert state.buffer_meta == []
    assert state.buffers.shape == (3, 0)


def test_buffers_are_padded_too(device):
    _, state = _init(BufferedModel, device=device)

    assert _raw_width(state.buffer_meta) == 5
    assert state.buffers.shape[1] == 8
    assert state.buffers.shape[1] % _CONSOLIDATION_ALIGNMENT == 0
    # The real buffer values survive the pad.
    assert torch.equal(state.buffers_dict["scale"], torch.ones(3, 5, device=device))


def test_pad_lanes_stay_zero_after_steps(device):
    """The load-bearing invariant.

    Nothing writes a gradient into the pad (`_consolidated_grads` only covers the
    offset table), and Adam on a zero param with a zero grad has a delta of exactly
    zero. So after training the pad must still be bit-zero in every `(N, T)` buffer —
    if it is not, the pad has become live state and could leak a NaN.
    """
    torch.manual_seed(0)
    n = 3
    module, state = _init(make_model, n=n, device=device)
    T = state.flat_params.shape[1]
    assert T > RAW_T, "fixture no longer has pad lanes to check"

    x = torch.randn(n, 16, 4, device=device)
    y = torch.randn(n, 16, 1, device=device)

    def loss_fn(params, buffers, x, y):
        return ((module(params, buffers, x) - y) ** 2).mean()

    grad_loss = vmap(grad_and_value(loss_fn, argnums=0), in_dims=(0, 0, 0, 0))

    for _ in range(5):
        grads, _ = grad_loss(state.params_dict, state.buffers_dict, x, y)
        Adam.apply_gradient(state, grads)

    # Trained at all?
    assert state.optimizer_state["state_steps"].max() == 5

    zeros = torch.zeros(n, T - RAW_T, device=state.flat_params.device)
    for name, buf in (
        ("params", state.flat_params),
        ("grads", state.flat_grads),
        ("exp_avgs", state.optimizer_state["exp_avgs"]),
        ("exp_avg_sqs", state.optimizer_state["exp_avg_sqs"]),
    ):
        assert torch.equal(buf[:, RAW_T:], zeros), f"{name} pad lanes are not zero"
