"""End-to-end smoke test of the core State + Adam path on CPU.

`StatelessModule.init` -> `Adam.init` -> `vmap(grad)` -> `Adam.apply_gradient`.
Cheap, and the first thing to fail on a blocking bug in `state.py` (consolidate vs
consolidated, the `flat_params` view aliasing).
"""

import torch
from torch.func import vmap, grad_and_value
from torch.nn import Linear, ReLU, Sequential

from torchstrap.stateless import StatelessModule
from torchstrap.optimizer import Adam


def make_model():
    return Sequential(Linear(4, 8), ReLU(), Linear(8, 1))


def test_state_adam_smoke():
    torch.manual_seed(0)
    N = 3
    module, state = StatelessModule.init(
        make_model,
        Adam,
        num_replicas=N,
        init_randomness="different",
        optimizer_kwargs=dict(lr=1e-2, weight_decay=0.0),
    )


    # per-replica data (N, B, features)
    x = torch.randn(N, 16, 4)
    y = torch.randn(N, 16, 1)

    def loss_fn(params, buffers, x, y):
        pred = module(params, buffers, x)
        return ((pred - y) ** 2).mean()

    grad_loss = vmap(grad_and_value(loss_fn, argnums=0), in_dims=(0, 0, 0, 0))

    before = state.flat_params.clone()
    for i in range(5):
        grads, loss = grad_loss(state.params_dict, state.buffers_dict, x, y)
        Adam.apply_gradient(state, grads)

    moved = (state.flat_params - before).abs().max().item()
    assert moved > 0, "params did not move!"
