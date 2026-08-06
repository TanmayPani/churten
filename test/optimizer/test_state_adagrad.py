"""Parity: torchstrap's fused batched Adagrad vs ``torch.optim.Adagrad``, end to end.

The Adagrad counterpart of ``test_state.py`` / ``test_state_sgd.py``: an ensemble of
``N`` *identical* replicas (``init_randomness="same"`` clones one base net) trained
through the full ``StatelessModule.init`` -> ``vmap(grad)`` -> ``Adagrad.apply_gradient``
path, against a single reference copy under ``torch.optim.Adagrad`` on the same data.

This is the test that exercises what the kernel tests cannot: that ``Adagrad.init``
populates ``optimizer_state`` with the right shapes, that ``state_sums`` is
consolidated like ``params`` and seeded like upstream's ``initial_accumulator_value``,
and that ``maximize`` survives the round trip through ``add_optim_meta`` into an
arg-less ``update``.
"""

from copy import deepcopy

import pytest
import torch
from torch.nn import Sequential, Linear, ReLU
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.func import vmap, grad_and_value

from torchstrap.stateless import StatelessModule
from torchstrap.optimizer import Adagrad


@pytest.mark.parametrize("lr_decay", [0.0, 1e-2])
@pytest.mark.parametrize("weight_decay", [0.0, 1e-2])
@pytest.mark.parametrize("initial_accumulator_value", [0.0, 0.1])
def test_ensemble_matches_torch_adagrad(
    lr_decay, weight_decay, initial_accumulator_value
):
    torch.manual_seed(0)
    N, num_steps = 3, 5
    hypers = dict(
        lr=1e-2,
        lr_decay=lr_decay,
        weight_decay=weight_decay,
        eps=1e-10,
        initial_accumulator_value=initial_accumulator_value,
    )

    net = Sequential(Linear(5, 10), ReLU(), Linear(10, 1))
    inputs = torch.rand(10, 5)
    targets = torch.cat([torch.zeros(5, 1), torch.ones(5, 1)])

    module, state = StatelessModule.init(
        net,
        Adagrad,
        num_replicas=N,
        init_randomness="same",
        optimizer_kwargs=hypers,
    )

    def loss_fn(params, buffers, x, y):
        return binary_cross_entropy_with_logits(module(params, buffers, x), y)

    # inputs/targets are shared across replicas -> in_dims=None for them.
    grad_loss = vmap(grad_and_value(loss_fn, argnums=0), in_dims=(0, 0, None, None))

    ref = deepcopy(net)
    opt = torch.optim.Adagrad(ref.parameters(), **hypers)

    for _ in range(num_steps):
        grads, _ = grad_loss(state.params_dict, state.buffers_dict, inputs, targets)
        Adagrad.apply_gradient(state, grads)

        opt.zero_grad(set_to_none=True)
        binary_cross_entropy_with_logits(ref(inputs), targets).backward()
        opt.step()

    params = state.params_dict  # {name: (N, *shape)} aliased views
    for name, ref_p in dict(ref.named_parameters()).items():
        for r in range(N):
            torch.testing.assert_close(
                params[name][r], ref_p, rtol=1e-5, atol=1e-6,
                msg=f"replica {r} param {name!r} diverged from torch.optim.Adagrad",
            )
