"""Parity: torchstrap's fused batched SGD vs ``torch.optim.SGD``, end to end.

The SGD counterpart of ``test_state.py``: an ensemble of ``N`` *identical* replicas
(``init_randomness="same"`` clones one base net) trained through the full
``StatelessModule.init`` -> ``vmap(grad)`` -> ``SGD.apply_gradient`` path, against a
single reference copy under ``torch.optim.SGD`` on the same data. Every replica sees
identical init, data and hyperparameters, so all ``N`` must track the reference step
for step.

This is the test that exercises the parts the kernel tests cannot: that ``SGD.init``
populates ``optimizer_state`` with the right shapes, that the momentum buffer is
consolidated like ``params``, and that the static ``nesterov``/``maximize`` flags
survive the round trip through ``add_optim_meta`` into an arg-less ``update``.
"""

from copy import deepcopy

import pytest
import torch
from torch.nn import Sequential, Linear, ReLU
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.func import vmap, grad_and_value

from torchstrap.stateless import StatelessModule
from torchstrap.optimizer import SGD


@pytest.mark.parametrize(
    "momentum,dampening,nesterov",
    [
        (0.0, 0.0, False),   # plain SGD — no momentum buffer is allocated at all
        (0.9, 0.0, False),
        (0.9, 0.3, False),
        (0.9, 0.0, True),
    ],
)
@pytest.mark.parametrize("weight_decay", [0.0, 1e-2])
def test_ensemble_matches_torch_sgd(momentum, dampening, nesterov, weight_decay):
    torch.manual_seed(0)
    N, num_steps = 3, 5
    hypers = dict(
        lr=1e-2,
        momentum=momentum,
        dampening=dampening,
        weight_decay=weight_decay,
        nesterov=nesterov,
    )

    net = Sequential(Linear(5, 10), ReLU(), Linear(10, 1))
    inputs = torch.rand(10, 5)
    targets = torch.cat([torch.zeros(5, 1), torch.ones(5, 1)])

    module, state = StatelessModule.init(
        net,
        SGD,
        num_replicas=N,
        init_randomness="same",
        optimizer_kwargs=hypers,
    )

    # The buffer exists iff momentum is non-zero — ATen's depth 3 vs 2.
    assert ("momentum_buffers" in state.optimizer_state.keys()) == (momentum != 0.0)

    def loss_fn(params, buffers, x, y):
        return binary_cross_entropy_with_logits(module(params, buffers, x), y)

    # inputs/targets are shared across replicas -> in_dims=None for them.
    grad_loss = vmap(grad_and_value(loss_fn, argnums=0), in_dims=(0, 0, None, None))

    ref = deepcopy(net)
    opt = torch.optim.SGD(ref.parameters(), **hypers)

    for _ in range(num_steps):
        grads, _ = grad_loss(state.params_dict, state.buffers_dict, inputs, targets)
        SGD.apply_gradient(state, grads)

        opt.zero_grad(set_to_none=True)
        binary_cross_entropy_with_logits(ref(inputs), targets).backward()
        opt.step()

    params = state.params_dict  # {name: (N, *shape)} aliased views
    for name, ref_p in dict(ref.named_parameters()).items():
        for r in range(N):
            torch.testing.assert_close(
                params[name][r], ref_p, rtol=1e-5, atol=1e-6,
                msg=f"replica {r} param {name!r} diverged from torch.optim.SGD",
            )


def test_nesterov_requires_momentum():
    """`torch.optim.SGD`'s validation, applied across the ensemble."""
    net = Sequential(Linear(4, 2))
    with pytest.raises(ValueError, match="Nesterov"):
        StatelessModule.init(
            net, SGD, num_replicas=2, init_randomness="same",
            optimizer_kwargs=dict(lr=1e-2, momentum=0.0, nesterov=True),
        )
    with pytest.raises(ValueError, match="Nesterov"):
        StatelessModule.init(
            net, SGD, num_replicas=2, init_randomness="same",
            optimizer_kwargs=dict(lr=1e-2, momentum=0.9, dampening=0.1, nesterov=True),
        )
