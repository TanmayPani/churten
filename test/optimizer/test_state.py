"""Parity: torchstrap's fused batched Adam vs ``torch.optim.AdamW``.

New-API manual-loop spec. We build an ensemble of ``N`` *identical* replicas
(``init_randomness="same"`` clones one base net) and train them with the fused
``adam_step_`` via ``Adam.apply_gradient``. A single reference copy of the same net
is trained with ``torch.optim.AdamW`` on the same data. Because torchstrap's Adam
uses decoupled weight decay by default (== AdamW) and every replica sees identical
init + data + hyperparameters, all ``N`` replicas must track the AdamW reference
step for step.

Run: ``uv run python test/optimizer/test_state.py``
"""

from copy import deepcopy

import torch
from torch.nn import Sequential, Linear, ReLU
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.func import vmap, grad_and_value
from torch.optim import AdamW

from torchstrap.stateless import StatelessModule
from torchstrap.optimizer import Adam


HYPERS = dict(lr=1e-2, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2)


def test_ensemble_matches_torch_adamw():
    torch.manual_seed(0)
    N, num_steps = 3, 5

    net = Sequential(Linear(5, 10), ReLU(), Linear(10, 1))
    inputs = torch.rand(10, 5)
    targets = torch.cat([torch.zeros(5, 1), torch.ones(5, 1)])

    # --- torchstrap ensemble: N identical replicas of `net` ---
    module, state = StatelessModule.init(
        net,
        Adam,
        num_replicas=N,
        init_randomness="same",
        optimizer_kwargs=dict(
            lr=HYPERS["lr"],
            beta1=HYPERS["betas"][0],
            beta2=HYPERS["betas"][1],
            eps=HYPERS["eps"],
            weight_decay=HYPERS["weight_decay"],
        ),
    )

    def loss_fn(params, buffers, x, y):
        return binary_cross_entropy_with_logits(module(params, buffers, x), y)

    # inputs/targets are shared across replicas -> in_dims=None for them.
    grad_loss = vmap(grad_and_value(loss_fn, argnums=0), in_dims=(0, 0, None, None))

    # --- reference: the same net under torch.optim.AdamW ---
    ref = deepcopy(net)
    opt = AdamW(ref.parameters(), **HYPERS)

    for _ in range(num_steps):
        grads, _ = grad_loss(state.params_dict, state.buffers_dict, inputs, targets)
        Adam.apply_gradient(state, grads)

        opt.zero_grad(set_to_none=True)
        binary_cross_entropy_with_logits(ref(inputs), targets).backward()
        opt.step()

    ref_params = dict(ref.named_parameters())
    params = state.params_dict  # {name: (N, *shape)} aliased views
    for name, ref_p in ref_params.items():
        for r in range(N):
            torch.testing.assert_close(
                params[name][r], ref_p, rtol=1e-5, atol=1e-6,
                msg=f"replica {r} param {name!r} diverged from AdamW",
            )

    print(f"Pass — {N} replicas match torch.optim.AdamW over {num_steps} steps.")
