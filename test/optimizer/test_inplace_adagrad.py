"""Numerical equivalence: the consolidated in-place batched ``adagrad_step_`` vs
``torch.optim.Adagrad`` applied per-replica.

The Adagrad counterpart of ``test_inplace_adam.py`` / ``test_inplace_sgd.py``. The op
operates on a SINGLE consolidated ``(R, T)`` per-replica buffer, so the reference
below builds per-parameter tensors and steps each replica the classic way, then
compares against the consolidated cat of the same tensors. Adagrad is elementwise per
coordinate, so any consistent consolidation order is valid.

``test_distinct_hparams`` is the load-bearing one: it varies
``lr``/``lr_decay``/``weight_decay``/``eps`` per replica via ``linspace``. A
per-replica gather that has collapsed to a scalar — the characteristic bug of a
hand-written batched kernel — passes every uniform-hyperparameter test and fails only
here. ``lr_decay`` matters twice over, since it is what couples the update to the
per-replica step counter.
"""

import pytest
import torch

from torchstrap.optimizer.adagrad import adagrad_step_


def _consolidate(tensors, R):
    """Pack a list of per-replica ``(R, *shape)`` tensors into one ``(R, T)``."""
    return torch.cat([t.reshape(R, -1) for t in tensors], dim=1).contiguous()


def _ref_per_replica(init, grads_per_step, lr, lr_decay, weight_decay, eps, maximize):
    """Reference: one ``torch.optim.Adagrad`` per replica, stepped in a Python loop."""
    R = init[0].shape[0]
    params = [[t[r].clone().requires_grad_(True) for t in init] for r in range(R)]
    optims = [
        torch.optim.Adagrad(
            params[r],
            lr=float(lr[r]),
            lr_decay=float(lr_decay[r]),
            weight_decay=float(weight_decay[r]),
            eps=float(eps[r]),
            maximize=maximize,
        )
        for r in range(R)
    ]
    for grads in grads_per_step:
        for r in range(R):
            for p, g in zip(params[r], grads):
                p.grad = g[r].clone()
            optims[r].step()
    return [
        torch.stack([params[r][i].detach() for r in range(R)])
        for i in range(len(init))
    ]


def _run(device, lr, lr_decay, weight_decay, eps, maximize, num_steps=5, seed=42):
    R = 4
    shapes = [(5, 7), (7,), (3,)]
    dev = torch.device(device)

    g = torch.Generator(device=dev).manual_seed(seed)
    init = [torch.randn(R, *s, device=dev, generator=g) for s in shapes]
    grads_per_step = [
        [torch.randn(R, *s, device=dev, generator=g) for s in shapes]
        for _ in range(num_steps)
    ]

    ref = _ref_per_replica(init, grads_per_step, lr, lr_decay, weight_decay, eps, maximize)

    p_new = _consolidate(init, R)
    # torch.optim.Adagrad seeds the accumulator at initial_accumulator_value=0.
    s_new = torch.zeros_like(p_new)
    steps = torch.zeros(R, device=dev)
    mask = torch.ones(R, dtype=torch.bool, device=dev)

    for grads in grads_per_step:
        adagrad_step_(
            p_new, _consolidate(grads, R), s_new, steps,
            lr, lr_decay, weight_decay, eps, mask, maximize=maximize,
        )

    return _consolidate(ref, R), p_new


def _full(R, value, dev):
    return torch.full((R,), value, device=dev)


@pytest.mark.parametrize("lr_decay", [0.0, 1e-2])
@pytest.mark.parametrize("weight_decay", [0.0, 1e-2])
@pytest.mark.parametrize("maximize", [False, True])
def test_matches_torch_optim_adagrad(device, lr_decay, weight_decay, maximize):
    dev = torch.device(device)
    R = 4
    ref, got = _run(
        device, _full(R, 1e-2, dev), _full(R, lr_decay, dev),
        _full(R, weight_decay, dev), _full(R, 1e-10, dev), maximize,
    )
    torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-6)


def test_distinct_hparams(device):
    dev = torch.device(device)
    R = 4
    ref, got = _run(
        device,
        torch.linspace(1e-3, 5e-2, R, device=dev),    # lr
        torch.linspace(0.0, 5e-2, R, device=dev),     # lr_decay
        torch.linspace(0.0, 5e-2, R, device=dev),     # weight_decay
        torch.linspace(1e-10, 1e-6, R, device=dev),   # eps
        maximize=False,
    )
    torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-6)


def test_initial_accumulator_value(device):
    """`Adagrad.init` seeds `state_sums`, matching `torch.optim.Adagrad`'s option."""
    from torchstrap.optimizer import Adagrad
    from torchstrap.stateless import StatelessModule
    from torch.nn import Linear, ReLU, Sequential

    dev = torch.device(device)
    _, state = StatelessModule.init(
        lambda device=dev: Sequential(Linear(4, 8), ReLU(), Linear(8, 1)).to(device),
        Adagrad,
        num_replicas=3,
        init_randomness="different",
        optimizer_kwargs=dict(lr=1e-2, initial_accumulator_value=0.1),
        device=dev,
    )
    sums = state.optimizer_state["state_sums"]
    torch.testing.assert_close(sums, torch.full_like(sums, 0.1))
