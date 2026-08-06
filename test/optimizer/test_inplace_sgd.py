"""Numerical equivalence: the consolidated in-place batched ``sgd_step_`` vs
``torch.optim.SGD`` applied per-replica.

The SGD counterpart of ``test_inplace_adam.py``. The op operates on a SINGLE
consolidated ``(R, T)`` per-replica buffer (every parameter of a replica
concatenated along T), so the reference below builds per-parameter tensors and steps
each replica the classic way, then compares against the consolidated cat of the same
tensors. SGD is elementwise per coordinate, so any consistent consolidation order is
valid.

``test_distinct_hparams`` is the load-bearing one: it varies ``lr``/``momentum``/
``dampening``/``weight_decay`` per replica via ``linspace``. A per-replica gather
that has collapsed to a scalar — the characteristic bug of a hand-written batched
kernel — passes every uniform-hyperparameter test and fails only here.
"""

import pytest
import torch

from torchstrap.optimizer.sgd import sgd_step_


def _consolidate(tensors, R):
    """Pack a list of per-replica ``(R, *shape)`` tensors into one ``(R, T)``."""
    return torch.cat([t.reshape(R, -1) for t in tensors], dim=1).contiguous()


def _ref_per_replica(init, grads_per_step, lr, momentum, dampening, weight_decay,
                     nesterov, maximize):
    """Reference: one ``torch.optim.SGD`` per replica, stepped in a Python loop."""
    R = init[0].shape[0]
    params = [[t[r].clone().requires_grad_(True) for t in init] for r in range(R)]
    optims = [
        torch.optim.SGD(
            params[r],
            lr=float(lr[r]),
            momentum=float(momentum[r]),
            dampening=float(dampening[r]),
            weight_decay=float(weight_decay[r]),
            nesterov=nesterov,
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


def _run(device, lr, momentum, dampening, weight_decay, nesterov, maximize,
         num_steps=5, seed=42):
    R = 4
    shapes = [(5, 7), (7,), (3,)]
    dev = torch.device(device)

    g = torch.Generator(device=dev).manual_seed(seed)
    init = [torch.randn(R, *s, device=dev, generator=g) for s in shapes]
    grads_per_step = [
        [torch.randn(R, *s, device=dev, generator=g) for s in shapes]
        for _ in range(num_steps)
    ]

    ref = _ref_per_replica(
        init, grads_per_step, lr, momentum, dampening, weight_decay,
        nesterov, maximize,
    )

    has_momentum = bool(momentum.ne(0).any())
    p_new = _consolidate(init, R)
    mb_new = torch.zeros_like(p_new) if has_momentum else None
    s_new = torch.zeros(R, device=dev)
    mask = torch.ones(R, dtype=torch.bool, device=dev)

    for grads in grads_per_step:
        sgd_step_(
            p_new, _consolidate(grads, R), mb_new, s_new,
            lr, momentum, dampening, weight_decay, mask,
            nesterov=nesterov, maximize=maximize,
        )

    return _consolidate(ref, R), p_new


def _full(R, value, dev):
    return torch.full((R,), value, device=dev)


@pytest.mark.parametrize("momentum", [0.0, 0.9])
@pytest.mark.parametrize("weight_decay", [0.0, 1e-2])
@pytest.mark.parametrize("maximize", [False, True])
def test_matches_torch_optim_sgd(device, momentum, weight_decay, maximize):
    dev = torch.device(device)
    ref, got = _run(
        device,
        _full(4, 1e-2, dev), _full(4, momentum, dev), _full(4, 0.0, dev),
        _full(4, weight_decay, dev), nesterov=False, maximize=maximize,
    )
    torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("dampening", [0.0, 0.3])
def test_matches_torch_optim_sgd_dampening(device, dampening):
    dev = torch.device(device)
    ref, got = _run(
        device,
        _full(4, 1e-2, dev), _full(4, 0.9, dev), _full(4, dampening, dev),
        _full(4, 1e-2, dev), nesterov=False, maximize=False,
    )
    torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("weight_decay", [0.0, 1e-2])
def test_matches_torch_optim_sgd_nesterov(device, weight_decay):
    dev = torch.device(device)
    ref, got = _run(
        device,
        _full(4, 1e-2, dev), _full(4, 0.9, dev), _full(4, 0.0, dev),
        _full(4, weight_decay, dev), nesterov=True, maximize=False,
    )
    torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("nesterov", [False, True])
def test_distinct_hparams(device, nesterov):
    """Every hyperparameter genuinely varies per replica.

    Nesterov requires dampening == 0 (torch.optim.SGD enforces it), so dampening
    only varies in the non-nesterov case.
    """
    dev = torch.device(device)
    R = 4
    lr = torch.linspace(1e-3, 5e-2, R, device=dev)
    momentum = torch.linspace(0.5, 0.95, R, device=dev)
    dampening = (
        torch.zeros(R, device=dev) if nesterov
        else torch.linspace(0.0, 0.4, R, device=dev)
    )
    weight_decay = torch.linspace(0.0, 5e-2, R, device=dev)

    ref, got = _run(
        device, lr, momentum, dampening, weight_decay,
        nesterov=nesterov, maximize=False,
    )
    torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-6)


def test_zero_momentum_replica_inside_a_momentum_call(device):
    """A replica with momentum == 0 in a call that carries buffers is plain SGD.

    ATen cannot express this — its momentum is a single scalar, so momentum == 0 and
    "no momentum buffer" are the same condition. torchstrap decides buffer existence
    once per call and keeps the *value* per replica, so this configuration is
    reachable, and both kernels handle it arithmetically (`0 * buf + 1 * g == g`)
    rather than by branching, because the branch was measured to break CUDA
    bit-exactness. This is the check that the arithmetic really does reduce.
    """
    dev = torch.device(device)
    R = 4
    momentum = torch.tensor([0.0, 0.9, 0.0, 0.5], device=dev)
    ref, got = _run(
        device, _full(R, 1e-2, dev), momentum, _full(R, 0.0, dev),
        _full(R, 1e-2, dev), nesterov=False, maximize=False,
    )
    torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-6)
