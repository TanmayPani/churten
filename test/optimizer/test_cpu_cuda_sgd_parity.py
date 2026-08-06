"""CPU-vs-CUDA parity for the consolidated ``sgd_step_``, plus ``opcheck``.

The SGD counterpart of ``test_cpu_cuda_parity.py`` (kept as its own file so each
operator's registration and cross-device checks stay independently readable).

Two things make this the sharpest test of a hand-written kernel:

  * The ``distinct_hparams`` sweep gives every replica its own
    ``lr``/``momentum``/``dampening``/``weight_decay``. Every other test uses uniform
    values, which cannot tell a genuine per-replica gather apart from a value that
    collapsed to a broadcast scalar.
  * ``shapes`` is run in two flavours so that ``T`` is and is not a multiple of
    ``kILP``, covering both ATen's vectorized and its ragged access pattern.

For the much stricter claim — that each kernel is *bit*-identical to ATen's own
fused SGD **on its own device** — see ``test_aten_sgd_parity.py``.

CPU-vs-CUDA is deliberately *not* bit-exact and cannot be made so: torchstrap ports
ATen's CPU formulation on CPU and ATen's CUDA formulation on CUDA, and upstream's two
differ in where they contract. SGD's two formulations are much closer than Adam's
(there is no double-precision bias correction to diverge over — only fma placement),
so this file holds a single tight tolerance rather than Adam's split one.
"""

import pytest
import torch

from torchstrap.optimizer.sgd import sgd_step_  # noqa: F401 - defines the op


def _check_opcheck(device):
    """torch.library.opcheck — schema/alias annotations, the fake kernel, and that
    the declared mutations match what the kernel actually writes."""
    R, T = 3, 8
    U = lambda x: torch.full((R,), x, device=device)
    args = (
        torch.randn(R, T, device=device),
        torch.randn(R, T, device=device),
        torch.zeros(R, T, device=device),
        torch.zeros(R, device=device),
        U(1e-2), U(0.9), U(0.0), U(1e-2),
        torch.tensor([True, False, True], device=device),
    )
    kwargs = dict(nesterov=False, maximize=False)
    torch.library.opcheck(torch.ops.torchstrap.sgd_step_.default, args, kwargs)


def _check_opcheck_no_momentum(device):
    """The depth-2 instantiation: `momentum_buffers` absent."""
    R, T = 3, 8
    U = lambda x: torch.full((R,), x, device=device)
    args = (
        torch.randn(R, T, device=device),
        torch.randn(R, T, device=device),
        None,
        torch.zeros(R, device=device),
        U(1e-2), U(0.0), U(0.0), U(1e-2),
        torch.tensor([True, False, True], device=device),
    )
    kwargs = dict(nesterov=False, maximize=False)
    torch.library.opcheck(torch.ops.torchstrap.sgd_step_.default, args, kwargs)


def _consolidate(tensors, R):
    return torch.cat([t.reshape(R, -1) for t in tensors], dim=1).contiguous()


def _make_state(R, shapes, device, has_momentum, dtype=torch.float32, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    rnd = lambda s: torch.randn(R, *s, device=device, dtype=dtype, generator=g)
    params = _consolidate([rnd(s) for s in shapes], R)
    grads = _consolidate([rnd(s) for s in shapes], R)
    momentum_buffers = _consolidate([rnd(s) for s in shapes], R) if has_momentum else None
    # Start mid-training, so `is_first_step` is False and the momentum recurrence is
    # actually exercised rather than short-circuited to `buf = grad`.
    state_steps = torch.full((R,), 1.0, device=device, dtype=dtype)
    return params, grads, momentum_buffers, state_steps


def _clone_to(args, device):
    to = lambda t: None if t is None else t.detach().to(device).contiguous()
    return tuple(to(t) for t in args)


def _compare(name, a, b, atol, rtol):
    b_cpu = b.cpu()
    if not torch.allclose(a, b_cpu, atol=atol, rtol=rtol):
        diff = (a - b_cpu).abs().max().item()
        raise AssertionError(
            f"{name} cpu vs cuda diverged: max abs diff {diff:.2e} "
            f"(atol={atol:.0e}, rtol={rtol:.0e})"
        )


def _hparams(R, distinct, has_momentum, nesterov):
    if not distinct:
        return (
            torch.full((R,), 1e-2),
            torch.full((R,), 0.9 if has_momentum else 0.0),
            torch.zeros(R),
            torch.full((R,), 1e-2),
        )
    return (
        torch.linspace(1e-3, 5e-2, R),                                   # lr
        torch.linspace(0.5, 0.95, R) if has_momentum else torch.zeros(R),  # momentum
        torch.zeros(R) if nesterov else torch.linspace(0.0, 0.4, R),     # dampening
        torch.linspace(0.0, 1e-1, R),                                    # weight_decay
    )


def _case(R, shapes, has_momentum, nesterov, maximize, num_steps=3,
          distinct_hparams=False, atol=1e-5, rtol=1e-5):
    cpu_args = _make_state(R, shapes, torch.device("cpu"), has_momentum, seed=42)
    cuda_args = _clone_to(cpu_args, torch.device("cuda"))

    lr, mom, damp, wd = _hparams(R, distinct_hparams, has_momentum, nesterov)
    mask = torch.ones(R, dtype=torch.bool)
    cu = lambda t: t.cuda()
    lr_cu, mom_cu, damp_cu, wd_cu, mask_cu = (
        cu(lr), cu(mom), cu(damp), cu(wd), cu(mask)
    )

    flags = dict(nesterov=nesterov, maximize=maximize)

    for step in range(num_steps):
        gen = torch.Generator(device="cpu").manual_seed(100 + step)
        cpu_args[1].normal_(generator=gen)
        cuda_args[1].copy_(cpu_args[1], non_blocking=False)

        sgd_step_(*cpu_args, lr, mom, damp, wd, mask, **flags)
        sgd_step_(*cuda_args, lr_cu, mom_cu, damp_cu, wd_cu, mask_cu, **flags)

    _compare("params", cpu_args[0], cuda_args[0], atol, rtol)
    if has_momentum:
        _compare("momentum_buffers", cpu_args[2], cuda_args[2], atol, rtol)
    _compare("state_steps", cpu_args[3], cuda_args[3], atol, rtol)


def test_opcheck(device):
    _check_opcheck(torch.device(device))


def test_opcheck_no_momentum(device):
    _check_opcheck_no_momentum(torch.device(device))


# T = 45 exercises ATen's ragged path, T = 52 its kILP-wide vectorized one.
_SHAPE_SETS = {
    "T=45": [(5, 7), (7,), (3,)],
    "T=52": [(5, 8), (8,), (4,)],
}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device available")
@pytest.mark.parametrize("tag", list(_SHAPE_SETS))
@pytest.mark.parametrize("distinct_hparams", [False, True])
@pytest.mark.parametrize("has_momentum", [False, True])
@pytest.mark.parametrize("maximize", [False, True])
def test_cpu_matches_cuda(tag, distinct_hparams, has_momentum, maximize):
    _case(
        4, _SHAPE_SETS[tag], has_momentum, nesterov=False, maximize=maximize,
        distinct_hparams=distinct_hparams,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device available")
@pytest.mark.parametrize("tag", list(_SHAPE_SETS))
@pytest.mark.parametrize("distinct_hparams", [False, True])
def test_cpu_matches_cuda_nesterov(tag, distinct_hparams):
    _case(
        4, _SHAPE_SETS[tag], has_momentum=True, nesterov=True, maximize=False,
        distinct_hparams=distinct_hparams,
    )
