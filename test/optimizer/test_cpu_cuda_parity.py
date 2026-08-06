"""CPU-vs-CUDA parity for the consolidated ``adam_step_``.

Builds the same random ``(R, T)`` consolidated state on both devices, runs the
same steps through each, and asserts they agree element-wise. Also runs
``torch.library.opcheck``, the tutorial's recommended registration test.

Two things make this the sharpest test of a hand-written kernel:

  * The ``distinct_hparams`` sweep gives every replica its own
    ``lr``/``beta1``/``beta2``/``eps``/``weight_decay``. Every other test uses
    uniform values, which cannot tell a genuine per-replica gather apart from a
    value that collapsed to a broadcast scalar.
  * ``shapes`` is run in two flavours so that ``T`` is and is not a multiple of
    ``kILP``, covering both ATen's vectorized and its ragged access pattern.

For the much stricter claim -- that each kernel is *bit*-identical to ATen's own
fused Adam **on its own device** -- see ``test_aten_fused_parity.py``.

CPU-vs-CUDA is deliberately *not* bit-exact and cannot be made so: torchstrap
ports ATen's CPU formulation on CPU and ATen's CUDA formulation on CUDA, and
upstream's two differ (double bias corrections and a conditional lerp on CPU,
`fma`-chained single precision on CUDA). Being equal to the platform you actually
run on is the guarantee a caller can use; being equal to the other platform is
not, since the gradients feeding the optimizer are far further apart across
devices than its arithmetic is. So this file is a tolerance test by design.
"""

import pytest
import torch

from torchstrap.optimizer.adam import adam_step_  # noqa: F401 - defines the op


def _check_opcheck(device):
    """torch.library.opcheck — the tutorial's recommended registration test.

    Validates schema/alias annotations, the fake kernel, and that the declared
    mutations match what the kernel actually writes.
    """
    R, T = 3, 8
    p = torch.randn(R, T, device=device)
    U = lambda x: torch.full((R,), x, device=device)
    args = (
        p, torch.randn(R, T, device=device),
        torch.zeros(R, T, device=device), torch.zeros(R, T, device=device),
        None, torch.zeros(R, device=device),
        U(1e-2), U(0.9), U(0.999), U(1e-8), U(1e-2),
        torch.tensor([True, False, True], device=device),
    )
    kwargs = dict(amsgrad=False, maximize=False, decoupled_weight_decay=True)
    torch.library.opcheck(torch.ops.torchstrap.adam_step_.default, args, kwargs)


def _consolidate(tensors, R):
    """Pack a list of per-replica ``(R, *shape)`` tensors into one ``(R, T)``."""
    return torch.cat([t.reshape(R, -1) for t in tensors], dim=1).contiguous()


def _make_state(R, shapes, device, dtype=torch.float32, seed=0, amsgrad=False):
    g = torch.Generator(device=device).manual_seed(seed)
    rnd = lambda s: torch.randn(R, *s, device=device, dtype=dtype, generator=g)
    params = _consolidate([rnd(s) for s in shapes], R)
    grads = _consolidate([rnd(s) for s in shapes], R)
    exp_avgs = _consolidate([rnd(s).abs() for s in shapes], R)
    exp_avg_sqs = _consolidate([rnd(s).abs() for s in shapes], R)
    max_es = (
        _consolidate([rnd(s).abs() for s in shapes], R) if amsgrad else None
    )
    state_steps = torch.full((R,), 1.0, device=device, dtype=dtype)
    return params, grads, exp_avgs, exp_avg_sqs, max_es, state_steps


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


def _hparams(R, distinct):
    if not distinct:
        return (
            torch.full((R,), 1e-2),
            torch.full((R,), 0.9),
            torch.full((R,), 0.999),
            torch.full((R,), 1e-8),
            torch.full((R,), 1e-2),
        )
    return (
        torch.linspace(1e-3, 5e-2, R),  # lr
        torch.linspace(0.80, 0.95, R),  # beta1
        torch.linspace(0.990, 0.9999, R),  # beta2
        torch.linspace(1e-9, 1e-7, R),  # eps
        torch.linspace(0.0, 1e-1, R),  # weight_decay
    )


def _case(
    R,
    shapes,
    amsgrad,
    maximize,
    decoupled_weight_decay,
    num_steps=3,
    distinct_hparams=False,
    atol=1e-5,
    rtol=1e-5,
):
    cpu_args = _make_state(R, shapes, torch.device("cpu"), amsgrad=amsgrad, seed=42)
    cuda_args = _clone_to(cpu_args, torch.device("cuda"))

    lr, b1, b2, eps, wd = _hparams(R, distinct_hparams)
    mask = torch.ones(R, dtype=torch.bool)
    cu = lambda t: t.cuda()
    lr_cu, b1_cu, b2_cu, eps_cu, wd_cu, mask_cu = (
        cu(lr), cu(b1), cu(b2), cu(eps), cu(wd), cu(mask),
    )

    flags = dict(
        amsgrad=amsgrad,
        maximize=maximize,
        decoupled_weight_decay=decoupled_weight_decay,
    )

    for step in range(num_steps):
        gen = torch.Generator(device="cpu").manual_seed(100 + step)
        cpu_args[1].normal_(generator=gen)
        cuda_args[1].copy_(cpu_args[1], non_blocking=False)

        adam_step_(*cpu_args[:6], lr, b1, b2, eps, wd, mask, **flags)
        adam_step_(
            *cuda_args[:6], lr_cu, b1_cu, b2_cu, eps_cu, wd_cu, mask_cu, **flags
        )

    _compare("params", cpu_args[0], cuda_args[0], atol, rtol)
    _compare("exp_avgs", cpu_args[2], cuda_args[2], atol, rtol)
    _compare("exp_avg_sqs", cpu_args[3], cuda_args[3], atol, rtol)
    if amsgrad:
        _compare("max_exp_avg_sqs", cpu_args[4], cuda_args[4], atol, rtol)
    _compare("state_steps", cpu_args[5], cuda_args[5], atol, rtol)


def test_opcheck(device):
    """torch.library.opcheck — schema/alias annotations, the fake kernel, and
    that the declared mutations match what the kernel actually writes."""
    _check_opcheck(torch.device(device))


# T = 45 exercises ATen's ragged path, T = 52 its kILP-wide vectorized one.
_SHAPE_SETS = {
    "T=45": [(5, 7), (7,), (3,)],
    "T=52": [(5, 8), (8,), (4,)],
}

# The distinct sweep deliberately stacks the extreme end of every hyperparameter
# range onto the last replica, and Adam's bias correction is badly conditioned
# there: with beta2 = 0.9999 and a small step, `1 - beta2**step` is ~2e-4, i.e.
# ~3.7 decimal digits of cancellation, riding the largest lr in the sweep. This is
# exactly where the two upstream formulations part ways on purpose — ATen's CPU
# kernel absorbs that cancellation in float64 while ATen's CUDA kernel does it in
# float32 — so the residual there is a real, intended difference between the two
# ports, not a kernel defect, and it gets a looser bound. It stays far sharper
# than the bug it exists to catch: a per-replica hyperparameter that collapsed to
# a broadcast scalar moves params by O(lr), ~1e-2.
_TOL = {False: 1e-5, True: 1e-4}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device available")
@pytest.mark.parametrize("tag", list(_SHAPE_SETS))
@pytest.mark.parametrize("distinct_hparams", [False, True])
@pytest.mark.parametrize("amsgrad", [False, True])
@pytest.mark.parametrize("maximize", [False, True])
@pytest.mark.parametrize("decoupled", [False, True])
def test_cpu_matches_cuda(tag, distinct_hparams, amsgrad, maximize, decoupled):
    tol = _TOL[distinct_hparams]
    _case(
        4, _SHAPE_SETS[tag], amsgrad, maximize, decoupled,
        distinct_hparams=distinct_hparams, atol=tol, rtol=tol,
    )
