"""CPU-vs-CUDA parity for the consolidated ``adagrad_step_``, plus ``opcheck``.

The two backends compute this update in different precisions, and it matters less than
it looks. ATen's CUDA ``adagrad_math`` takes its hyperparameters as ``const double&``,
so for float parameters both

    grad += param * weight_decay
    param = param - corrected_lr * grad / (std::sqrt(state_sum) + eps)

promote to **fp64 on the device**. ATen's CPU kernel does the opposite: it computes
``clr = lr / (1 + (step - 1) * lr_decay)`` on the host in double and then narrows it
(``Vec(scalar_t(clr))``, ``Vec(scalar_t(eps))``, ``Vec(scalar_t(weight_decay))``), so
its element loop is entirely float.

**Measured, that split costs about 1 ulp** — max abs 1.2e-7, max rel 7.1e-7 across the
whole sweep below, with ``state_sums`` usually bit-identical. So this file holds one
tight tolerance, and unlike ``test_cpu_cuda_parity.py`` the ``distinct_hparams`` sweep
needs no relaxation: Adagrad's update is short and well conditioned, with nothing like
Adam's ``1 - beta2**step`` cancellation for the two precisions to part ways over.

Do not "fix" the split by widening ``csrc/cpu/adagrad.cpp`` to double. That would break
the CPU bit-exactness gate — see ``test_aten_adagrad_parity.py``, which is the claim
that actually matters — in exchange for agreement with a platform the caller is not
running on.
"""

import pytest
import torch

from torchstrap.optimizer.adagrad import adagrad_step_  # noqa: F401 - defines the op


def _check_opcheck(device):
    """torch.library.opcheck — schema/alias annotations, the fake kernel, and that
    the declared mutations match what the kernel actually writes."""
    R, T = 3, 8
    U = lambda x: torch.full((R,), x, device=device)
    args = (
        torch.randn(R, T, device=device),
        torch.randn(R, T, device=device),
        torch.rand(R, T, device=device),
        torch.zeros(R, device=device),
        U(1e-2), U(1e-2), U(1e-2), U(1e-10),
        torch.tensor([True, False, True], device=device),
    )
    torch.library.opcheck(
        torch.ops.torchstrap.adagrad_step_.default, args, dict(maximize=False)
    )


def _consolidate(tensors, R):
    return torch.cat([t.reshape(R, -1) for t in tensors], dim=1).contiguous()


def _make_state(R, shapes, device, dtype=torch.float32, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    rnd = lambda s: torch.randn(R, *s, device=device, dtype=dtype, generator=g)
    params = _consolidate([rnd(s) for s in shapes], R)
    grads = _consolidate([rnd(s) for s in shapes], R)
    state_sums = _consolidate([rnd(s).abs() for s in shapes], R)
    # Start mid-training so `lr_decay` actually bites.
    state_steps = torch.full((R,), 3.0, device=device, dtype=dtype)
    return params, grads, state_sums, state_steps


def _clone_to(args, device):
    return tuple(t.detach().to(device).contiguous() for t in args)


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
            torch.full((R,), 1e-2),
            torch.full((R,), 1e-2),
            torch.full((R,), 1e-10),
        )
    return (
        torch.linspace(1e-3, 5e-2, R),   # lr
        torch.linspace(0.0, 5e-2, R),    # lr_decay
        torch.linspace(0.0, 1e-1, R),    # weight_decay
        torch.linspace(1e-10, 1e-6, R),  # eps
    )


def _case(R, shapes, maximize, num_steps=3, distinct_hparams=False,
          atol=1e-5, rtol=1e-5):
    cpu_args = _make_state(R, shapes, torch.device("cpu"), seed=42)
    cuda_args = _clone_to(cpu_args, torch.device("cuda"))

    lr, lrd, wd, eps = _hparams(R, distinct_hparams)
    mask = torch.ones(R, dtype=torch.bool)
    cu = lambda t: t.cuda()
    lr_cu, lrd_cu, wd_cu, eps_cu, mask_cu = (
        cu(lr), cu(lrd), cu(wd), cu(eps), cu(mask)
    )

    for step in range(num_steps):
        gen = torch.Generator(device="cpu").manual_seed(100 + step)
        cpu_args[1].normal_(generator=gen)
        cuda_args[1].copy_(cpu_args[1], non_blocking=False)

        adagrad_step_(*cpu_args, lr, lrd, wd, eps, mask, maximize=maximize)
        adagrad_step_(
            *cuda_args, lr_cu, lrd_cu, wd_cu, eps_cu, mask_cu, maximize=maximize
        )

    _compare("params", cpu_args[0], cuda_args[0], atol, rtol)
    _compare("state_sums", cpu_args[2], cuda_args[2], atol, rtol)
    _compare("state_steps", cpu_args[3], cuda_args[3], atol, rtol)


def test_opcheck(device):
    _check_opcheck(torch.device(device))


# T = 45 exercises ATen's ragged path, T = 52 its kILP-wide vectorized one.
_SHAPE_SETS = {
    "T=45": [(5, 7), (7,), (3,)],
    "T=52": [(5, 8), (8,), (4,)],
}

# Measured across every row below: params max abs 1.2e-7 / max rel 7.1e-7, state_sums
# almost always exactly 0. 1e-5 leaves ~2 decades of headroom over that and is still
# far sharper than the bug it exists to catch — a per-replica hyperparameter that
# collapsed to a broadcast scalar moves params by O(lr), ~1e-2.
_TOL = 1e-5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device available")
@pytest.mark.parametrize("tag", list(_SHAPE_SETS))
@pytest.mark.parametrize("distinct_hparams", [False, True])
@pytest.mark.parametrize("maximize", [False, True])
def test_cpu_matches_cuda(tag, distinct_hparams, maximize):
    _case(
        4, _SHAPE_SETS[tag], maximize,
        distinct_hparams=distinct_hparams, atol=_TOL, rtol=_TOL,
    )
