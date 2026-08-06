"""Bit-exact parity between torchstrap's Adagrad kernels and ATen's own fused Adagrad.

The Adagrad member of the parity family (see `test_aten_fused_parity.py` and
`test_aten_sgd_parity.py`). Adagrad completes ATen's *entire* fused set — after this
there is no upstream fused kernel left to be bit-exact against.

The two backends get here by different routes, which is worth knowing when this test
fails:

  * ``csrc/cuda/adagrad.cu`` **includes** ``fused_adagrad_utils.cuh`` and calls
    ``at::native::adagrad_math`` verbatim (the header ships in the wheel, unlike
    SGD's). A failure there is addressing or launch geometry, not arithmetic.
  * ``csrc/cpu/adagrad.cpp`` is a **port** of ``FusedAdagradKernel.cpp``. A failure
    there is the usual suspect: a reassociated expression or a lost fma.

``lr_decay`` is swept because it is the only thing that makes ``corrected_lr`` depend
on the step count, and torchstrap derives that from its own ``(R,)`` counter rather
than ATen's per-tensor ``state_steps`` — so a step-indexing error is invisible at
``lr_decay = 0`` and shows up immediately here.
"""

import pytest
import torch

from torchstrap.optimizer.adagrad import adagrad_step_

from test_aten_fused_parity import f32  # same directory; see rootdir conftest


def _case(device, T, lr_decay, weight_decay, eps, maximize, num_steps=3, seed=0):
    dev = torch.device(device)
    gen = torch.Generator(device=dev).manual_seed(seed)

    def rnd():
        return torch.randn(T, device=dev, generator=gen)

    p_ref = rnd()
    s_ref = rnd().abs()  # state_sum is non-negative in practice
    p_ts, s_ts = p_ref.clone(), s_ref.clone()

    lr = f32(1e-2)
    lr_decay = f32(lr_decay)
    weight_decay = f32(weight_decay)
    eps = f32(eps)

    # ATen reads state_steps as-is (torch.optim bumps before calling); torchstrap
    # bumps inside the op. Start both at 0 so each iteration runs on the same step.
    step_ref = torch.zeros((), device=dev)
    step_ts = torch.zeros(1, device=dev)

    def vec(x):
        return torch.full((1,), x, device=dev)

    mask = torch.ones(1, dtype=torch.bool, device=dev)

    for _ in range(num_steps):
        g = torch.randn(T, device=dev, generator=gen)

        step_ref += 1
        torch._fused_adagrad_(
            [p_ref],
            [g],
            [s_ref],
            [step_ref],
            lr=lr,
            lr_decay=lr_decay,
            weight_decay=weight_decay,
            eps=eps,
            maximize=maximize,
            grad_scale=None,
            found_inf=None,
        )

        adagrad_step_(
            p_ts.view(1, T),
            g.view(1, T),
            s_ts.view(1, T),
            step_ts,
            vec(lr),
            vec(lr_decay),
            vec(weight_decay),
            vec(eps),
            mask,
            maximize=maximize,
        )

    if dev.type == "cuda":
        torch.cuda.synchronize()

    def same(name, a, b):
        if not torch.equal(a, b):
            diff = (a - b).abs().max().item()
            n = (a != b).sum().item()
            raise AssertionError(
                f"{name} is not bit-identical to ATen on {dev.type}: "
                f"{n}/{a.numel()} elements differ, max abs diff {diff:.3e}"
            )

    same("params", p_ref, p_ts)
    same("state_sums", s_ref, s_ts)
    if not torch.equal(step_ref.view(1), step_ts):
        raise AssertionError(f"state_steps {step_ref.item()} vs {step_ts.item()}")


# Same reasoning as the Adam and SGD parity files: `Vectorized<float>::size()` is 8 on
# AVX2 / 16 on AVX512, so T=64 has no tail either way while T=45 and T=1000 do. On
# CUDA, T=64 exercises the vectorized kILP path and T=45 the ragged one.
_SHAPES = {"cpu": (64, 45, 1000), "cuda": (64, 45)}


@pytest.mark.parametrize("T", sorted(set(_SHAPES["cpu"]) | set(_SHAPES["cuda"])))
@pytest.mark.parametrize("lr_decay", [0.0, 1e-2])
@pytest.mark.parametrize("weight_decay", [0.0, 1e-2])
@pytest.mark.parametrize("maximize", [False, True])
def test_bit_identical_to_aten(device, T, lr_decay, weight_decay, maximize):
    if T not in _SHAPES[device]:
        pytest.skip(f"T={T} is not part of the {device} sweep")
    _case(device, T, lr_decay, weight_decay, eps=1e-10, maximize=maximize)


@pytest.mark.parametrize("T", [64, 45])
@pytest.mark.parametrize("eps", [1e-10, 1e-6])
def test_eps_bit_identical_to_aten(device, T, eps):
    """`eps` is a `const double&` on CUDA and a narrowed `scalar_t` on CPU — the
    single largest source of the CPU/CUDA gap, so pin each side to its own upstream."""
    _case(device, T, lr_decay=1e-2, weight_decay=1e-2, eps=eps, maximize=False)
