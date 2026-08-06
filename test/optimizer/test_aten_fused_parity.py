"""Bit-exact parity between torchstrap's kernels and ATen's own fused Adam.

Both hand-written kernels are ports of ATen's fused Adam **for the device they
run on**, so for the case ATen can also express -- a single replica, uniform
hyperparameters, nothing frozen -- each must agree with `torch._fused_adam_` /
`torch._fused_adamw_` to the **bit**, not merely to a tolerance:

  * ``csrc/cuda/adam.cu``      ports ``adam_math`` from ``fused_adam_utils.cuh``
  * ``csrc/cpu/adam.cpp`` ports ``adam_math`` from ``cpu/FusedAdamKernel.cpp``

Those two upstream formulations are *not* the same -- ATen deliberately gives
each backend its own (the CPU one computes bias corrections in double and uses a
conditional lerp for ``exp_avg``), so torchstrap's CPU and CUDA results are not
bit-equal to each other either. What each *is* bit-equal to is
``torch.optim.Adam(fused=True)`` on its own device, which is the guarantee a
caller can use. This test is the gate on that; a failure means a port drifted --
a reassociated expression, a lost fma, a changed bias correction -- even when the
1e-5 tests still pass.

Both the ``float4`` path (``T % 4 == 0``) and the scalar path are covered on
CUDA, as are the vectorized body and the ``T % Vec::size()`` scalar tail on CPU,
and ``weight_decay=0`` is included so ATen's ``if (weight_decay != 0)`` branch is
exercised both ways.
"""

from itertools import product

import pytest
import torch

from torchstrap.optimizer.adam import adam_step_


def f32(x: float) -> float:
    """Round a Python float through float32.

    torchstrap carries per-replica hyperparameters as float32 ``(R,)`` tensors,
    while ATen takes them as C++ ``double`` -- and its CPU kernel *keeps* them in
    double through the bias corrections. So the two only see the same numbers if
    the value handed to ATen is already exactly representable in float32.
    Without this, a mismatch would just mean "different inputs", not "different
    math". On CUDA it is a no-op (ATen narrows to float there anyway).
    """
    return torch.tensor(x, dtype=torch.float32).item()


def _case(device, T, amsgrad, maximize, decoupled, weight_decay, num_steps=3, seed=0):
    dev = torch.device(device)
    gen = torch.Generator(device=dev).manual_seed(seed)

    def rnd():
        return torch.randn(T, device=dev, generator=gen)

    p_ref, m_ref, v_ref = rnd(), rnd().abs(), rnd().abs()
    mx_ref = rnd().abs() if amsgrad else None

    p_ts, m_ts, v_ts = p_ref.clone(), m_ref.clone(), v_ref.clone()
    mx_ts = mx_ref.clone() if amsgrad else None

    lr, beta1, beta2, eps = f32(1e-2), f32(0.9), f32(0.999), f32(1e-8)
    weight_decay = f32(weight_decay)

    # ATen reads state_steps as-is (torch.optim bumps before calling); torchstrap
    # bumps inside the op. Start both at 0 so each iteration runs on the same step.
    s_ref = torch.zeros((), device=dev)
    s_ts = torch.zeros(1, device=dev)

    fused = torch._fused_adamw_ if decoupled else torch._fused_adam_

    def vec(x):
        return torch.full((1,), x, device=dev)

    mask = torch.ones(1, dtype=torch.bool, device=dev)

    for _ in range(num_steps):
        g = torch.randn(T, device=dev, generator=gen)

        s_ref += 1
        fused(
            [p_ref],
            [g],
            [m_ref],
            [v_ref],
            [mx_ref] if amsgrad else [],
            [s_ref],
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            weight_decay=weight_decay,
            eps=eps,
            amsgrad=amsgrad,
            maximize=maximize,
            grad_scale=None,
            found_inf=None,
        )

        adam_step_(
            p_ts.view(1, T),
            g.view(1, T),
            m_ts.view(1, T),
            v_ts.view(1, T),
            mx_ts.view(1, T) if amsgrad else None,
            s_ts,
            vec(lr),
            vec(beta1),
            vec(beta2),
            vec(eps),
            vec(weight_decay),
            mask,
            amsgrad=amsgrad,
            maximize=maximize,
            decoupled_weight_decay=decoupled,
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
    same("exp_avgs", m_ref, m_ts)
    same("exp_avg_sqs", v_ref, v_ts)
    if amsgrad:
        same("max_exp_avg_sqs", mx_ref, mx_ts)
    if not torch.equal(s_ref.view(1), s_ts):
        raise AssertionError(f"state_steps {s_ref.item()} vs {s_ts.item()}")


# `Vectorized<float>::size()` is 8 on AVX2 / 16 on AVX512: T=64 is a whole number
# of vectors either way (no tail), T=45 and T=1000 leave a ragged tail on both.
# The tail is the one place ATen's own vectorized and scalar formulations differ
# in fma contraction, so it needs its own coverage. T=1000 also spans more than
# one `at::parallel_for` task on a multi-core box.
#
# On CUDA, T=64 exercises the vectorized kILP path and T=45 the ragged one.
_SHAPES = {"cpu": (64, 45, 1000), "cuda": (64, 45)}


@pytest.mark.parametrize("T", sorted(set(_SHAPES["cpu"]) | set(_SHAPES["cuda"])))
@pytest.mark.parametrize("amsgrad", [False, True])
@pytest.mark.parametrize("maximize", [False, True])
@pytest.mark.parametrize("decoupled", [False, True])
@pytest.mark.parametrize("weight_decay", [0.0, 1e-2])
def test_bit_identical_to_aten(
    device, T, amsgrad, maximize, decoupled, weight_decay
):
    if T not in _SHAPES[device]:
        pytest.skip(f"T={T} is not part of the {device} sweep")
    _case(device, T, amsgrad, maximize, decoupled, weight_decay)
