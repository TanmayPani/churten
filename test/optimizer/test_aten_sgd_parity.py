"""Bit-exact parity between torchstrap's SGD kernels and ATen's own fused SGD.

The SGD counterpart of `test_aten_fused_parity.py`, and the authoritative numerics
gate for `csrc/cpu/sgd.cpp` and `csrc/cuda/sgd.cu`. Both are **ports** — unlike
Adam's CUDA kernel, neither can `#include` ATen's math, because `sgd_math` lives in
an anonymous namespace inside `FusedSgdKernel.cu` / `FusedSGDKernel.cpp` and is not
installed in the wheel. So for the case ATen can also express — a single replica,
uniform hyperparameters, nothing frozen — each must agree with `torch._fused_sgd_`
to the **bit**, not merely to a tolerance. A failure means a port drifted: a
reassociated expression, a lost fma, a `std::fma` someone "tidied" in.

ATen's two formulations are not the same as each other (its CPU vectorized body and
its scalar tail are not even the same as each other — see the table in
`csrc/cpu/sgd.cpp`), so torchstrap's CPU and CUDA results are not bit-equal to each
other either. What each *is* bit-equal to is `torch.optim.SGD(fused=True)` on its
own device.

`momentum=0.0` is in the sweep on purpose: it is the empirical check that a replica
whose momentum is exactly zero, inside a call that *does* carry momentum buffers,
reduces to plain SGD — the per-replica `momentum != 0` test that ATen cannot
express, since its momentum is a single scalar.
"""

import pytest
import torch

from torchstrap.optimizer.sgd import sgd_step_

from test_aten_fused_parity import f32  # same directory; see rootdir conftest


def _case(
    device, T, momentum, dampening, weight_decay, nesterov, maximize,
    num_steps=3, seed=0,
):
    dev = torch.device(device)
    gen = torch.Generator(device=dev).manual_seed(seed)

    def rnd():
        return torch.randn(T, device=dev, generator=gen)

    p_ref = rnd()
    p_ts = p_ref.clone()

    # ATen decides "does a momentum buffer exist" from the caller's scalar momentum;
    # torchstrap decides it once per call (SGD.init's `has_momentum`) and then tests
    # each replica's momentum at runtime. Allocate on the same condition ATen uses so
    # the momentum=0 row really does compare the depth-3-with-zero-momentum path
    # against ATen's depth-2 path.
    has_buffer = momentum != 0.0
    mb_ref = rnd() if has_buffer else None
    mb_ts = mb_ref.clone() if has_buffer else None

    lr = f32(1e-2)
    momentum = f32(momentum)
    dampening = f32(dampening)
    weight_decay = f32(weight_decay)

    # torchstrap bumps state_steps inside the op and derives is_first_step from it;
    # ATen takes is_first_step as a host bool. Start at 0 so step 1 is the first.
    s_ts = torch.zeros(1, device=dev)

    def vec(x):
        return torch.full((1,), x, device=dev)

    mask = torch.ones(1, dtype=torch.bool, device=dev)

    for i in range(num_steps):
        g = torch.randn(T, device=dev, generator=gen)
        is_first_step = i == 0

        torch._fused_sgd_(
            [p_ref],
            [g],
            [mb_ref] if has_buffer else [],
            weight_decay=weight_decay,
            momentum=momentum,
            lr=lr,
            dampening=dampening,
            nesterov=nesterov,
            maximize=maximize,
            is_first_step=is_first_step,
            grad_scale=None,
            found_inf=None,
        )

        sgd_step_(
            p_ts.view(1, T),
            g.view(1, T),
            mb_ts.view(1, T) if has_buffer else None,
            s_ts,
            vec(lr),
            vec(momentum),
            vec(dampening),
            vec(weight_decay),
            mask,
            nesterov=nesterov,
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
    if has_buffer:
        same("momentum_buffers", mb_ref, mb_ts)
    assert s_ts.item() == num_steps


# Same reasoning as test_aten_fused_parity: `Vectorized<float>::size()` is 8 on AVX2
# / 16 on AVX512, so T=64 has no tail either way while T=45 and T=1000 do, and the
# tail is where ATen's own vectorized and scalar formulations disagree. On CUDA,
# T=64 exercises the vectorized kILP path and T=45 the ragged one.
_SHAPES = {"cpu": (64, 45, 1000), "cuda": (64, 45)}


@pytest.mark.parametrize("T", sorted(set(_SHAPES["cpu"]) | set(_SHAPES["cuda"])))
@pytest.mark.parametrize("momentum", [0.0, 0.9])
@pytest.mark.parametrize("weight_decay", [0.0, 1e-2])
@pytest.mark.parametrize("maximize", [False, True])
def test_bit_identical_to_aten(device, T, momentum, weight_decay, maximize):
    if T not in _SHAPES[device]:
        pytest.skip(f"T={T} is not part of the {device} sweep")
    _case(device, T, momentum, 0.0, weight_decay, nesterov=False, maximize=maximize)


@pytest.mark.parametrize("T", [64, 45])
@pytest.mark.parametrize("dampening", [0.0, 0.3])
@pytest.mark.parametrize("weight_decay", [0.0, 1e-2])
def test_dampening_bit_identical_to_aten(device, T, dampening, weight_decay):
    """Dampening only reaches the arithmetic through the momentum branch."""
    _case(device, T, 0.9, dampening, weight_decay, nesterov=False, maximize=False)


@pytest.mark.parametrize("T", [64, 45])
@pytest.mark.parametrize("weight_decay", [0.0, 1e-2])
@pytest.mark.parametrize("maximize", [False, True])
def test_nesterov_bit_identical_to_aten(device, T, weight_decay, maximize):
    """Nesterov requires momentum > 0 and dampening == 0, as torch.optim.SGD does."""
    _case(device, T, 0.9, 0.0, weight_decay, nesterov=True, maximize=maximize)
