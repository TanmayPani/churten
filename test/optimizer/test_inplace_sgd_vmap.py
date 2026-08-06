"""Vmap composability of the consolidated ``sgd_step_``:

  (a) an outer ``vmap`` over a synthetic batch dim B — each call steps an R==1
      single-replica consolidated ``(T,)`` state — must match
  (b) a direct call with the same states stacked into the kernel's R==B ``(B, T)``
      layout.

This exercises the op's ``register_vmap`` rule (``_sgd_step_vmap``), which treats the
outer vmap dim as the replica dim.

``test_strided_rt_view`` is the only coverage of either SGD kernel's strided path —
``in_dims=0`` makes ``movedim`` a no-op, so the case above never reaches it.
"""

import pytest
import torch
from torch.func import vmap

from torchstrap.optimizer.sgd import sgd_step_


def _case(device, has_momentum, B=3, T=52):
    dev = torch.device(device)
    g = torch.Generator(device=dev).manual_seed(11)

    params_dir = torch.randn(B, T, device=dev, generator=g)
    grads_dir = torch.randn(B, T, device=dev, generator=g)
    mb_dir = torch.randn(B, T, device=dev, generator=g)
    steps_dir = torch.zeros(B, device=dev)

    # Vary lr and momentum across the batch to surface broadcast bugs.
    lr_dir = torch.linspace(5e-3, 5e-2, B, device=dev)
    mom_dir = (
        torch.linspace(0.5, 0.95, B, device=dev)
        if has_momentum
        else torch.zeros(B, device=dev)
    )
    damp_dir = torch.zeros(B, device=dev)
    wd_dir = torch.full((B,), 1e-2, device=dev)
    mask_dir = torch.ones(B, dtype=torch.bool, device=dev)

    # ---- Path (a): outer vmap over the B-dim, calling sgd_step_ on R==1 views.
    p_v, g_v, mb_v, s_v = (
        t.clone() for t in (params_dir, grads_dir, mb_dir, steps_dir)
    )

    def step_one(p, gr, mb, s, lr, mom, damp, wd, mask):
        return sgd_step_(
            p.unsqueeze(0), gr.unsqueeze(0),
            mb.unsqueeze(0) if has_momentum else None,
            s.view(1),
            lr.view(1), mom.view(1), damp.view(1), wd.view(1),
            mask.view(1).to(torch.bool),
            nesterov=False, maximize=False,
        )

    vmap(step_one, in_dims=(0,) * 9)(
        p_v, g_v, mb_v, s_v, lr_dir, mom_dir, damp_dir, wd_dir, mask_dir,
    )

    # ---- Path (b): direct call with the same (B, T) stacked layout.
    p_d, g_d, mb_d, s_d = (
        t.clone() for t in (params_dir, grads_dir, mb_dir, steps_dir)
    )
    sgd_step_(
        p_d, g_d, mb_d if has_momentum else None, s_d,
        lr_dir, mom_dir, damp_dir, wd_dir, mask_dir,
        nesterov=False, maximize=False,
    )

    def cmp(name, a, b):
        if not torch.allclose(a, b, atol=1e-6, rtol=1e-5):
            diff = (a - b).abs().max().item()
            raise AssertionError(f"{name} vmap vs direct diverged: max diff {diff:.2e}")

    cmp("params", p_v, p_d)
    if has_momentum:
        cmp("momentum_buffers", mb_v, mb_d)
    cmp("state_steps", s_v, s_d)


def _case_strided(device, has_momentum, R=4, T=52):
    """A non-contiguous ``(R, T)`` buffer must give the same result as a contiguous one.

    This is the layout ``_sgd_step_vmap``'s ``movedim(d, 0)`` produces for an outer
    vmap with ``bdim != 0``. Both hand-written kernels have a dedicated strided path
    for it, and there is no Python wrapper left that could bail out to a reference
    implementation.

    On **CUDA** the comparison is ``torch.equal``: both paths gather into ``r_args``
    and run the same per-element ``sgd_math``, so only the addressing differs.

    On **CPU** it is a tolerance, because the two paths are genuinely different
    formulations by design — contiguous goes through ``sgd_span_vec``'s
    ``at::vec::fmadd``, strided through ``sgd_span_scalar``'s multiply-then-add. That
    is ATen's own vec-vs-tail inconsistency, reproduced on purpose so each half stays
    bit-exact against upstream. Worth ~1 ulp.
    """
    dev = torch.device(device)
    g = torch.Generator(device=dev).manual_seed(7)
    rand = lambda: torch.randn(R, T, device=dev, generator=g)

    ref = [rand(), rand(), rand()]
    steps_ref = torch.zeros(R, device=dev)

    strided = []
    for t in ref:
        backing = torch.empty(T, R, device=dev).t()  # (R, T) view, inner stride R
        backing.copy_(t)
        assert not backing.is_contiguous()
        strided.append(backing)
    steps_str = steps_ref.clone()

    lr = torch.linspace(5e-3, 5e-2, R, device=dev)
    mom = torch.full((R,), 0.9 if has_momentum else 0.0, device=dev)
    damp = torch.zeros(R, device=dev)
    wd = torch.full((R,), 1e-2, device=dev)
    mask = torch.ones(R, dtype=torch.bool, device=dev)
    mask[1] = False  # frozen rows must stay untouched on both layouts

    for buffers, steps in ((ref, steps_ref), (strided, steps_str)):
        for _ in range(3):
            sgd_step_(
                buffers[0], buffers[1], buffers[2] if has_momentum else None, steps,
                lr, mom, damp, wd, mask, nesterov=False, maximize=False,
            )

    exact = dev.type == "cuda"
    names = ("params", "grads", "momentum_buffers")
    for i, name in enumerate(names):
        if name == "momentum_buffers" and not has_momentum:
            continue
        a, b = ref[i], strided[i]
        ok = torch.equal(a, b) if exact else torch.allclose(a, b, atol=1e-6, rtol=1e-6)
        if not ok:
            diff = (a - b).abs().max().item()
            raise AssertionError(
                f"{name} differs between contiguous and strided (R, T) on "
                f"{device}: max abs diff {diff:.3e} "
                f"({'expected bit-identical' if exact else 'beyond 1 ulp'})"
            )
    if not torch.equal(steps_ref, steps_str):
        raise AssertionError("state_steps differ between layouts")
    if not torch.equal(ref[0][1], strided[0][1]):
        raise AssertionError("frozen row diverged between layouts")


@pytest.mark.parametrize("has_momentum", [False, True])
def test_composes_under_vmap(device, has_momentum):
    _case(device, has_momentum)


@pytest.mark.parametrize("has_momentum", [False, True])
def test_strided_rt_view(device, has_momentum):
    _case_strided(device, has_momentum)
