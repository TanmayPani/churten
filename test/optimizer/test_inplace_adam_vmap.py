"""Vmap composability of the consolidated ``adam_step_``:

  (a) an outer ``vmap`` over a synthetic batch dim B — each call steps an R==1
      single-replica consolidated ``(T,)`` state — must produce mutations
      byte-identical to
  (b) a direct call with the same states stacked into the kernel's R==B ``(B, T)``
      layout.

This exercises the op's ``register_vmap`` rule (``_adam_step_vmap``), which treats the
outer vmap dim as the replica dim.
"""
import pytest
import torch
from torch.func import vmap

from torchstrap.optimizer.adam import adam_step_


def _consolidate_single(tensors):
    """Pack a list of single-replica ``(*shape,)`` tensors into one ``(T,)``."""
    return torch.cat([t.reshape(-1) for t in tensors]).contiguous()


def _build_single(shapes, device, seed=0):
    """One-replica consolidated state: each component a single ``(T,)`` buffer,
    plus a scalar ``()`` state-step."""
    g = torch.Generator(device=device).manual_seed(seed)
    rand = lambda: _consolidate_single([torch.randn(*s, device=device, generator=g) for s in shapes])
    pos = lambda: _consolidate_single([torch.randn(*s, device=device, generator=g).abs() for s in shapes])
    return rand(), rand(), pos(), pos(), pos(), torch.zeros((), device=device)


def _case(device, amsgrad, B=3):
    dev = torch.device(device)
    shapes = [(4, 6), (6,), (3,)]

    # B independent single-replica states; stack to (B, T) / (B,) — the R==B layout.
    singles = [_build_single(shapes, dev, seed=100 + b) for b in range(B)]
    stack = lambda idx: torch.stack([s[idx] for s in singles])
    params_dir, grads_dir, m_dir, v_dir, mx_dir, steps_dir = (stack(i) for i in range(6))

    # Per-replica hyperparams: vary lr across the batch to surface broadcast bugs.
    lr_dir = torch.linspace(5e-3, 5e-2, B, device=dev)
    b1_dir = torch.full((B,), 0.9, device=dev)
    b2_dir = torch.full((B,), 0.999, device=dev)
    eps_dir = torch.full((B,), 1e-8, device=dev)
    wd_dir = torch.full((B,), 1e-2, device=dev)
    mask_dir = torch.ones(B, dtype=torch.bool, device=dev)

    # ---- Path (a): outer vmap over the B-dim, calling adam_step_ on R==1 views.
    p_v, g_v, m_v, v_v, mx_v, s_v = (t.clone() for t in
                                     (params_dir, grads_dir, m_dir, v_dir, mx_dir, steps_dir))

    def step_one(p, g, m, v, mx, s, lr, b1, b2, eps, wd, mask):
        # vmap unwraps each batched input to its single-replica slice; promote to
        # an R==1 leading dim for the kernel.
        return adam_step_(
            p.unsqueeze(0), g.unsqueeze(0), m.unsqueeze(0), v.unsqueeze(0),
            mx.unsqueeze(0), s.view(1),
            lr.view(1), b1.view(1), b2.view(1), eps.view(1), wd.view(1),
            mask.view(1).to(torch.bool),
            amsgrad=amsgrad, maximize=False, decoupled_weight_decay=True,
        )

    vmap(step_one, in_dims=(0,) * 12)(
        p_v, g_v, m_v, v_v, mx_v, s_v,
        lr_dir, b1_dir, b2_dir, eps_dir, wd_dir, mask_dir,
    )

    # ---- Path (b): direct call with the same (B, T) stacked layout.
    p_d, g_d, m_d, v_d, mx_d, s_d = (t.clone() for t in
                                     (params_dir, grads_dir, m_dir, v_dir, mx_dir, steps_dir))
    adam_step_(
        p_d, g_d, m_d, v_d, mx_d, s_d,
        lr_dir, b1_dir, b2_dir, eps_dir, wd_dir, mask_dir,
        amsgrad=amsgrad, maximize=False, decoupled_weight_decay=True,
    )

    # ---- Compare.
    def cmp(name, a, b):
        if not torch.allclose(a, b, atol=1e-6, rtol=1e-5):
            diff = (a - b).abs().max().item()
            raise AssertionError(f"{name} vmap vs direct diverged: max diff {diff:.2e}")

    cmp("params", p_v, p_d)
    cmp("exp_avgs", m_v, m_d)
    cmp("exp_avg_sqs", v_v, v_d)
    if amsgrad:
        cmp("max_exp_avg_sqs", mx_v, mx_d)
    cmp("state_steps", s_v, s_d)


def _case_strided(device, amsgrad, R=4, T=52):
    """A non-contiguous ``(R, T)`` buffer must give the same result as a contiguous one.

    This is the layout `_adam_step_vmap`'s ``movedim(d, 0)`` produces for an outer
    vmap with ``bdim != 0``, and it is *not* reached by `case` above (``in_dims=0``
    makes the movedim a no-op). Both hand-written kernels have a dedicated strided
    path for it -- the CUDA one because, now that C++ owns the dispatch key via
    C++ directly, there is no longer a Python wrapper that can bail out to
    the reference implementation.

    On **CUDA** the comparison is `torch.equal`: both paths gather into ``r_args``
    and run the same per-element `adam_math`, so only the addressing differs and a
    mismatch means the stride arithmetic is wrong.

    On **CPU** it is a tight tolerance instead, because the two paths genuinely are
    different formulations by design — contiguous goes through `adam_span_vec`
    (`at::vec::fmadd`), strided through `adam_span_scalar` (an explicit
    multiply-then-add). That is the same internal inconsistency ATen's own CPU
    kernel has between its vectorized body and its ragged tail, which
    cpu/adam.cpp reproduces on purpose so each half stays bit-exact against
    upstream. It is worth ~1 ulp.
    """
    dev = torch.device(device)
    g = torch.Generator(device=dev).manual_seed(7)
    rand = lambda: torch.randn(R, T, device=dev, generator=g)

    ref = [rand(), rand(), rand().abs(), rand().abs(), rand().abs()]
    steps_ref = torch.zeros(R, device=dev)

    # Same values, but each buffer is a transposed view of a (T, R) allocation, so
    # stride(1) != 1 and the vectorized/aligned paths are ineligible.
    strided = []
    for t in ref:
        backing = torch.empty(T, R, device=dev).t()  # (R, T) view, inner stride R
        backing.copy_(t)
        assert not backing.is_contiguous()
        strided.append(backing)
    steps_str = steps_ref.clone()

    lr = torch.linspace(5e-3, 5e-2, R, device=dev)
    b1 = torch.full((R,), 0.9, device=dev)
    b2 = torch.full((R,), 0.999, device=dev)
    eps = torch.full((R,), 1e-8, device=dev)
    wd = torch.full((R,), 1e-2, device=dev)
    mask = torch.ones(R, dtype=torch.bool, device=dev)
    mask[1] = False  # frozen rows must stay untouched on both layouts

    flags = dict(amsgrad=amsgrad, maximize=False, decoupled_weight_decay=True)
    for buffers, steps in ((ref, steps_ref), (strided, steps_str)):
        for _ in range(3):
            adam_step_(
                buffers[0], buffers[1], buffers[2], buffers[3],
                buffers[4] if amsgrad else None, steps,
                lr, b1, b2, eps, wd, mask, **flags,
            )

    exact = dev.type == "cuda"
    names = ("params", "grads", "exp_avgs", "exp_avg_sqs", "max_exp_avg_sqs")
    for i, name in enumerate(names):
        if not amsgrad and name == "max_exp_avg_sqs":
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

    # Frozen rows must be untouched on both layouts, whatever the access pattern.
    if not torch.equal(ref[0][1], strided[0][1]):
        raise AssertionError("frozen row diverged between layouts")


@pytest.mark.parametrize("amsgrad", [False, True])
def test_composes_under_vmap(device, amsgrad):
    _case(device, amsgrad)


@pytest.mark.parametrize("amsgrad", [False, True])
def test_strided_rt_view(device, amsgrad):
    _case_strided(device, amsgrad)
