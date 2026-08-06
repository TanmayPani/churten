"""Vmap composability of the consolidated ``adagrad_step_``, plus the strided path.

Same two-path structure as the Adam and SGD vmap tests:

  (a) an outer ``vmap`` over a synthetic batch dim B — each call steps an R==1
      single-replica consolidated ``(T,)`` state — must match
  (b) a direct call with the same states stacked into the kernel's R==B ``(B, T)``
      layout.

``test_strided_rt_view`` is the only coverage of either Adagrad kernel's strided
path — ``in_dims=0`` makes ``movedim`` a no-op, so the case above never reaches it.
"""

import torch
from torch.func import vmap

from torchstrap.optimizer.adagrad import adagrad_step_


def _case(device, B=3, T=52):
    dev = torch.device(device)
    g = torch.Generator(device=dev).manual_seed(11)

    params_dir = torch.randn(B, T, device=dev, generator=g)
    grads_dir = torch.randn(B, T, device=dev, generator=g)
    sums_dir = torch.randn(B, T, device=dev, generator=g).abs()
    steps_dir = torch.zeros(B, device=dev)

    # Vary lr and lr_decay across the batch to surface broadcast bugs.
    lr_dir = torch.linspace(5e-3, 5e-2, B, device=dev)
    lrd_dir = torch.linspace(0.0, 5e-2, B, device=dev)
    wd_dir = torch.full((B,), 1e-2, device=dev)
    eps_dir = torch.full((B,), 1e-10, device=dev)
    mask_dir = torch.ones(B, dtype=torch.bool, device=dev)

    # ---- Path (a): outer vmap over the B-dim, calling adagrad_step_ on R==1 views.
    p_v, g_v, s_v, st_v = (
        t.clone() for t in (params_dir, grads_dir, sums_dir, steps_dir)
    )

    def step_one(p, gr, s, st, lr, lrd, wd, eps, mask):
        return adagrad_step_(
            p.unsqueeze(0), gr.unsqueeze(0), s.unsqueeze(0), st.view(1),
            lr.view(1), lrd.view(1), wd.view(1), eps.view(1),
            mask.view(1).to(torch.bool),
            maximize=False,
        )

    vmap(step_one, in_dims=(0,) * 9)(
        p_v, g_v, s_v, st_v, lr_dir, lrd_dir, wd_dir, eps_dir, mask_dir,
    )

    # ---- Path (b): direct call with the same (B, T) stacked layout.
    p_d, g_d, s_d, st_d = (
        t.clone() for t in (params_dir, grads_dir, sums_dir, steps_dir)
    )
    adagrad_step_(
        p_d, g_d, s_d, st_d, lr_dir, lrd_dir, wd_dir, eps_dir, mask_dir,
        maximize=False,
    )

    def cmp(name, a, b):
        if not torch.allclose(a, b, atol=1e-6, rtol=1e-5):
            diff = (a - b).abs().max().item()
            raise AssertionError(f"{name} vmap vs direct diverged: max diff {diff:.2e}")

    cmp("params", p_v, p_d)
    cmp("state_sums", s_v, s_d)
    cmp("state_steps", st_v, st_d)


def _case_strided(device, R=4, T=52):
    """A non-contiguous ``(R, T)`` buffer must give the same result as a contiguous one.

    This is the layout ``_adagrad_step_vmap``'s ``movedim(d, 0)`` produces for an
    outer vmap with ``bdim != 0``.

    On **CUDA** the comparison is ``torch.equal``: both paths gather into ``r_args``
    and run the same ``at::native::adagrad_math``, so only the addressing differs.

    On **CPU** it is a tolerance, because the two paths are genuinely different
    formulations by design — contiguous goes through ``adagrad_span_vec``, strided
    through ``adagrad_span_scalar``. That is ATen's own vec-vs-tail inconsistency,
    reproduced on purpose so each half stays bit-exact against upstream.
    """
    dev = torch.device(device)
    g = torch.Generator(device=dev).manual_seed(7)

    ref = [
        torch.randn(R, T, device=dev, generator=g),
        torch.randn(R, T, device=dev, generator=g),
        torch.randn(R, T, device=dev, generator=g).abs(),
    ]
    steps_ref = torch.zeros(R, device=dev)

    strided = []
    for t in ref:
        backing = torch.empty(T, R, device=dev).t()  # (R, T) view, inner stride R
        backing.copy_(t)
        assert not backing.is_contiguous()
        strided.append(backing)
    steps_str = steps_ref.clone()

    lr = torch.linspace(5e-3, 5e-2, R, device=dev)
    lrd = torch.full((R,), 1e-2, device=dev)
    wd = torch.full((R,), 1e-2, device=dev)
    eps = torch.full((R,), 1e-10, device=dev)
    mask = torch.ones(R, dtype=torch.bool, device=dev)
    mask[1] = False  # frozen rows must stay untouched on both layouts

    for buffers, steps in ((ref, steps_ref), (strided, steps_str)):
        for _ in range(3):
            adagrad_step_(
                buffers[0], buffers[1], buffers[2], steps,
                lr, lrd, wd, eps, mask, maximize=False,
            )

    exact = dev.type == "cuda"
    for i, name in enumerate(("params", "grads", "state_sums")):
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


def test_composes_under_vmap(device):
    _case(device)


def test_strided_rt_view(device):
    _case_strided(device)
