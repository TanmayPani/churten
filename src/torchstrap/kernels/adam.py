"""The `torchstrap::adam_step_` operator.

Defined and dispatched exactly the way ATen defines `_fused_adam_`, one level
out of tree:

| ATen | torchstrap |
|---|---|
| the `_fused_adam_` entry in `native_functions.yaml` | `TORCH_LIBRARY(torchstrap, m)` in `csrc/stubs.cpp` |
| `dispatch: CPU: _fused_adam_kernel_cpu_` | `TORCH_LIBRARY_IMPL(torchstrap, CPU, m)` in `csrc/cpu/adam.cpp` |
| `dispatch: CUDA: _fused_adam_kernel_cuda_` | `TORCH_LIBRARY_IMPL(torchstrap, CUDA, m)` in `csrc/cuda/adam.cu` |
| `torch._fused_adam_(...)` | `torch.ops.torchstrap.adam_step_(...)` |

Both kernels are compiled ahead of time by `setup.py` into one extension,
`torchstrap.kernels._C`; importing it runs the static initializers above, and from then
on the dispatcher routes by device with no Python in the path. A device whose
kernel was not built raises `NotImplementedError` from the dispatcher, exactly as
`torch._fused_adam_` does on a device ATen did not implement.

The only things this module adds are the two rules the dispatcher cannot infer
from C++: `register_fake` (meta/`torch.compile`) and `register_vmap` (so the op
composes inside an outer `torch.func.vmap`, which is the point of torchstrap).

The Helion kernel at the bottom is **dead code**, kept for reference. Nothing
imports or registers it.
"""

import functools

from typing import TYPE_CHECKING

import torch

# Runs TORCH_LIBRARY / TORCH_LIBRARY_IMPL: defines torchstrap::adam_step_ and
# claims the CPU and (if built) CUDA keys. Nothing else in the package touches
# the dispatcher.
import torchstrap.kernels._C  # noqa: F401
from torchstrap.kernels._consolidated import hyp, hyp_mask, lead_squeeze

if TYPE_CHECKING:  # the real module, so `hl.constexpr` annotations resolve
    import helion.language as hl

adam_step_ = torch.ops.torchstrap.adam_step_

__all__ = ["adam_step_"]


# ===========================================================================
# Fake and vmap rules
# ===========================================================================


def _adam_step_fake(
    params,
    grads,
    exp_avgs,
    exp_avg_sqs,
    max_exp_avg_sqs,
    state_steps,
    lr,
    beta1,
    beta2,
    eps,
    weight_decay,
    active_mask,
    amsgrad,
    maximize,
    decoupled_weight_decay,
):
    return torch.empty((), device=params[0].device, dtype=params[0].dtype)


def _adam_step_vmap(
    info,
    in_dims,
    params,
    grads,
    exp_avgs,
    exp_avg_sqs,
    max_exp_avg_sqs,
    state_steps,
    lr,
    beta1,
    beta2,
    eps,
    weight_decay,
    active_mask,
    amsgrad,
    maximize,
    decoupled_weight_decay,
):
    # See `kernels/_consolidated.py` for what `lead_squeeze` / `hyp` do and why.
    params_b = lead_squeeze(params, in_dims[0])
    grads_b = lead_squeeze(grads, in_dims[1])
    exp_avgs_b = lead_squeeze(exp_avgs, in_dims[2])
    exp_avg_sqs_b = lead_squeeze(exp_avg_sqs, in_dims[3])
    max_exp_avg_sqs_b = (
        lead_squeeze(max_exp_avg_sqs, in_dims[4])
        if max_exp_avg_sqs is not None
        else None
    )
    state_steps_b = lead_squeeze(state_steps, in_dims[5])

    B = params_b.shape[0]

    lr_b, b1_b, b2_b, eps_b, wd_b = (
        hyp(lr, in_dims[6], B),
        hyp(beta1, in_dims[7], B),
        hyp(beta2, in_dims[8], B),
        hyp(eps, in_dims[9], B),
        hyp(weight_decay, in_dims[10], B),
    )
    mask_b = hyp_mask(active_mask, in_dims[11], B)

    out = adam_step_(
        params_b,
        grads_b,
        exp_avgs_b,
        exp_avg_sqs_b,
        max_exp_avg_sqs_b,
        state_steps_b,
        lr_b,
        b1_b,
        b2_b,
        eps_b,
        wd_b,
        mask_b,
        amsgrad=amsgrad,
        maximize=maximize,
        decoupled_weight_decay=decoupled_weight_decay,
    )
    return out, None

torch.library.register_fake("torchstrap::adam_step_", _adam_step_fake)
torch.library.register_vmap("torchstrap::adam_step_", _adam_step_vmap)


# ===========================================================================
# Helion — DEAD CODE
# ===========================================================================
#
# Kept for reference only: nothing registers or calls any of the below. It was
# the backend for accelerators nvcc cannot target (ROCm, XPU); wiring it back in
# means adding a `register_kernel` for that device and nothing else.


# `hl` (helion.language) is a module global bound by `_helion()` on first use, via
# `globals()` rather than an assignment statement. Two constraints force that:
#
#   * helion rejects a kernel that closes over anything ("ClosuresNotSupported"),
#     so `hl` inside the kernel body must resolve as a *global*, not a freevar --
#     which rules out making it a local of `_helion`.
#   * assigning `hl = None` here as a placeholder would declare it a second time,
#     and a type checker then narrows it to a variable, which makes the
#     `amsgrad: hl.constexpr` annotations invalid type expressions. Writing through
#     `globals()` leaves the TYPE_CHECKING import above as the only declaration.


@functools.lru_cache(maxsize=1)
def _helion():
    """The compiled Helion kernel, built on first call.

    Helion is imported here rather than at module scope so a missing or broken
    install degrades to the PyTorch path at *call* time instead of breaking
    `import torchstrap` (helion pulls in triton, which is not cheap). The kernel
    has to be defined in here too, since `@helion.kernel` needs helion at
    definition time and reads the `hl.constexpr` annotations for real.
    """
    import helion
    import helion.language

    globals()["hl"] = helion.language

    # Config pinned by hand, not autotuned: the autotuner converges on a
    # pathological `block_sizes=[1, 128]` ~5-6x off the memory roofline, because
    # the bandwidth-saturating candidates hit Helion's 60 s compile timeout and get
    # eliminated. That trap is why the hand-written kernels exist.
    @helion.kernel(
        config=helion.Config(
            block_sizes=[1, 2048],
            num_warps=4,
            num_stages=1,
            pid_type="flat",
        ),
        static_shapes=False,
    )
    def helion_adam_kernel(
        p: torch.Tensor,  # (R, T)
        g: torch.Tensor,
        m: torch.Tensor,
        v: torch.Tensor,
        mx: torch.Tensor,
        lr_r: torch.Tensor,  # (R,)
        b1_r: torch.Tensor,
        b2_r: torch.Tensor,
        eps_r: torch.Tensor,
        wd_r: torch.Tensor,
        mask_r: torch.Tensor,
        bc1_r: torch.Tensor,
        bc2_r: torch.Tensor,
        amsgrad: hl.constexpr,
        maximize: hl.constexpr,
        decoupled_wd: hl.constexpr,
    ) -> None:
        R, n = p.shape
        for tile_r, tile_n in hl.tile([R, n]):
            # Per-replica scalars load once per replica-tile as (block_r,) and
            # reshape to (block_r, 1) so they broadcast over the T axis.
            lr_e = lr_r[tile_r][:, None]
            b1_e = b1_r[tile_r][:, None]
            b2_e = b2_r[tile_r][:, None]
            eps_e = eps_r[tile_r][:, None]
            wd_e = wd_r[tile_r][:, None]
            bc1_e = bc1_r[tile_r][:, None]
            bc2_e = bc2_r[tile_r][:, None]
            # Row-only active mask (block_r, 1): frozen replicas keep their old
            # values via the branchless `torch.where` below, so we can store
            # UNCONDITIONALLY (vectorized `block_ptr` stores) instead of predicated
            # pointer stores — the n-tail is still covered by Helion's boundary
            # check. Note this is where Helion loses to the C++ kernels on a
            # partly frozen ensemble: it still moves every frozen row's bytes.
            active = mask_r[tile_r][:, None] != 0

            p_e = p[tile_r, tile_n]
            g_e = g[tile_r, tile_n]
            m_e = m[tile_r, tile_n]
            v_e = v[tile_r, tile_n]

            if maximize:
                g_eff = -g_e
            else:
                g_eff = g_e
            if decoupled_wd:
                # `p_dec` feeds the active update; the original `p_e` is what
                # frozen rows store back, so decoupled decay never touches frozen
                # weights.
                p_dec = p_e * (1.0 - lr_e * wd_e)
            else:
                g_eff = g_eff + p_e * wd_e
                p_dec = p_e

            m_new = m_e * b1_e + g_eff * (1.0 - b1_e)
            v_new = v_e * b2_e + g_eff * g_eff * (1.0 - b2_e)

            if amsgrad:
                mx_e = mx[tile_r, tile_n]
                mx_max = torch.maximum(mx_e, v_new)
                denom = torch.sqrt(mx_max) / torch.sqrt(bc2_e) + eps_e
                hl.store(mx, [tile_r, tile_n], torch.where(active, mx_max, mx_e))
            else:
                denom = torch.sqrt(v_new) / torch.sqrt(bc2_e) + eps_e

            p_upd = p_dec - (lr_e / bc1_e) * m_new / denom

            # Branchless gated stores: active rows get the update, frozen rows get
            # their unchanged old value (an exact no-op).
            hl.store(p, [tile_r, tile_n], torch.where(active, p_upd, p_e))
            hl.store(m, [tile_r, tile_n], torch.where(active, m_new, m_e))
            hl.store(v, [tile_r, tile_n], torch.where(active, v_new, v_e))

    return helion_adam_kernel


@functools.lru_cache(maxsize=1)
def has_helion() -> bool:
    """Whether the Helion backend can actually be loaded."""
    try:
        _helion()
    except ImportError:
        return False
    return True


def helion_adam_kernel():
    """The compiled Helion kernel object (exposed for the config sweep bench)."""
    return _helion()


def helion_adam(
    params: torch.Tensor,
    grads: torch.Tensor,
    exp_avgs: torch.Tensor,
    exp_avg_sqs: torch.Tensor,
    max_exp_avg_sqs: torch.Tensor | None,
    state_steps: torch.Tensor,
    lr: torch.Tensor,
    beta1: torch.Tensor,
    beta2: torch.Tensor,
    eps: torch.Tensor,
    weight_decay: torch.Tensor,
    active_mask: torch.Tensor,
    amsgrad: bool,
    maximize: bool,
    decoupled_weight_decay: bool,
) -> torch.Tensor:
    kernel = _helion()

    # Bump state-steps once for all params (active replicas only).
    state_steps.add_(active_mask.to(state_steps.dtype))
    # Helion's `tensor[index]` gather wants a numeric dtype, not bool.
    mask_int = active_mask.to(torch.uint8)

    # The kernel consumes precomputed bias corrections, not `state_steps`.
    bc1 = 1.0 - beta1.pow(state_steps)
    bc2 = 1.0 - beta2.pow(state_steps)

    # `mx` must be a real tensor for Helion's signature; when not amsgrad, alias
    # to `p` and the constexpr branch never touches it.
    kernel(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs if amsgrad else params,
        lr,
        beta1,
        beta2,
        eps,
        weight_decay,
        mask_int,
        bc1,
        bc2,
        hl.constexpr(amsgrad),
        hl.constexpr(maximize),
        hl.constexpr(decoupled_weight_decay),
    )

    return torch.zeros((), device=params[0].device, dtype=params[0].dtype)
