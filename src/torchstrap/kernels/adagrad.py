"""The `torchstrap::adagrad_step_` operator.

Defined and dispatched exactly the way ATen defines `_fused_adagrad_`, one level out
of tree:

| ATen | torchstrap |
|---|---|
| the `_fused_adagrad_` entry in `native_functions.yaml` | `TORCH_LIBRARY(torchstrap, m)` in `csrc/stubs.cpp` |
| `dispatch: CPU: _fused_adagrad_kernel_cpu_` | `TORCH_LIBRARY_IMPL(torchstrap, CPU, m)` in `csrc/cpu/adagrad.cpp` |
| `dispatch: CUDA: _fused_adagrad_kernel_cuda_` | `TORCH_LIBRARY_IMPL(torchstrap, CUDA, m)` in `csrc/cuda/adagrad.cu` |
| `torch._fused_adagrad_(...)` | `torch.ops.torchstrap.adagrad_step_(...)` |

Adagrad completes ATen's *entire* fused family (Adam/AdamW/SGD/Adagrad) — RMSprop,
Adadelta, NAdam, RAdam, Adamax, Rprop and ASGD have no fused reference on either
device, so nothing after this can be held to the bit-exactness bar.

The only things this module adds are the two rules the dispatcher cannot infer from
C++: `register_fake` (meta / `torch.compile`) and `register_vmap`.
"""

import torch

# Runs TORCH_LIBRARY / TORCH_LIBRARY_IMPL: defines torchstrap::adagrad_step_ and
# claims the CPU and (if built) CUDA keys.
import torchstrap.kernels._C  # noqa: F401
from torchstrap.kernels._consolidated import hyp, hyp_mask, lead_squeeze

adagrad_step_ = torch.ops.torchstrap.adagrad_step_

__all__ = ["adagrad_step_"]


def _adagrad_step_fake(
    params,
    grads,
    state_sums,
    state_steps,
    lr,
    lr_decay,
    weight_decay,
    eps,
    active_mask,
    maximize,
):
    return torch.empty((), device=params[0].device, dtype=params[0].dtype)


def _adagrad_step_vmap(
    info,
    in_dims,
    params,
    grads,
    state_sums,
    state_steps,
    lr,
    lr_decay,
    weight_decay,
    eps,
    active_mask,
    maximize,
):
    # See `kernels/_consolidated.py` for what `lead_squeeze` / `hyp` do and why.
    params_b = lead_squeeze(params, in_dims[0])
    grads_b = lead_squeeze(grads, in_dims[1])
    state_sums_b = lead_squeeze(state_sums, in_dims[2])
    state_steps_b = lead_squeeze(state_steps, in_dims[3])

    B = params_b.shape[0]

    lr_b, lrd_b, wd_b, eps_b = (
        hyp(lr, in_dims[4], B),
        hyp(lr_decay, in_dims[5], B),
        hyp(weight_decay, in_dims[6], B),
        hyp(eps, in_dims[7], B),
    )
    mask_b = hyp_mask(active_mask, in_dims[8], B)

    out = adagrad_step_(
        params_b,
        grads_b,
        state_sums_b,
        state_steps_b,
        lr_b,
        lrd_b,
        wd_b,
        eps_b,
        mask_b,
        maximize=maximize,
    )
    return out, None


torch.library.register_fake("torchstrap::adagrad_step_", _adagrad_step_fake)
torch.library.register_vmap("torchstrap::adagrad_step_", _adagrad_step_vmap)
