"""The `torchstrap::sgd_step_` operator.

Defined and dispatched exactly the way ATen defines `_fused_sgd_`, one level out
of tree:

| ATen | torchstrap |
|---|---|
| the `_fused_sgd_` entry in `native_functions.yaml` | `TORCH_LIBRARY(torchstrap, m)` in `csrc/stubs.cpp` |
| `dispatch: CPU: _fused_sgd_kernel_cpu_` | `TORCH_LIBRARY_IMPL(torchstrap, CPU, m)` in `csrc/cpu/sgd.cpp` |
| `dispatch: CUDA: _fused_sgd_kernel_cuda_` | `TORCH_LIBRARY_IMPL(torchstrap, CUDA, m)` in `csrc/cuda/sgd.cu` |
| `torch._fused_sgd_(...)` | `torch.ops.torchstrap.sgd_step_(...)` |

Both kernels are compiled ahead of time by `setup.py` into the same extension as
Adam's, `torchstrap.kernels._C`; importing it runs the static initializers and from
then on the dispatcher routes by device with no Python in the path. A device whose
kernel was not built raises `NotImplementedError` from the dispatcher, exactly as
`torch._fused_sgd_` does on a device ATen did not implement.

The only things this module adds are the two rules the dispatcher cannot infer from
C++: `register_fake` (meta / `torch.compile`) and `register_vmap`.
"""

import torch

# Runs TORCH_LIBRARY / TORCH_LIBRARY_IMPL: defines torchstrap::sgd_step_ and claims
# the CPU and (if built) CUDA keys.
import torchstrap.kernels._C  # noqa: F401
from torchstrap.kernels._consolidated import hyp, hyp_mask, lead_squeeze

sgd_step_ = torch.ops.torchstrap.sgd_step_

__all__ = ["sgd_step_"]


def _sgd_step_fake(
    params,
    grads,
    momentum_buffers,
    state_steps,
    lr,
    momentum,
    dampening,
    weight_decay,
    active_mask,
    nesterov,
    maximize,
):
    return torch.empty((), device=params[0].device, dtype=params[0].dtype)


def _sgd_step_vmap(
    info,
    in_dims,
    params,
    grads,
    momentum_buffers,
    state_steps,
    lr,
    momentum,
    dampening,
    weight_decay,
    active_mask,
    nesterov,
    maximize,
):
    # See `kernels/_consolidated.py` for what `lead_squeeze` / `hyp` do and why.
    params_b = lead_squeeze(params, in_dims[0])
    grads_b = lead_squeeze(grads, in_dims[1])
    momentum_buffers_b = (
        lead_squeeze(momentum_buffers, in_dims[2])
        if momentum_buffers is not None
        else None
    )
    state_steps_b = lead_squeeze(state_steps, in_dims[3])

    B = params_b.shape[0]

    lr_b, mom_b, damp_b, wd_b = (
        hyp(lr, in_dims[4], B),
        hyp(momentum, in_dims[5], B),
        hyp(dampening, in_dims[6], B),
        hyp(weight_decay, in_dims[7], B),
    )
    mask_b = hyp_mask(active_mask, in_dims[8], B)

    out = sgd_step_(
        params_b,
        grads_b,
        momentum_buffers_b,
        state_steps_b,
        lr_b,
        mom_b,
        damp_b,
        wd_b,
        mask_b,
        nesterov=nesterov,
        maximize=maximize,
    )
    return out, None


torch.library.register_fake("torchstrap::sgd_step_", _sgd_step_fake)
torch.library.register_vmap("torchstrap::sgd_step_", _sgd_step_vmap)
