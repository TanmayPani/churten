"""The `SGD` optimizer — a `GradientTransformation`, not an instance.

This module holds only the optimizer. The `torchstrap::sgd_step_` operator and its
two kernels (`csrc/cpu/sgd.cpp`, `csrc/cuda/sgd.cu`) live in
`torchstrap.kernels.sgd`; no dispatch key is claimed from the `optimizer` package.
"""

import torch

from torchstrap.state import State
from torchstrap.optimizer.grad_transform import GradientTransformation
from torchstrap.kernels.sgd import sgd_step_

__all__ = ["SGD", "sgd_step_"]


def _any_nonzero(value: float | torch.Tensor) -> bool:
    return bool(torch.as_tensor(value).ne(0).any())


def _all_zero(value: float | torch.Tensor) -> bool:
    return not _any_nonzero(value)


class SGD(metaclass=GradientTransformation):
    @classmethod
    def init(
        cls,
        state: State,
        *,
        lr: float | torch.Tensor = 1e-3,
        momentum: float | torch.Tensor = 0.0,
        dampening: float | torch.Tensor = 0.0,
        weight_decay: float | torch.Tensor = 0.0,
        nesterov: bool = False,
        maximize: bool = False,
    ) -> State:
        # `torch.optim.SGD`'s validation, applied across the whole ensemble.
        if nesterov and (_all_zero(momentum) or _any_nonzero(dampening)):
            raise ValueError(
                "Nesterov momentum requires a non-zero momentum and zero dampening"
            )

        state.add_optim_state("lr", lr)
        state.add_optim_state("momentum", momentum)
        state.add_optim_state("dampening", dampening)
        state.add_optim_state("weight_decay", weight_decay)
        state.add_optim_state("state_steps", 0)

        # Whether the momentum buffer EXISTS is one call-level decision — ATen's
        # depth 3 vs 2 — while its *value* stays per-replica. So a sweep that
        # includes momentum=0 for some replicas allocates the buffer for the whole
        # ensemble; both kernels then give those replicas plain SGD and leave their
        # buffer rows untouched, via a per-replica `momentum != 0` test that ATen
        # has no way to express (its momentum is a single scalar).
        has_momentum = _any_nonzero(momentum)
        if has_momentum:
            state.add_optim_state("momentum_buffers", per_param=True)

        # Static (non-tensor) flags persisted so `update` (called arg-less by
        # `apply_gradient`) honors the configured SGD variant. They ride through
        # memmap / masked-select.
        state.add_optim_meta("nesterov", nesterov)
        state.add_optim_meta("maximize", maximize)
        state.add_optim_meta("has_momentum", has_momentum)

        return state

    @classmethod
    def update(
        cls,
        state: State,
        nesterov: bool = False,
        maximize: bool = False,
    ) -> State:
        # `apply_gradient` calls update(state) arg-less, so the static SGD-variant
        # flags come from the optimizer_state where SGD.init stashed them (falling
        # back to the kwarg defaults if a state predates them).
        opt = state.optimizer_state
        opt_keys = opt.keys()
        if "nesterov" in opt_keys:
            nesterov = bool(opt["nesterov"])
        if "maximize" in opt_keys:
            maximize = bool(opt["maximize"])

        active_mask = torch.atleast_1d(state.active_mask)
        params = torch.atleast_2d(state.flat_params)
        grads = torch.atleast_2d(state.flat_grads)
        momentum_buffers = (
            torch.atleast_2d(opt["momentum_buffers"])
            if "momentum_buffers" in opt_keys
            else None
        )
        state_steps = torch.atleast_1d(opt["state_steps"])
        lr = torch.atleast_1d(opt["lr"])
        momentum = torch.atleast_1d(opt["momentum"])
        dampening = torch.atleast_1d(opt["dampening"])
        weight_decay = torch.atleast_1d(opt["weight_decay"])

        sgd_step_(
            params,
            grads,
            momentum_buffers,
            state_steps,
            lr,
            momentum,
            dampening,
            weight_decay,
            active_mask,
            nesterov=nesterov,
            maximize=maximize,
        )
        return state
