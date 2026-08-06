"""The `Adam` optimizer — a `GradientTransformation`, not an instance.

This module holds only the optimizer. The `torchstrap::adam_step_` operator and
its two kernels (`csrc/cpu/adam.cpp`, `csrc/cuda/adam.cu`) live in
`torchstrap.kernels.adam`; no dispatch key is claimed from the `optimizer`
package.
"""

import torch

from torchstrap.state import State
from torchstrap.optimizer.grad_transform import GradientTransformation
from torchstrap.kernels.adam import adam_step_

__all__ = ["Adam", "adam_step_"]


class Adam(metaclass=GradientTransformation):
    @classmethod
    def init(
        cls,
        state: State,
        *,
        lr: float | torch.Tensor = 1e-3,
        beta1: float | torch.Tensor = 0.9,
        beta2: float | torch.Tensor = 0.999,
        eps: float | torch.Tensor = 1e-8,
        weight_decay: float | torch.Tensor = 1e-2,
        amsgrad: bool = False,
        maximize: bool = False,
        decoupled_weight_decay: bool = True,
    ) -> State:
        state.add_optim_state("lr", lr)
        state.add_optim_state("beta1", beta1)
        state.add_optim_state("beta2", beta2)
        state.add_optim_state("eps", eps)
        state.add_optim_state("weight_decay", weight_decay)
        state.add_optim_state("state_steps", 0)
        state.add_optim_state("exp_avgs", per_param=True)
        state.add_optim_state("exp_avg_sqs", per_param=True)
        if amsgrad:
            state.add_optim_state("max_exp_avg_sqs", per_param=True)

        # Static (non-tensor) flags persisted so `update` (called arg-less by
        # `apply_gradient`) honors the configured Adam variant — e.g. AdamW via
        # decoupled_weight_decay=True. They ride through memmap / masked-select.
        state.add_optim_meta("amsgrad", amsgrad)
        state.add_optim_meta("maximize", maximize)
        state.add_optim_meta("decoupled_weight_decay", decoupled_weight_decay)

        return state

    @classmethod
    def update(
        cls,
        state: State,
        decoupled_weight_decay: bool = True,
        maximize: bool = False,
        amsgrad: bool = False,
    ) -> State:
        # The optimizer state is self-contained AND consolidated: params and each
        # moment hold a single `(N, T)` buffer, so the fused op runs ONE launch over
        # the whole ensemble. `grads` arrives as the single consolidated `(N, T)`
        # gradient buffer.
        #
        # `apply_gradient` calls update(state) arg-less, so the static Adam-variant
        # flags come from the optimizer_state where Adam.init stashed them (falling
        # back to the kwarg defaults if a state predates them).
        opt = state.optimizer_state
        opt_keys = opt.keys()
        if "amsgrad" in opt_keys:
            amsgrad = bool(opt["amsgrad"])
        if "maximize" in opt_keys:
            maximize = bool(opt["maximize"])
        if "decoupled_weight_decay" in opt_keys:
            decoupled_weight_decay = bool(opt["decoupled_weight_decay"])

        active_mask = torch.atleast_1d(state.active_mask)
        params = torch.atleast_2d(state.flat_params)
        grads = torch.atleast_2d(state.flat_grads)
        exp_avgs = torch.atleast_2d(state.optimizer_state["exp_avgs"])
        exp_avg_sqs = torch.atleast_2d(state.optimizer_state["exp_avg_sqs"])
        max_exp_avg_sqs = (
            torch.atleast_2d(state.optimizer_state["max_exp_avg_sqs"])
            if "max_exp_avg_sqs" in state.optimizer_state.keys()
            else None
        )
        state_steps = torch.atleast_1d(state.optimizer_state["state_steps"])
        lr = torch.atleast_1d(state.optimizer_state["lr"])
        beta1 = torch.atleast_1d(state.optimizer_state["beta1"])
        beta2 = torch.atleast_1d(state.optimizer_state["beta2"])
        eps = torch.atleast_1d(state.optimizer_state["eps"])
        weight_decay = torch.atleast_1d(state.optimizer_state["weight_decay"])

        adam_step_(
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
            amsgrad=amsgrad,
            maximize=maximize,
            decoupled_weight_decay=decoupled_weight_decay,
        )
        return state
