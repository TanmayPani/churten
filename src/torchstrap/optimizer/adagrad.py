"""The `Adagrad` optimizer — a `GradientTransformation`, not an instance.

This module holds only the optimizer. The `torchstrap::adagrad_step_` operator and
its two kernels (`csrc/cpu/adagrad.cpp`, `csrc/cuda/adagrad.cu`) live in
`torchstrap.kernels.adagrad`; no dispatch key is claimed from the `optimizer`
package.
"""

import torch

from torchstrap.state import State
from torchstrap.optimizer.grad_transform import GradientTransformation
from torchstrap.kernels.adagrad import adagrad_step_

__all__ = ["Adagrad", "adagrad_step_"]


class Adagrad(metaclass=GradientTransformation):
    @classmethod
    def init(
        cls,
        state: State,
        *,
        lr: float | torch.Tensor = 1e-2,
        lr_decay: float | torch.Tensor = 0.0,
        weight_decay: float | torch.Tensor = 0.0,
        eps: float | torch.Tensor = 1e-10,
        initial_accumulator_value: float | torch.Tensor = 0.0,
        maximize: bool = False,
    ) -> State:
        state.add_optim_state("lr", lr)
        state.add_optim_state("lr_decay", lr_decay)
        state.add_optim_state("weight_decay", weight_decay)
        state.add_optim_state("eps", eps)
        state.add_optim_state("state_steps", 0)
        state.add_optim_state("state_sums", per_param=True)

        # `torch.optim.Adagrad` seeds the accumulator with a constant (default 0).
        # It is a plain fill of the consolidated `(N, T)` buffer, including the pad
        # lanes — harmless, since the offset table never covers them and no gradient
        # is ever written there, so those lanes contribute nothing.
        if not isinstance(initial_accumulator_value, torch.Tensor):
            if initial_accumulator_value != 0.0:
                state.optimizer_state["state_sums"].fill_(initial_accumulator_value)
        else:
            state.optimizer_state["state_sums"].copy_(
                initial_accumulator_value.expand_as(
                    state.optimizer_state["state_sums"]
                )
            )

        # Static (non-tensor) flag persisted so `update` (called arg-less by
        # `apply_gradient`) honors the configured variant. It rides through memmap /
        # masked-select.
        state.add_optim_meta("maximize", maximize)

        return state

    @classmethod
    def update(cls, state: State, maximize: bool = False) -> State:
        # `apply_gradient` calls update(state) arg-less, so the static flag comes
        # from the optimizer_state where Adagrad.init stashed it.
        opt = state.optimizer_state
        opt_keys = opt.keys()
        if "maximize" in opt_keys:
            maximize = bool(opt["maximize"])

        active_mask = torch.atleast_1d(state.active_mask)
        params = torch.atleast_2d(state.flat_params)
        grads = torch.atleast_2d(state.flat_grads)
        state_sums = torch.atleast_2d(opt["state_sums"])
        state_steps = torch.atleast_1d(opt["state_steps"])
        lr = torch.atleast_1d(opt["lr"])
        lr_decay = torch.atleast_1d(opt["lr_decay"])
        weight_decay = torch.atleast_1d(opt["weight_decay"])
        eps = torch.atleast_1d(opt["eps"])

        adagrad_step_(
            params,
            grads,
            state_sums,
            state_steps,
            lr,
            lr_decay,
            weight_decay,
            eps,
            active_mask,
            maximize=maximize,
        )
        return state
