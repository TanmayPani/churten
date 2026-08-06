from typing import Any, Optional
from collections.abc import Sequence

from torch import Tensor
from torch.nn import Module, ModuleList

from torchstrap.state import State


class GradientTransformation(type):
    def __new__(
        cls,
        name: str,
        bases: tuple,
        attrs: dict[str, Any],
    ):
        if "init" not in attrs:
            raise TypeError(
                "Need to define an 'init' method to make an optimizer class!"
            )
        if "update" not in attrs:
            raise TypeError(
                "Need to define an 'update' method to make an optimizer class!"
            )

        return super().__new__(cls, name, bases, attrs)

    def apply_gradient(cls, state: State, grads: dict[str, Tensor]) -> State:
        state._consolidated_grads.update_(grads)
        getattr(cls, "update")(state)
        return state
