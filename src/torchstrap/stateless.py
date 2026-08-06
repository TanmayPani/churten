from itertools import chain
from functools import partial
from copy import deepcopy
from collections.abc import Callable
from typing import Optional, Any, Self, Literal

import torch
from torch import Tensor
from torch.func import functional_call
from torch.nn import Module, ModuleList, Identity

from torchstrap.optimizer import GradientTransformation
from torchstrap.state import State
from torchstrap.history import History
from torchstrap.utils.typing import LoaderT


# Per-batch dtype policy for the (input, target, sample_weight) triple. A `DTypeSpec`
# is one of:
#   "compute"     -> cast to the State's compute dtype (state.dtype); the default,
#                    right for float features / BCE & regression targets / weights.
#   None          -> preserve the tensor's own dtype (device move only); use this for
#                    integer classification targets (cross_entropy) or index inputs.
#   a torch.dtype -> cast to exactly that dtype (e.g. torch.bfloat16 inputs).
DTypeSpec = torch.dtype | Literal["compute"] | None


def _move_batch_member(
    t: Optional[Tensor],
    *,
    device: Optional[torch.device],
    dtype_spec: DTypeSpec,
    compute_dtype: torch.dtype,
    non_blocking: bool = True,
) -> Optional[Tensor]:
    """Move one batch member to ``device`` and apply the ``DTypeSpec`` casting policy."""
    if t is None:
        return None
    dtype = compute_dtype if dtype_spec == "compute" else dtype_spec
    if dtype is None:  # preserve original dtype — device move only
        return t.to(device=device, non_blocking=non_blocking)
    return t.to(device=device, dtype=dtype, non_blocking=non_blocking)


def _init_models(
    model: Module | Callable[..., Module],
    init_randomness: Literal["same", "different"],
    *args: Any,
    num_replicas: int = 1,
    requires_grad: bool = False,
    **kwargs: Any,
) -> ModuleList:
    """Build a `ModuleList` of `num_replicas` replicas.

    A single function (rather than `plum` multiple dispatch) because the natural
    overload split is `model: Module` vs `model: Callable[..., Module]`, and every
    `nn.Module` is itself callable — so those type-sets overlap and plum cannot
    order them, making any Module-instance dispatch ambiguous. We instead branch
    on the disjoint `init_randomness` and resolve Module-vs-factory with
    `isinstance`. `num_replicas`/`requires_grad` are named (not folded into
    `**kwargs`), so only genuine model kwargs (e.g. device/dtype) reach `model()`.
    """
    if init_randomness == "different":
        # N independently-constructed replicas (distinct initial weights): a bare
        # factory/class is called N times; a Module *instance* contributes its
        # class as the factory.
        factory = type(model) if isinstance(model, Module) else model
        return ModuleList(
            factory(*args, **kwargs).requires_grad_(requires_grad)
            for _ in range(num_replicas)
        )

    if init_randomness == "same":
        # All replicas are clones of one shared base: a Module instance is the
        # base directly; a factory/class is called once to build it.
        base = model if isinstance(model, Module) else model(*args, **kwargs)
        to_kwargs = {k: kwargs[k] for k in ("device", "dtype") if k in kwargs}
        return ModuleList(
            deepcopy(base).to(**to_kwargs).requires_grad_(requires_grad)
            for _ in range(num_replicas)
        )

    raise ValueError(
        f"init_randomness must be 'same' or 'different', got {init_randomness!r}"
    )


class StatelessModule(Module):
    _compiled_eval: Optional[Callable] = None
    _compiled_eval_wgrad: Optional[Callable] = None

    def __init__(self, base_model: Module):
        super().__init__()
        self._base_model = base_model.to(device="meta")

    @classmethod
    def init(
        cls: type[Self],
        model: Module | Callable[..., Module],
        optimizer: GradientTransformation,
        *args,
        num_replicas: int = 1,
        requires_grad: bool = False,
        init_randomness: Literal["same", "different"] = "same",
        optimizer_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ):
        if optimizer_kwargs is None:
            optimizer_kwargs = {}

        models = _init_models(
            model,
            init_randomness,
            *args,
            num_replicas=num_replicas,
            requires_grad=requires_grad,
            **kwargs,
        )

        state = State.from_models(models)
        optimizer.init(state, **optimizer_kwargs)

        return cls(models[0]), state

    def forward(
        self,
        params: dict[str, Tensor],
        buffers: dict[str, Tensor],
        *args,
        **kwargs,
    ):
        return functional_call(
            self._base_model,
            (params, buffers),
            args,
            kwargs,
        )
