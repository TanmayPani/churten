from numbers import Number
from dataclasses import field
from typing import Optional, Self, Any
from collections import defaultdict
from collections.abc import Sequence

from plum import Dispatcher

import torch
from torch.func import stack_module_state
from torch.nn import Module, ModuleList

from tensordict import TensorDict, TensorClass, NonTensorData

_dispatch = Dispatcher()

# Round the consolidated row width `T` up to a multiple of this.
#
# The fused kernels take ATen's vectorized `load_store` path only when
# `n % kILP == 0` (kILP == 4) with `n = T - chunk_offset`. Every chunk offset is a
# multiple of kChunkSize, so `n ≡ T (mod 4)` for every chunk of every row — a `T`
# that is not a multiple of 4 puts the WHOLE ensemble on the ragged
# element-by-element path, not just the last chunk. ATen cannot avoid this (it does
# not own the allocation); we do, because we build the cat.
#
# 4 is the minimum that unlocks it, and it also satisfies the other half of ATen's
# guard: with `T % 4 == 0` every row base `data_ptr + r*T` is 16-byte aligned, which
# is what `is_aligned` checks (kILP * sizeof(float)).
#
# Measured at the spirals width (R=100, T=264705 → 264708), 4 reps: CUDA 3.443 →
# 3.231 ms all-active and 870 → 820 µs at 75% frozen, i.e. +6.6%. CPU is unaffected
# either way — its ragged tail is ≤7 elements out of 264705. 16 was also tried (to
# cover `Vectorized<float>::size()` on AVX2/AVX512) and is consistently ~0.6% slower
# on CUDA for no CPU gain, so it is not worth the extra padding.
_CONSOLIDATION_ALIGNMENT = 4


def _cat_consolidated(
    flat: list[torch.Tensor],
    batch_size: tuple | torch.Size,
    device: str | torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Cat the flattened leaves into one `(*batch_size, T)` buffer.

    `torch.cat([])` raises, so the (rare) param-less case and the (common)
    buffer-less one get an empty `(N, 0)` storage that the offset-table views and
    `functional_call` both handle trivially. An empty consolidation is deliberately
    *not* padded — `(N, 0)` must stay `(N, 0)`.

    The pad lanes are zero and stay zero for the lifetime of the state: the offset
    table never covers them, so `_consolidated_grads.update_` never writes a gradient
    there, and Adam on a zero param with a zero grad has a delta of exactly zero
    (`exp_avg` stays 0, `denom` is `eps`), as does AdamW's `param *= 1 - lr*wd`.
    """
    if not flat:
        return torch.empty((*batch_size, 0), device=device, dtype=dtype)

    storage = torch.cat([t.to(device=device, dtype=dtype) for t in flat], dim=-1)
    pad = -storage.shape[-1] % _CONSOLIDATION_ALIGNMENT
    if pad:
        storage = torch.cat(
            [storage, torch.zeros((*batch_size, pad), device=device, dtype=dtype)],
            dim=-1,
        )
    return storage


def consolidate_params_and_bufffers_dict(
    params_and_buffers_dict: tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]],
    batch_size: Optional[tuple | torch.Size] = None,
    device: Optional[str | torch.device] = None,
    dtype: Optional[torch.dtype] = None,
):
    params_dict, buffers_dict = params_and_buffers_dict
    device = device or next(iter(params_dict.values())).device
    dtype = dtype or next(iter(params_dict.values())).dtype
    batch_size = batch_size or ()
    batch_dims = len(batch_size)

    offset = 0
    consolidated: dict[str, dict[str, Any]] = {
        "params": {"metadata": []},
        "buffers": {"metadata": []},
    }

    flat_params = []
    for t_key, t_val in params_dict.items():
        t_shape = t_val.shape
        if t_shape[:batch_dims] != batch_size:
            raise ValueError(
                f"Tensor at key={t_key} has incompatible shape {t_shape} for batch size {batch_size}"
            )

        flat_params.append(t_val.view(*batch_size, -1))
        num_elements = flat_params[-1].shape[-1]
        consolidated["params"]["metadata"].append(
            (t_key, t_shape, offset, num_elements)
        )
        offset += num_elements
    consolidated["params"]["storage"] = _cat_consolidated(
        flat_params, batch_size, device, dtype
    )

    offset = 0
    flat_buffers = []
    for t_key, t_val in buffers_dict.items():
        t_shape = t_val.shape
        if t_shape[:batch_dims] != batch_size:
            raise ValueError(
                f"Tensor at key={t_key} has incompatible shape {t_shape} for batch size {batch_size}"
            )

        flat_buffers.append(t_val.view(*batch_size, -1))
        num_elements = flat_buffers[-1].shape[-1]
        consolidated["buffers"]["metadata"].append(
            (t_key, t_shape, offset, num_elements)
        )
        offset += num_elements

    consolidated["buffers"]["storage"] = _cat_consolidated(
        flat_buffers, batch_size, device, dtype
    )

    return consolidated


def consolidated_dict_view(tcons: torch.Tensor, metadata: list[tuple]):
    return {
        key: tcons[..., offset : offset + num_elements].view(*shape)
        for key, shape, offset, num_elements in metadata
    }


class State(TensorClass, frozen=True):  # type: ignore[call-arg]
    # `params` / `buffers` are the consolidated, replica-major `(N, T)` flat buffers
    # produced by `consolidate_params_and_bufffers_dict` (row i == replica i; `T` is
    # rounded up to `_CONSOLIDATION_ALIGNMENT` so the fused kernels reach ATen's
    # vectorized path, and the ≤3 pad lanes stay exactly zero — see that constant).
    # The cat IS the source of truth — nothing is re-consolidated
    # through tensordict — so per-replica `lr` / `active_mask` / freeze are exact.
    # The per-name `(N, *shape)` views `functional_call` needs are rebuilt on demand
    # from the offset tables (`param_meta` / `buffer_meta`) as *aliased* views into
    # these buffers; writing through a view writes the consolidated buffer.
    params: torch.Tensor
    buffers: torch.Tensor
    param_meta: Any
    buffer_meta: Any
    param_buffer_dtype: torch.dtype = field(default=torch.float32)
    optimizer_state: TensorDict = field(default_factory=TensorDict)
    active_mask: Optional[torch.Tensor] = None

    # Consolidated `(N, T)` gradient buffer, laid out exactly like `params`.
    # `apply_gradient` scatters the vmap'd per-name grad dict into it via
    # `_consolidated_grads` (aliased views); `Adam.update` reads it back as
    # `flat_grads`. Built in `__post_init__`; frozen forbids rebinding it but
    # allows the in-place scatter.
    grads: torch.Tensor = field(init=False)

    @classmethod
    def from_models(cls: type[Self], modules: Sequence[Module] | ModuleList) -> Self:
        batch_size = (len(modules),)
        consolidated = consolidate_params_and_bufffers_dict(
            stack_module_state(modules), batch_size=batch_size
        )
        params = consolidated["params"]["storage"]
        buffers = consolidated["buffers"]["storage"]

        # The offset tables are *single* python objects (a list per buffer group),
        # not per-replica data — wrap each as one NonTensorData so tensordict stores
        # it opaquely instead of reading its length as a batch dim.
        return cls(
            params=params,
            buffers=buffers,
            param_meta=NonTensorData(consolidated["params"]["metadata"]),
            buffer_meta=NonTensorData(consolidated["buffers"]["metadata"]),
            param_buffer_dtype=params.dtype,
            batch_size=batch_size,
            device=params.device,
        )

    def __post_init__(self):
        # frozen=True installs the dataclass guard as __setattr__, so `self.x = y`
        # raises; the tensorclass isn't locked yet during the *initial* __post_init__,
        # so set the init=False / defaulted fields through the tensorclass setter.
        #
        # Idempotency: tensorclass ops that reconstruct the instance (`.cpu()`,
        # `.detach()`, `state[idx]`, `.clone()`, …) re-run __post_init__ on the
        # already-built — and *locked* — result, where these fields are present.
        # Only set when actually absent so reconstruction doesn't hit the lock.
        if self.get("active_mask", None) is None:
            self.set(
                "active_mask",
                torch.ones(self.batch_size, dtype=torch.bool, device=self.device),
            )
        if self.get("grads", None) is None:
            self.set("grads", torch.zeros_like(self.params))

    @property
    def flat_params(self: Self) -> torch.Tensor:
        return self.params

    @property
    def flat_grads(self: Self) -> torch.Tensor:
        return self.grads

    @property
    def params_dict(self: Self) -> dict[str, torch.Tensor]:
        # Aliased `(N, *shape)` views into the consolidated `params` buffer, rebuilt
        # each access (frozen forbids caching). Fed to `functional_call`.
        return consolidated_dict_view(self.params, self.param_meta)

    @property
    def buffers_dict(self: Self) -> dict[str, torch.Tensor]:
        return consolidated_dict_view(self.buffers, self.buffer_meta)

    @property
    def _consolidated_grads(self: Self) -> TensorDict:
        # Aliased per-name views into the consolidated `grads` buffer. `apply_gradient`
        # does `state._consolidated_grads.update_(grads)` to scatter the per-name grad
        # dict in place; `flat_grads` then reads the populated `(N, T)` buffer back.
        return TensorDict(
            consolidated_dict_view(self.grads, self.param_meta),
            batch_size=self.batch_size,
            device=self.device,
        )

    def add_optim_state(
        self: Self, key: str, value: Any = None, per_param: bool = False
    ):
        # optimizer_state is a child of the frozen (locked) State graph, so adding a
        # new entry (a structural change) requires unlocking the root first, then
        # restoring the lock. Unlike the old tensordict-consolidate path, nothing is
        # cached off the flat buffers, so no rebuild is needed inside the window.
        value_t = self.like_tensor(value, per_param=per_param)
        td = self._tensordict
        was_locked = td.is_locked
        if was_locked:
            td.unlock_()
        try:
            self.optimizer_state[key] = value_t
        finally:
            if was_locked:
                td.lock_()

    def add_optim_meta(self: Self, key: str, value: Any):
        # Store a STATIC (non-tensor) optimizer flag (e.g. amsgrad / maximize /
        # decoupled_weight_decay) as a NonTensorData under optimizer_state — it is
        # not per-replica and must not be tensorized by `like_tensor`. Same
        # unlock/relock dance as add_optim_state (structural change to the frozen
        # graph); NonTensorData auto-unwraps on read and rides through
        # memmap / masked-select.
        td = self._tensordict
        was_locked = td.is_locked
        if was_locked:
            td.unlock_()
        try:
            self.optimizer_state[key] = NonTensorData(value)
        finally:
            if was_locked:
                td.lock_()

    @_dispatch
    def like_tensor(self, value: None, per_param: bool = False):
        if not per_param:
            return torch.zeros(
                self.batch_size,
                dtype=self.param_buffer_dtype,
                device=self.device,
            )
        return torch.zeros_like(self.flat_params, memory_format=torch.preserve_format)

    @_dispatch
    def like_tensor(self, value: int | float | complex, per_param: bool = False):
        if not per_param:
            return torch.full(
                self.batch_size,
                fill_value=value,
                dtype=self.param_buffer_dtype,
                device=self.device,
            )
        return torch.full_like(
            self.flat_params, fill_value=value, memory_format=torch.preserve_format
        )

    @_dispatch
    def like_tensor(self, value: torch.Tensor, per_param: bool = False):
        v = torch.as_tensor(value, dtype=self.param_buffer_dtype, device=self.device)
        if not per_param:
            return v.expand(self.batch_size).clone()

        return v.expand_as(self.flat_params).clone()
