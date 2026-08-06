"""Vectorized, per-replica learning-rate scheduler callbacks for ensemble training.

torchstrap trains N model replicas in parallel; the learning rate lives as a
per-replica ``(N,)`` ``Vector`` at ``state.optimizer_state.lr`` and is consumed
directly each step by the fused ``adam_step_`` op. These schedulers therefore
operate on that ``(N,)`` vector **on-device**, writing it **in place**, and are
**freeze-aware**: each replica carries its own schedule clock that only advances
while the replica is active (``state.active_mask``). This is what makes a genuine
*per-replica* ``ReduceLROnPlateau`` possible.

Nothing here depends on ``torch.optim`` — the policies are native vectorized
closed-form / state-machine functions of a per-replica step counter.
"""

import sys
import math
from dataclasses import dataclass

from beartype.typing import Optional, Callable, Sequence, Union

import torch
from torch import Tensor

__all__ = [
    'LRScheduler',
    'StepLR',
    'MultiStepLR',
    'ExponentialLR',
    'CosineAnnealingLR',
    'LambdaLR',
    'WarmRestartLR',
    'ReduceLROnPlateau',
]


def _as_vec(val, num_replicas: int, device: torch.device) -> Tensor:
    """Broadcast a scalar / sequence hyperparameter to a ``(N,)`` float tensor."""
    t = torch.as_tensor(val, dtype=torch.float32, device=device)
    if t.ndim == 0:
        t = t.expand(num_replicas).clone()
    return t


@dataclass
class ScheduleContext:
    """Per-step inputs handed to a policy. All tensors are ``(N,)`` on the lr device."""
    base_lrs: Tensor          # lr captured at train begin (.clone())
    t: Tensor                 # per-replica clock (advances only while active)
    current_lr: Tensor        # live state.optimizer_state.lr, flattened to (N,)
    active_mask: Tensor       # (N,) bool
    score: Optional[Tensor] = None   # monitored epoch metric (plateau only)


class _Schedule:
    """Base policy. ``__call__`` returns the new ``(N,)`` lr vector."""

    needs_score: bool = False
    allows_batch_step: bool = True

    def reset(self, num_replicas: int, device: torch.device) -> None:
        """(Re)allocate any per-replica state on the lr device."""

    def __call__(self, ctx: ScheduleContext) -> Tensor:
        raise NotImplementedError


# -----------------------------------------------------------------------------
# Closed-form policies: pure functions of base_lrs and the per-replica clock t.
# -----------------------------------------------------------------------------
class StepLR(_Schedule):
    def __init__(self, step_size, gamma=0.1, **kwargs):
        self.step_size = step_size
        self.gamma = gamma

    def __call__(self, ctx: ScheduleContext) -> Tensor:
        exponent = torch.floor(ctx.t / self.step_size)
        return ctx.base_lrs * (self.gamma ** exponent)


class MultiStepLR(_Schedule):
    def __init__(self, milestones, gamma=0.1, **kwargs):
        self.milestones = sorted(milestones)
        self.gamma = gamma
        self._m: Optional[Tensor] = None

    def reset(self, num_replicas: int, device: torch.device) -> None:
        self._m = torch.as_tensor(self.milestones, dtype=torch.float32, device=device)

    def __call__(self, ctx: ScheduleContext) -> Tensor:
        assert self._m is not None
        counts = (ctx.t[:, None] >= self._m[None, :]).sum(dim=-1).to(ctx.base_lrs.dtype)
        return ctx.base_lrs * (self.gamma ** counts)


class ExponentialLR(_Schedule):
    def __init__(self, gamma, **kwargs):
        self.gamma = gamma

    def __call__(self, ctx: ScheduleContext) -> Tensor:
        return ctx.base_lrs * (self.gamma ** ctx.t)


class CosineAnnealingLR(_Schedule):
    def __init__(self, T_max, eta_min=0.0, **kwargs):
        self.T_max = T_max
        self.eta_min = eta_min

    def __call__(self, ctx: ScheduleContext) -> Tensor:
        cos = torch.cos(math.pi * ctx.t / self.T_max)
        return self.eta_min + 0.5 * (ctx.base_lrs - self.eta_min) * (1.0 + cos)


class LambdaLR(_Schedule):
    """``lr = base_lrs * lr_lambda(t)``.

    ``lr_lambda`` receives the per-replica clock as a ``(N,)`` float tensor and
    must return a tensor (or scalar) factor that broadcasts over ``(N,)``.
    """

    def __init__(self, lr_lambda: Callable[[Tensor], object], **kwargs):
        self.lr_lambda = lr_lambda

    def __call__(self, ctx: ScheduleContext) -> Tensor:
        factor = self.lr_lambda(ctx.t)
        if not isinstance(factor, Tensor):
            factor = torch.as_tensor(
                factor, dtype=ctx.base_lrs.dtype, device=ctx.base_lrs.device,
            )
        return ctx.base_lrs * factor


class WarmRestartLR(_Schedule):
    """Stochastic Gradient Descent with Warm Restarts (SGDR), vectorized over N.

    Uses its own per-replica ``min_lr`` / ``max_lr`` (not the captured base lr),
    matching the original semantics. The per-replica restart-period search is a
    fixed-length tensor loop (no host sync) so it is safe for batch stepping.

    References
    ----------
    Ilya Loshchilov and Frank Hutter, 2017, "Stochastic Gradient Descent with
    Warm Restarts," ICLR. https://arxiv.org/pdf/1608.03983.pdf
    """

    def __init__(
        self,
        min_lr=1e-6,
        max_lr=0.05,
        base_period=10,
        period_mult=2,
        max_restarts=64,
        **kwargs,
    ):
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.base_period = base_period
        self.period_mult = period_mult
        self.max_restarts = max_restarts
        self._min: Optional[Tensor] = None
        self._max: Optional[Tensor] = None

    def reset(self, num_replicas: int, device: torch.device) -> None:
        self._min = _as_vec(self.min_lr, num_replicas, device)
        self._max = _as_vec(self.max_lr, num_replicas, device)

    def __call__(self, ctx: ScheduleContext) -> Tensor:
        assert self._min is not None and self._max is not None
        epoch_idx = ctx.t.clone()
        period = torch.full_like(epoch_idx, float(self.base_period))
        # Fixed-length, branchless restart search: per replica, peel off whole
        # restart periods while epoch_idx/period > 1. Bounded by max_restarts so
        # there is no `.any()` host sync (safe inside the per-batch loop).
        for _ in range(self.max_restarts):
            past = (epoch_idx / period) > 1.0
            epoch_idx = torch.where(past, epoch_idx - (period + 1), epoch_idx)
            period = torch.where(past, period * self.period_mult, period)
        cos = torch.cos(epoch_idx * math.pi / period)
        return self._min + 0.5 * (self._max - self._min) * (1.0 + cos)


# -----------------------------------------------------------------------------
# Stateful per-replica plateau policy.
# -----------------------------------------------------------------------------
class ReduceLROnPlateau(_Schedule):
    """Reduce lr per replica when its monitored score stops improving.

    Unlike ``torch.optim.lr_scheduler.ReduceLROnPlateau`` (one scalar metric for
    the whole optimizer), this runs one plateau state machine per replica over
    the ``(N,)`` epoch score, fully vectorized and gated by ``active_mask`` so a
    frozen replica's counters and lr stay put.
    """

    needs_score = True
    allows_batch_step = False

    def __init__(
        self,
        mode='min',
        factor=0.1,
        patience=10,
        threshold=1e-4,
        threshold_mode='rel',
        cooldown=0,
        min_lr=0.0,
        eps=1e-8,
        **kwargs,
    ):
        if mode not in ('min', 'max'):
            raise ValueError(f"Invalid mode '{mode}', expected 'min' or 'max'.")
        if threshold_mode not in ('rel', 'abs'):
            raise ValueError(
                f"Invalid threshold_mode '{threshold_mode}', expected 'rel' or 'abs'."
            )
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.cooldown = cooldown
        self.min_lr = min_lr
        self.eps = eps
        self.best: Optional[Tensor] = None
        self.num_bad: Optional[Tensor] = None
        self.cooldown_counter: Optional[Tensor] = None
        self._min_lr: Optional[Tensor] = None

    def reset(self, num_replicas: int, device: torch.device) -> None:
        worst = float('inf') if self.mode == 'min' else float('-inf')
        self.best = torch.full((num_replicas,), worst, dtype=torch.float32, device=device)
        self.num_bad = torch.zeros(num_replicas, dtype=torch.float32, device=device)
        self.cooldown_counter = torch.zeros(num_replicas, dtype=torch.float32, device=device)
        self._min_lr = _as_vec(self.min_lr, num_replicas, device)

    def _is_better(self, score: Tensor, best: Tensor) -> Tensor:
        if self.threshold_mode == 'rel':
            if self.mode == 'min':
                return score < best * (1.0 - self.threshold)
            return score > best * (1.0 + self.threshold)
        if self.mode == 'min':
            return score < best - self.threshold
        return score > best + self.threshold

    def __call__(self, ctx: ScheduleContext) -> Tensor:
        assert (
            self.best is not None and self.num_bad is not None
            and self.cooldown_counter is not None and self._min_lr is not None
        )
        assert ctx.score is not None, "ReduceLROnPlateau requires a monitored score."
        score = ctx.score
        active = ctx.active_mask
        zeros = torch.zeros_like(self.num_bad)

        improved = self._is_better(score, self.best)
        best_new = torch.where(improved, score, self.best)
        num_bad = torch.where(improved, zeros, self.num_bad + 1)

        in_cooldown = self.cooldown_counter > 0
        cd_dec = torch.where(in_cooldown, self.cooldown_counter - 1, self.cooldown_counter)
        num_bad = torch.where(in_cooldown, zeros, num_bad)

        reduce = num_bad > self.patience
        reduced_lr = torch.maximum(ctx.current_lr * self.factor, self._min_lr)
        new_lr = torch.where(reduce, reduced_lr, ctx.current_lr)
        cd_final = torch.where(
            reduce, torch.full_like(self.cooldown_counter, float(self.cooldown)), cd_dec,
        )
        num_bad = torch.where(reduce, zeros, num_bad)

        # Gate every state update by `active`: frozen replicas keep their best /
        # counters / lr unchanged.
        self.best = torch.where(active, best_new, self.best)
        self.num_bad = torch.where(active, num_bad, self.num_bad)
        self.cooldown_counter = torch.where(active, cd_final, self.cooldown_counter)
        return torch.where(active, new_lr, ctx.current_lr)


class LRScheduler:
    """Drive a vectorized LR policy over ``state.optimizer_state["lr"]``.

    A plain callable: ``sched(state)`` advances the schedule one tick and writes
    the new ``(N,)`` lr **in place** into ``state.optimizer_state["lr"]`` (so the
    fused ``adam_step_`` consumes it directly and it is checkpointed for free).
    Cadence is by placement — call it once per epoch, or once per batch. Plateau
    policies need the monitored ``(N,)`` epoch score, passed as ``score=``.

    The per-replica clock advances only for **active** replicas
    (``state.active_mask``), so a frozen replica holds its last lr. Internal
    state (``base_lrs_``/``t_``/the policy) is lazily captured on the first call.

    Parameters
    ----------
    policy : str | type
        A policy class in this module (or its name), e.g. ``'CosineAnnealingLR'``,
        ``'WarmRestartLR'``, ``'ReduceLROnPlateau'``.
    **kwargs
        Forwarded to the policy constructor (e.g. ``T_max=``, ``gamma=``,
        ``patience=``).
    """

    def __init__(self, policy='WarmRestartLR', **kwargs):
        self.policy = policy
        self.kwargs = kwargs
        self.policy_ = None
        self.base_lrs_ = None
        self.t_ = None

    def _get_policy_cls(self):
        if isinstance(self.policy, str):
            return getattr(sys.modules[__name__], self.policy)
        return self.policy

    def _lazy_init(self, state):
        policy_cls = self._get_policy_cls()
        self.policy_ = policy_cls(**self.kwargs)
        lr = state.optimizer_state["lr"]
        device = lr.device
        self.base_lrs_ = lr.detach().reshape(-1).clone()
        num_replicas = self.base_lrs_.shape[0]
        self.t_ = torch.zeros(num_replicas, dtype=torch.float32, device=device)
        self.policy_.reset(num_replicas, device)

    def __call__(self, state, score=None):
        if self.policy_ is None:
            self._lazy_init(state)
        assert self.policy_ is not None and self.t_ is not None
        assert self.base_lrs_ is not None

        lr = state.optimizer_state["lr"]
        shape = lr.shape
        current = lr.detach().reshape(-1)
        active = state.active_mask.reshape(-1).to(current.device)

        # Advance the clock only for active replicas (frozen ones hold their lr).
        self.t_ = self.t_ + active.to(self.t_.dtype)

        score_vec = None
        if self.policy_.needs_score:
            assert score is not None, (
                f"{type(self.policy_).__name__} needs a score; pass score=..."
            )
            score_vec = torch.as_tensor(
                score, dtype=current.dtype, device=current.device,
            ).reshape(-1)

        ctx = ScheduleContext(
            base_lrs=self.base_lrs_,
            t=self.t_,
            current_lr=current,
            active_mask=active,
            score=score_vec,
        )
        new_lr = self.policy_(ctx)
        lr.copy_(new_lr.reshape(shape))
        return lr.detach()

    def simulate(self, steps: int, initial_lr: float):
        """Host-side preview of a closed-form schedule's lr curve (for plotting).

        Returns a list of ``steps`` lr values for a single replica. Not supported
        for plateau policies (which need a score stream).
        """
        policy = self._get_policy_cls()(**self.kwargs)
        if policy.needs_score:
            raise ValueError("simulate is only supported for closed-form schedules.")
        device = torch.device('cpu')
        base = torch.as_tensor([float(initial_lr)], dtype=torch.float32, device=device)
        active = torch.ones(1, dtype=torch.bool, device=device)
        policy.reset(1, device)
        lrs = []
        for k in range(steps):
            t = torch.full((1,), float(k + 1), dtype=torch.float32, device=device)
            ctx = ScheduleContext(base, t, base.clone(), active, None)
            lrs.append(float(policy(ctx)[0]))
        return lrs
