import shutil
import os
from pathlib import Path
from contextlib import suppress
from beartype.typing import Callable, Optional, Self

import torch

from torchstrap.state import State
from torchstrap.history import History


__all__ = ["Snapshot", "Checkpoint", "EarlyStopping"]


class Snapshot:
    """CPU-resident mirror of the whole ensemble `State` (a frozen ``TensorClass``).

    Holds one CPU `State` (`self._snap`) shaped exactly like the live state
    (`batch_size=[N]`) and supports masked per-replica row update/restore via a
    single bulk device<->host transfer of just the affected rows — `state[idx]`
    gathers/assigns whole replica rows across every leaf at once. `frozen` permits
    these masked-row `__setitem__`s (in-place index_copy) while forbidding field
    rebinding.

    The per-epoch best-row D2H is **pinned + asynchronous**: improved rows are
    gathered on device, copied non-blocking into a reused pinned staging buffer,
    then a single stream sync materializes them before the host-side scatter into
    the mirror — the same pin -> non_blocking -> one-sync discipline as
    ``History.flush_epoch`` (~2x D2H bandwidth vs pageable). Note: TensorClass
    *fancy* indexing returns a **copy**, so the async copy must land in the
    **plain leading slice** ``staging[:k]`` (a real view); only the in-place
    ``__setitem__`` scatters back into ``_snap``.
    """

    def __init__(self, state: State):
        # One bulk device->host transfer of the entire state into a CPU mirror.
        # `.clone()` is load-bearing: on a CPU live state `.cpu()` is a no-op that
        # would otherwise *alias* the live storage (mutations would leak in).
        self._snap: State = state.detach().cpu().clone()
        # Reused pinned (N, ...) buffer giving the async D2H a writable pinned
        # destination; allocated lazily on the first CUDA update (never on CPU).
        self._staging: Optional[State] = None

    @torch.no_grad()
    def update(
        self,
        state: State,
        extra_mask: torch.Tensor | None = None,
    ) -> bool:
        """Copy the active (and ``extra_mask``-selected) replicas' rows from the
        live state into the CPU mirror. **Mask-first early-out:** the `(N,)` mask
        is brought to host first, and when no row improved the whole D2H is
        skipped; otherwise only the improved rows are transferred (pinned + async
        on CUDA). Returns whether any row was committed (host bool) so the caller
        writes a file without its own extra sync.
        """
        mask = state.active_mask
        assert mask is not None
        if extra_mask is not None:
            mask = mask & extra_mask.to(mask.device)

        # Bring just the (N,) mask to host first (tiny; one sync on CUDA). On a
        # no-improvement epoch this is the *only* transfer — the full-state D2H is
        # skipped entirely.
        mask_cpu = mask.to("cpu")
        idx_cpu = mask_cpu.nonzero(as_tuple=False).flatten()
        k = int(idx_cpu.numel())
        if k == 0:
            return False

        idx_dev = idx_cpu.to(state.device)
        gathered = state[idx_dev]  # (k, ...) on the live device
        g_dev = gathered.device
        if g_dev is not None and g_dev.type == "cuda":
            if self._staging is None:
                self._staging = self._snap.clone().pin_memory()
            # Async D2H into the pinned staging slice-view, one sync, then a
            # host->host masked scatter into the mirror.
            self._staging[:k].copy_(gathered, non_blocking=True)  # type: ignore
            torch.cuda.current_stream().synchronize()
            self._snap[idx_cpu] = self._staging[:k]
        else:
            self._snap[idx_cpu] = gathered
        return True

    @torch.no_grad()
    def restore_to_live(
        self,
        state: State,
        mask: torch.Tensor | None = None,
    ) -> None:
        """Write the masked replicas' mirror rows back into the live device state.

        Unlike `update`, restore is **not** gated by `active_mask`: the rows that
        most need restoring at train-end are the *frozen* (inactive) replicas,
        which kept training past their best epoch before being frozen. `mask`
        names exactly the rows to restore; `mask=None` restores every replica
        (one in-place whole-state copy that preserves the param-aliasing invariant).
        """
        if mask is None:
            state.copy_(self._snap.to(state.device))  # type: ignore
            return

        mask_dev = torch.as_tensor(
            mask, dtype=torch.bool, device=state.device
        ).reshape(-1)
        idx_dev = mask_dev.nonzero(as_tuple=False).flatten()
        if idx_dev.numel() == 0:
            return
        idx_cpu = idx_dev.to("cpu")
        # Masked-row assignment is an in-place index_copy across every leaf, so the
        # live params keep their identity (and the forward dict views stay valid).
        state[idx_dev] = self._snap[idx_cpu].to(state.device)

    def to_file(
        self,
        root_dir_path: Path,
        file_name: Optional[str] = None,
    ):
        # TensorClass memmap directory (one contiguous, lazily-loadable checkpoint).
        # Clear any prior target first so re-checkpointing overwrites cleanly.
        target = Path(root_dir_path) / (file_name or "state_dict")
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()
        target.mkdir(parents=True, exist_ok=True)
        self._snap.memmap(str(target))

    @classmethod
    def from_file(
        cls,
        root_dir_path: Path,
        file_name: Optional[str] = None,
        **kwargs,
    ) -> Self:
        target = Path(root_dir_path) / (file_name or "state_dict")
        inst = cls.__new__(cls)
        inst._snap = State.load_memmap(str(target))
        inst._staging = None
        return inst


class Checkpoint:
    """Save the ensemble's best-so-far replica rows when a monitored score improves.

    A plain callable: ``ckpt(state, score)`` compares the per-replica ``(N,)``
    score against the best seen so far, snapshots the **improved** rows into a CPU
    ``Snapshot`` (gated by ``active_mask``), and — if any row improved — writes the
    memmap checkpoint dir (and the optional ``history`` JSON). Returns the ``(N,)``
    bool improved mask. ``load_best(state)`` reloads the saved snapshot and
    restores every row back into the live state (e.g. after training).
    """

    def __init__(
        self,
        lower_is_better: bool = True,
        root_dir: str | os.PathLike = "checkpoint",
        f_state: str = "state_dict",
        f_history: str = "history.json",
        fn_prefix: str = "",
        sink: Callable = print,
        verbose: bool = True,
    ):
        self.lower_is_better = lower_is_better
        self.root_dir = Path(root_dir)
        self.f_state = f_state
        self.f_history = f_history
        self.fn_prefix = fn_prefix
        self.sink = sink
        self.verbose = verbose
        self._snapshot: Optional[Snapshot] = None
        self._best: Optional[torch.Tensor] = None  # (N,) best score so far (host)

    @torch.no_grad()
    def __call__(self, state, score, history: Optional[History] = None):
        score_cpu = torch.as_tensor(score).detach().to("cpu").reshape(-1).float()
        if self._best is None:
            fill = float("inf") if self.lower_is_better else float("-inf")
            self._best = torch.full_like(score_cpu, fill)
        best = self._best

        improved = score_cpu.lt(best) if self.lower_is_better else score_cpu.gt(best)
        self._best = torch.where(improved, score_cpu, best)

        if self._snapshot is None:
            self._snapshot = Snapshot(state)
        any_improved = self._snapshot.update(state, improved.to(state.device))

        if any_improved:
            self.root_dir.mkdir(parents=True, exist_ok=True)
            self._snapshot.to_file(self.root_dir, self.fn_prefix + self.f_state)
            if history is not None:
                history.to_file(self.root_dir / (self.fn_prefix + self.f_history))
            self._sink(
                f"Checkpoint saved ({int(improved.sum())} replica(s) improved).",
                self.verbose,
            )
        return improved

    @torch.no_grad()
    def load_best(self, state, history: Optional[History] = None) -> None:
        with suppress(FileNotFoundError):
            snap = Snapshot.from_file(self.root_dir, self.fn_prefix + self.f_state)
            snap.restore_to_live(state)
            self._sink("Loaded best checkpoint.", self.verbose)
        if history is not None:
            with suppress(FileNotFoundError):
                loaded = History.from_file(
                    self.root_dir / (self.fn_prefix + self.f_history)
                )
                history.clear()
                history.update(loaded)

    def _sink(self, text, verbose):
        if (self.sink is not print) or verbose:
            self.sink(text)


class EarlyStopping:
    """Freeze replicas whose monitored score stops improving; stop when all freeze.

    A plain callable: ``stop = early(state, score)`` per epoch. Each replica keeps
    its own miss counter; when a replica's ``(N,)`` score fails to beat its dynamic
    threshold for ``patience`` epochs it is **frozen in place** via
    ``state.active_mask[i] = False`` (the fused optimizer / LR scheduler then skip
    it). Returns ``True`` once **every** replica is frozen, so the caller can
    ``break``. With ``track_best`` the best-so-far weights are snapshotted and
    ``restore_best(state)`` writes them back at the end.

    Counting, thresholds and freezing are vectorized over the ensemble and gated by
    ``active_mask`` (a frozen replica's counters/threshold hold).
    """

    def __init__(
        self,
        patience: int = 5,
        threshold: float = 1e-4,
        threshold_mode: str = "rel",
        lower_is_better: bool = True,
        track_best: bool = True,
        sink: Callable = print,
        verbose: bool = True,
    ):
        if threshold_mode not in ("rel", "abs"):
            raise ValueError(f"Invalid threshold mode: '{threshold_mode}'")
        self.patience = patience
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.lower_is_better = lower_is_better
        self.track_best = track_best
        self.sink = sink
        self.verbose = verbose
        self.misses_: Optional[torch.Tensor] = None
        self.dynamic_threshold_: Optional[torch.Tensor] = None
        self.best_weights_: Optional[Snapshot] = None
        self.best_epoch_: Optional[torch.Tensor] = None
        self._epoch = 0

    @torch.no_grad()
    def __call__(self, state, score) -> bool:
        score_cpu = torch.as_tensor(score).detach().to("cpu").reshape(-1).float()
        n = score_cpu.shape[0]
        if self.misses_ is None:
            worst = float("inf") if self.lower_is_better else float("-inf")
            self.misses_ = torch.zeros(n, dtype=torch.int)
            self.dynamic_threshold_ = torch.full((n,), worst)
            self.best_weights_ = Snapshot(state) if self.track_best else None
            self.best_epoch_ = torch.zeros(n, dtype=torch.int)
        assert self.dynamic_threshold_ is not None and self.best_epoch_ is not None

        active_cpu = state.active_mask.detach().to("cpu").reshape(-1)
        improved = self._is_score_improved(score_cpu).logical_and_(active_cpu)

        # Advance miss counters / threshold for active replicas only (frozen hold).
        zeros = torch.zeros_like(self.misses_)
        new_misses = torch.where(improved, zeros, self.misses_ + 1)
        self.misses_ = torch.where(active_cpu, new_misses, self.misses_)
        new_threshold = self._calc_new_threshold(score_cpu, improved)
        self.dynamic_threshold_ = torch.where(
            active_cpu, new_threshold, self.dynamic_threshold_
        )
        self.best_epoch_[improved] = self._epoch
        if self.best_weights_ is not None:
            self.best_weights_.update(state, improved.to(state.device))

        # Vectorized freeze: active replicas whose miss count just hit `patience`.
        newly_stopped = (self.misses_ >= self.patience) & active_cpu
        if bool(newly_stopped.any()):
            state.active_mask[newly_stopped.to(state.active_mask.device)] = False
            for i in newly_stopped.nonzero(as_tuple=False).flatten().tolist():
                self._sink(
                    f"Freezing replica {i}: score has not improved in the last "
                    f"{self.patience} epochs.",
                    self.verbose,
                )

        self._epoch += 1
        remaining = active_cpu & (~newly_stopped)
        all_stopped = not bool(remaining.any())
        if all_stopped:
            self._sink(
                "Stopping: no replica's score improved in the last "
                f"{self.patience} epochs.",
                self.verbose,
            )
        return all_stopped

    @torch.no_grad()
    def restore_best(self, state) -> None:
        if self.best_weights_ is None or self.best_epoch_ is None:
            return
        last_epoch = self._epoch - 1
        restore_mask = self.best_epoch_ != last_epoch
        self.best_weights_.restore_to_live(state, restore_mask)
        for i in restore_mask.nonzero(as_tuple=False).flatten().tolist():
            self._sink(
                f"Restoring replica {i} to its best epoch "
                f"{int(self.best_epoch_[i])}.",
                self.verbose,
            )

    def _is_score_improved(self, score: torch.Tensor) -> torch.Tensor:
        threshold = self.dynamic_threshold_
        assert threshold is not None
        if self.lower_is_better:
            return score.lt(threshold)
        return score.gt(threshold)

    def _calc_new_threshold(
        self, score: torch.Tensor, is_improved: torch.Tensor
    ) -> torch.Tensor:
        """Determine the new per-replica threshold from the score."""
        if self.threshold_mode == "rel":
            abs_threshold_change = torch.where(is_improved, self.threshold * score, 0.0)
        else:
            abs_threshold_change = torch.where(
                is_improved, torch.full_like(score, self.threshold), 0.0
            )
        if self.lower_is_better:
            return score - abs_threshold_change
        return score + abs_threshold_change

    def _sink(self, text, verbose):
        if (self.sink is not print) or verbose:
            self.sink(text)
