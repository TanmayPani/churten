"""Isolation tests for the callable callbacks (no skorch driver).

Builds a tiny 5-replica ensemble State + Adam, then drives each callable directly
the way a manual training loop would, asserting the per-replica / freeze / sync
semantics.
"""
from pathlib import Path

import torch
from torch import nn

from torchstrap.state import State
from torchstrap.optimizer import Adam
from torchstrap.callbacks import (
    EpochScore, EpochTimer, PrintLog, Checkpoint, EarlyStopping,
    LRScheduler, StepLR,
)


def _make_state(n=5, device="cpu"):
    models = [nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1)) for _ in range(n)]
    state = State.from_models(models).to(device)
    Adam.init(state, lr=1e-2, weight_decay=0.0)
    return state


def test_epoch_score():
    # weighted mean matches manual; unweighted is the plain mean.
    b = [torch.tensor([1.0, 2.0, 3.0]), torch.tensor([3.0, 2.0, 1.0])]
    s = EpochScore()
    torch.testing.assert_close(s(b), torch.tensor([2.0, 2.0, 2.0]))
    w = [torch.tensor(1.0), torch.tensor(3.0)]
    expected = (b[0] * 1 + b[1] * 3) / 4
    torch.testing.assert_close(s(b, w), expected)


def test_timer():
    t = EpochTimer()
    t.tic()
    for _ in range(1000):
        pass
    dur = t.toc()
    assert dur >= 0.0 and t.last == dur


def test_lr_scheduler(device):
    state = _make_state(device=device)
    sched = LRScheduler(StepLR, step_size=2, gamma=0.5)
    lr = state.optimizer_state["lr"]
    base = float(lr[0])

    # Freeze replica 0 BEFORE stepping: its clock must not advance -> lr held.
    state.active_mask[0] = False
    for _ in range(4):
        sched(state)
    # active replicas: t=4, floor(4/2)=2 -> base*0.5^2; frozen replica 0 holds base.
    assert abs(float(lr[1]) - base * 0.25) < 1e-6, float(lr[1])
    assert abs(float(lr[0]) - base) < 1e-9, float(lr[0])


def test_checkpoint(device, tmp_path):
    state = _make_state(device=device)
    d = tmp_path
    ckpt = Checkpoint(root_dir=d, verbose=False)
    # Epoch 1: all improve from +inf.
    imp = ckpt(state, torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0], device=device))
    assert bool(imp.all()), imp
    # Capture the saved best params for replica 2.
    best2 = state.params[2].clone()
    # Mutate live params, then a worse score -> no improvement, nothing saved.
    state.params.add_(1.0)
    imp = ckpt(state, torch.tensor([2.0, 2.0, 2.0, 2.0, 2.0], device=device))
    assert not bool(imp.any()), imp
    # load_best must restore replica 2 back to its epoch-1 weights.
    ckpt.load_best(state)
    torch.testing.assert_close(state.params[2], best2)
    assert (Path(d) / "state_dict").is_dir()


def test_early_stopping(device):
    state = _make_state(device=device)
    early = EarlyStopping(patience=2, track_best=True, verbose=False)
    # Replica 0 keeps improving; replicas 1-4 plateau at a constant score.
    stopped = False
    for epoch in range(6):
        score = torch.full((5,), 5.0, device=device)
        score[0] = 5.0 - epoch  # only replica 0 improves
        stopped = early(state, score)
        if stopped:
            break
    # Replicas 1-4 should be frozen (active_mask False); replica 0 still active.
    active = state.active_mask.to("cpu")
    assert bool(active[0]), active
    assert not bool(active[1:].any()), active
    assert not stopped  # replica 0 never plateaus -> not ALL stopped
    early.restore_best(state)


def test_early_stopping_all_stop(device):
    state = _make_state(n=3, device=device)
    early = EarlyStopping(patience=2, track_best=False, verbose=False)
    out = [early(state, torch.ones(3, device=device)) for _ in range(5)]
    assert out[-1] is True, out
    assert not bool(state.active_mask.any())


def test_snapshot_no_clobber(device):
    # Over several updates with a changing improved-mask, non-improved replicas'
    # best rows must never be clobbered (vs a blocking reference) and pinned on CUDA.
    from torchstrap.callbacks.training import Snapshot
    state = _make_state(n=5, device=device)
    snap = Snapshot(state)
    ref = {}  # replica -> expected best params (host)
    for step in range(4):
        # improve a rotating subset
        mask = torch.zeros(5, dtype=torch.bool, device=device)
        mask[step % 5] = True
        mask[(step + 1) % 5] = True
        state.params.add_(1.0)  # change live weights each step
        for i in mask.nonzero(as_tuple=False).flatten().tolist():
            ref[i] = state.params[i].detach().cpu().clone()
        snap.update(state, mask)
    # Every recorded best row must match the mirror.
    for i, expected in ref.items():
        torch.testing.assert_close(snap._snap.params[i], expected)
    if device == "cuda":
        assert snap._staging is not None and snap._staging.params.is_pinned()


def test_printlog():
    rows = []
    log = PrintLog(sink=rows.append)
    log(epoch=1, valid_loss=torch.tensor([0.5, 0.7, 0.9]), valid_loss_best=True, dur=0.01)
    log(epoch=2, valid_loss=torch.tensor([0.4, 0.5, 0.6]), dur=0.02)
    # header + separator + 2 rows
    assert len(rows) == 4, rows
    assert "epoch" in rows[0] and "valid_loss" in rows[0]
