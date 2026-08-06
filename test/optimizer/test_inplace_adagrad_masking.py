"""Active-mask semantics for the consolidated ``adagrad_step_``:
  - inactive replicas keep their params / state sums / step byte-identical.
  - active replicas advance.
  - edge cases: all-active and all-inactive.

The freeze is load-bearing for Adagrad specifically, not just an optimisation: a
frozen replica must not advance its step counter, because ``corrected_lr = lr / (1 +
(step - 1) * lr_decay)`` reads it. If a frozen replica's clock kept ticking it would
resume on a decayed learning rate it never earned — ``test_frozen_clock_does_not_decay``
pins that.
"""

import pytest
import torch

from torchstrap.optimizer.adagrad import adagrad_step_


def _consolidate(tensors, R):
    return torch.cat([t.reshape(R, -1) for t in tensors], dim=1).contiguous()


def _assert_rows_unchanged(name, before_rows, after_full, idx):
    after_rows = after_full.index_select(0, idx)
    if not torch.equal(before_rows, after_rows):
        diff = (before_rows - after_rows).abs().max().item()
        raise AssertionError(f"{name} inactive rows changed (max diff {diff:.2e})")


def _assert_rows_changed(name, before_full, after_full, idx):
    if torch.equal(before_full.index_select(0, idx), after_full.index_select(0, idx)):
        raise AssertionError(f"{name} active rows did not change")


def _case(device, R, inactive_indices):
    dev = torch.device(device)
    shapes = [(4, 6), (6,), (3,)]
    g = torch.Generator(device=dev).manual_seed(7)
    rnd = lambda: _consolidate(
        [torch.randn(R, *s, device=dev, generator=g) for s in shapes], R
    )
    params, grads = rnd(), rnd()
    state_sums = rnd().abs()
    state_steps = torch.full((R,), 2.0, device=dev)

    mask = torch.ones(R, dtype=torch.bool, device=dev)
    if inactive_indices:
        mask[torch.as_tensor(inactive_indices, device=dev)] = False
    inactive_idx = (~mask).nonzero(as_tuple=False).flatten()
    active_idx = mask.nonzero(as_tuple=False).flatten()

    tracked = [
        ("params", params),
        ("state_sums", state_sums),
        ("state_steps", state_steps),
    ]
    snap_inactive = {
        nm: t.index_select(0, inactive_idx).clone()
        for nm, t in tracked
        if inactive_idx.numel()
    }
    full_pre = {nm: t.clone() for nm, t in tracked}

    full = lambda x: torch.full((R,), x, device=dev)
    adagrad_step_(
        params, grads, state_sums, state_steps,
        full(1e-2), full(1e-2), full(1e-2), full(1e-10), mask,
        maximize=False,
    )

    for nm, t in tracked:
        if inactive_idx.numel():
            _assert_rows_unchanged(nm, snap_inactive[nm], t, inactive_idx)
        if active_idx.numel():
            _assert_rows_changed(nm, full_pre[nm], t, active_idx)


@pytest.mark.parametrize(
    "label,inactive",
    [
        ("all-active", []),
        ("one-inactive", [2]),
        ("multiple-inactive", [0, 3]),
        ("all-inactive", [0, 1, 2, 3, 4]),
    ],
)
def test_frozen_replicas_are_bit_identical(device, label, inactive):
    _case(device, 5, inactive)


def test_frozen_clock_does_not_decay(device):
    """A frozen replica resumes on the learning rate it was frozen at.

    Adagrad's `corrected_lr` depends on the step count, so this is a correctness
    property rather than a performance one: after being frozen for K steps, the
    replica's next update must be identical to the one it would have taken if it had
    never been frozen at all.
    """
    dev = torch.device(device)
    R, T = 2, 32
    lr_decay = 0.5  # exaggerated, so any clock drift is unmissable

    g = torch.Generator(device=dev).manual_seed(3)
    p0 = torch.randn(R, T, device=dev, generator=g)
    grad = torch.randn(R, T, device=dev, generator=g)
    s0 = torch.randn(R, T, device=dev, generator=g).abs()

    full = lambda x: torch.full((R,), x, device=dev)
    hyper = (full(1e-1), full(lr_decay), full(0.0), full(1e-10))

    # (a) replica 0 frozen for 3 steps, then one live step.
    pa, sa = p0.clone(), s0.clone()
    steps_a = torch.zeros(R, device=dev)
    mask = torch.ones(R, dtype=torch.bool, device=dev)
    mask[0] = False
    for _ in range(3):
        adagrad_step_(pa, grad, sa, steps_a, *hyper, mask, maximize=False)
    mask.fill_(True)
    adagrad_step_(pa, grad, sa, steps_a, *hyper, mask, maximize=False)

    # (b) the same replica taking its very first step, never frozen.
    pb, sb = p0.clone(), s0.clone()
    steps_b = torch.zeros(R, device=dev)
    adagrad_step_(
        pb, grad, sb, steps_b, *hyper,
        torch.ones(R, dtype=torch.bool, device=dev), maximize=False,
    )

    assert steps_a[0].item() == 1.0
    assert torch.equal(pa[0], pb[0]), "frozen replica's clock drifted"
    assert torch.equal(sa[0], sb[0])
