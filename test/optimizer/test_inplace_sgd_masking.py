"""Active-mask semantics for the consolidated ``sgd_step_``:
  - inactive replicas keep their params / momentum buffers / step byte-identical.
  - active replicas advance.
  - edge cases: all-active, all-inactive, and a replica frozen at ``state_steps == 0``.

That last one is the case ATen cannot get right and the reason SGD carries a
``(R,)`` step counter at all: upstream's ``is_first_step`` is one host bool for the
whole call, so a replica frozen before it ever stepped would be told "not the first
step" and would read an uninitialised momentum buffer the moment it thawed. Here the
mask keeps it out of the kernel entirely and its clock does not advance, so when it
resumes it still sees step 1.
"""

import pytest
import torch

from torchstrap.optimizer.sgd import sgd_step_


def _consolidate(tensors, R):
    return torch.cat([t.reshape(R, -1) for t in tensors], dim=1).contiguous()


def _build(R, shapes, device, has_momentum, seed=0, step=2.0):
    g = torch.Generator(device=device).manual_seed(seed)
    rnd = lambda: _consolidate(
        [torch.randn(R, *s, device=device, generator=g) for s in shapes], R
    )
    params, grads = rnd(), rnd()
    mb = rnd() if has_momentum else None
    state_steps = torch.full((R,), step, device=device)
    return params, grads, mb, state_steps


def _assert_rows_unchanged(name, before_rows, after_full, idx):
    after_rows = after_full.index_select(0, idx)
    if not torch.equal(before_rows, after_rows):
        diff = (before_rows - after_rows).abs().max().item()
        raise AssertionError(f"{name} inactive rows changed (max diff {diff:.2e})")


def _assert_rows_changed(name, before_full, after_full, idx):
    if torch.equal(before_full.index_select(0, idx), after_full.index_select(0, idx)):
        raise AssertionError(f"{name} active rows did not change")


def _case(device, R, inactive_indices, has_momentum, step=2.0):
    dev = torch.device(device)
    shapes = [(4, 6), (6,), (3,)]
    params, grads, mb, state_steps = _build(
        R, shapes, dev, has_momentum, seed=7, step=step
    )

    mask = torch.ones(R, dtype=torch.bool, device=dev)
    if inactive_indices:
        mask[torch.as_tensor(inactive_indices, device=dev)] = False
    inactive_idx = (~mask).nonzero(as_tuple=False).flatten()
    active_idx = mask.nonzero(as_tuple=False).flatten()

    tracked = [("params", params), ("state_steps", state_steps)]
    if has_momentum:
        tracked.append(("momentum_buffers", mb))

    snap_inactive = {
        nm: t.index_select(0, inactive_idx).clone()
        for nm, t in tracked
        if inactive_idx.numel()
    }
    full_pre = {nm: t.clone() for nm, t in tracked}

    lr = torch.full((R,), 1e-2, device=dev)
    momentum = torch.full((R,), 0.9 if has_momentum else 0.0, device=dev)
    dampening = torch.zeros(R, device=dev)
    wd = torch.full((R,), 1e-2, device=dev)

    sgd_step_(
        params, grads, mb, state_steps, lr, momentum, dampening, wd, mask,
        nesterov=False, maximize=False,
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
@pytest.mark.parametrize("has_momentum", [False, True])
def test_frozen_replicas_are_bit_identical(device, label, inactive, has_momentum):
    _case(device, 5, inactive, has_momentum)


def test_replica_frozen_before_its_first_step(device):
    """Freezing at step 0 must leave the momentum buffer pristine and the clock at 0.

    Its buffer would otherwise be read as if initialised — the exact failure ATen's
    scalar ``is_first_step`` cannot avoid for an ensemble.
    """
    dev = torch.device(device)
    R = 4
    T = 16
    params = torch.randn(R, T, device=dev)
    grads = torch.randn(R, T, device=dev)
    mb = torch.zeros(R, T, device=dev)
    state_steps = torch.zeros(R, device=dev)

    mask = torch.ones(R, dtype=torch.bool, device=dev)
    mask[1] = False

    lr = torch.full((R,), 1e-2, device=dev)
    momentum = torch.full((R,), 0.9, device=dev)
    dampening = torch.zeros(R, device=dev)
    wd = torch.zeros(R, device=dev)

    for _ in range(3):
        sgd_step_(
            params, grads, mb, state_steps, lr, momentum, dampening, wd, mask,
            nesterov=False, maximize=False,
        )

    assert state_steps[1].item() == 0.0
    assert torch.equal(mb[1], torch.zeros(T, device=dev))
    assert state_steps[0].item() == 3.0

    # Thawing it now must run its *first* step, not step 4: buf = grad exactly.
    mask.fill_(True)
    sgd_step_(
        params, grads, mb, state_steps, lr, momentum, dampening, wd, mask,
        nesterov=False, maximize=False,
    )
    assert torch.equal(mb[1], grads[1])
