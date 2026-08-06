"""Active-mask semantics for the consolidated ``adam_step_``:
  - inactive replicas keep their params/moments/step byte-identical.
  - active replicas advance correctly.
  - edge cases: all-active and all-inactive.

The op operates on one consolidated ``(R, T)`` per-replica buffer (every parameter
of a replica packed along T); the mask gates whole rows (replicas), so the row-wise
checks below run directly on the ``(R, T)`` state.
"""
import pytest
import torch

from torchstrap.optimizer.adam import adam_step_


def _consolidate(tensors, R):
    """Pack a list of per-replica ``(R, *shape)`` tensors into one ``(R, T)``."""
    return torch.cat([t.reshape(R, -1) for t in tensors], dim=1).contiguous()


def _build(R, shapes, device, amsgrad=False, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    params = _consolidate([torch.randn(R, *s, device=device, generator=g) for s in shapes], R)
    grads = _consolidate([torch.randn(R, *s, device=device, generator=g) for s in shapes], R)
    exp_avgs = _consolidate([torch.randn(R, *s, device=device, generator=g).abs() for s in shapes], R)
    exp_avg_sqs = _consolidate([torch.randn(R, *s, device=device, generator=g).abs() for s in shapes], R)
    max_es = (
        _consolidate([torch.randn(R, *s, device=device, generator=g).abs() for s in shapes], R)
        if amsgrad else None
    )
    # One shared (R,) step counter, started non-zero (realistic mid-training).
    state_steps = torch.full((R,), 2.0, device=device)
    return params, grads, exp_avgs, exp_avg_sqs, max_es, state_steps


def _assert_rows_unchanged(name, before_rows, after_full, idx):
    after_rows = after_full.index_select(0, idx)
    if not torch.equal(before_rows, after_rows):
        diff = (before_rows - after_rows).abs().max().item()
        raise AssertionError(f"{name} inactive rows changed (max diff {diff:.2e})")


def _assert_rows_changed(name, before_full, after_full, idx):
    a = before_full.index_select(0, idx)
    b = after_full.index_select(0, idx)
    if torch.equal(a, b):
        raise AssertionError(f"{name} active rows did not change")


def _case(device, R, inactive_indices, amsgrad):
    dev = torch.device(device)
    shapes = [(4, 6), (6,), (3,)]
    params, grads, exp_avgs, exp_avg_sqs, max_es, state_steps = _build(
        R, shapes, dev, amsgrad=amsgrad, seed=7,
    )

    mask = torch.ones(R, dtype=torch.bool, device=dev)
    if inactive_indices:
        mask[torch.as_tensor(inactive_indices, device=dev)] = False
    inactive_idx = (~mask).nonzero(as_tuple=False).flatten()
    active_idx = mask.nonzero(as_tuple=False).flatten()

    # Snapshots: inactive rows (must be byte-identical) and full pre-state.
    snap_inactive = {}
    if inactive_idx.numel():
        for nm, t in (("params", params), ("exp_avgs", exp_avgs),
                      ("exp_avg_sqs", exp_avg_sqs), ("state_steps", state_steps)):
            snap_inactive[nm] = t.index_select(0, inactive_idx).clone()
        if amsgrad:
            snap_inactive["max_exp_avg_sqs"] = max_es.index_select(0, inactive_idx).clone()
    full_pre = {"params": params.clone(), "exp_avgs": exp_avgs.clone(),
                "state_steps": state_steps.clone()}

    lr = torch.full((R,), 1e-2, device=dev)
    beta1 = torch.full((R,), 0.9, device=dev)
    beta2 = torch.full((R,), 0.999, device=dev)
    eps = torch.full((R,), 1e-8, device=dev)
    wd = torch.full((R,), 1e-2, device=dev)

    adam_step_(
        params, grads, exp_avgs, exp_avg_sqs, max_es, state_steps,
        lr, beta1, beta2, eps, wd, mask,
        amsgrad=amsgrad, maximize=False, decoupled_weight_decay=True,
    )

    if inactive_idx.numel():
        _assert_rows_unchanged("params", snap_inactive["params"], params, inactive_idx)
        _assert_rows_unchanged("exp_avgs", snap_inactive["exp_avgs"], exp_avgs, inactive_idx)
        _assert_rows_unchanged("exp_avg_sqs", snap_inactive["exp_avg_sqs"], exp_avg_sqs, inactive_idx)
        _assert_rows_unchanged("state_steps", snap_inactive["state_steps"], state_steps, inactive_idx)
        if amsgrad:
            _assert_rows_unchanged("max_exp_avg_sqs", snap_inactive["max_exp_avg_sqs"], max_es, inactive_idx)

    if active_idx.numel():
        _assert_rows_changed("params", full_pre["params"], params, active_idx)
        _assert_rows_changed("exp_avgs", full_pre["exp_avgs"], exp_avgs, active_idx)
        _assert_rows_changed("state_steps", full_pre["state_steps"], state_steps, active_idx)


@pytest.mark.parametrize(
    "label,inactive",
    [
        ("all-active", []),
        ("one-inactive", [2]),
        ("multiple-inactive", [0, 3]),
        ("all-inactive", [0, 1, 2, 3, 4]),
    ],
)
@pytest.mark.parametrize("amsgrad", [False, True])
def test_frozen_replicas_are_bit_identical(device, label, inactive, amsgrad):
    _case(device, 5, inactive, amsgrad)
