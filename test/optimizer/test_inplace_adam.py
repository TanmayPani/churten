"""Numerical equivalence: the consolidated in-place batched ``adam_step_`` vs
``torch.optim.adam.adam`` applied per-replica.

The op now operates on a SINGLE consolidated ``(R, T)`` per-replica buffer (every
parameter of a replica concatenated along T), not a list of per-parameter tensors.
This test builds the reference state as per-parameter lists (so the torch reference
can step each replica/param the classic way) and the new state as the consolidated
``(R, T)`` cat of those same tensors, then asserts they match step for step. Adam is
elementwise per coordinate, so any consistent consolidation order is valid.

"""
import pytest
import torch
from torch.optim.adam import adam as torch_adam

from torchstrap.optimizer.adam import adam_step_


def _consolidate(tensors, R):
    """Pack a list of per-replica ``(R, *shape)`` tensors into one ``(R, T)``."""
    return torch.cat([t.reshape(R, -1) for t in tensors], dim=1).contiguous()


def _ref_per_replica_step(params, grads, exp_avgs, exp_avg_sqs, max_es,
                          state_steps, lr, beta1, beta2, eps, weight_decay,
                          amsgrad, maximize, decoupled_weight_decay):
    """Reference: call torch.optim.adam.adam once per replica (the classic path)."""
    R = params[0].shape[0]
    for r in range(R):
        torch_adam(
            [t[r] for t in params], [t[r] for t in grads],
            [t[r] for t in exp_avgs], [t[r] for t in exp_avg_sqs],
            [t[r] for t in max_es], [t[r] for t in state_steps],
            amsgrad=amsgrad,
            has_complex=False,
            beta1=float(beta1[r]),
            beta2=float(beta2[r]),
            lr=float(lr[r]),
            weight_decay=float(weight_decay[r]),
            eps=float(eps[r]),
            maximize=maximize,
            foreach=False,
            capturable=False,
            differentiable=False,
            fused=False,
            grad_scale=None,
            found_inf=None,
            decoupled_weight_decay=decoupled_weight_decay,
        )


@pytest.mark.parametrize("amsgrad", [False, True])
@pytest.mark.parametrize("maximize", [False, True])
@pytest.mark.parametrize("decoupled_weight_decay", [False, True])
def test_matches_torch_optim_adam(
    device, amsgrad, maximize, decoupled_weight_decay, num_steps=5
):
    R = 4
    shapes = [(5, 7), (7,), (3,)]
    dev = torch.device(device)

    g = torch.Generator(device=dev).manual_seed(42)
    init = [torch.randn(R, *s, device=dev, generator=g) for s in shapes]

    # Reference: per-parameter lists.
    p_ref = [p.clone() for p in init]
    m_ref = [torch.zeros_like(p) for p in p_ref]
    v_ref = [torch.zeros_like(p) for p in p_ref]
    mx_ref = [torch.zeros_like(p) for p in p_ref] if amsgrad else []
    s_ref = [torch.zeros(R, device=dev) for _ in shapes]

    # New: one consolidated (R, T) buffer per state component.
    p_new = _consolidate(init, R)
    m_new = torch.zeros_like(p_new)
    v_new = torch.zeros_like(p_new)
    mx_new = torch.zeros_like(p_new) if amsgrad else None
    s_new = torch.zeros(R, device=dev)

    lr = torch.full((R,), 1e-2, device=dev)
    beta1 = torch.full((R,), 0.9, device=dev)
    beta2 = torch.full((R,), 0.999, device=dev)
    eps = torch.full((R,), 1e-8, device=dev)
    wd = torch.full((R,), 1e-2, device=dev)
    mask = torch.ones(R, device=dev, dtype=torch.bool)

    for step in range(num_steps):
        gg = torch.Generator(device=dev).manual_seed(100 + step)
        grads = [torch.randn(R, *s, device=dev, generator=gg) for s in shapes]

        _ref_per_replica_step(
            p_ref, grads, m_ref, v_ref, mx_ref, s_ref,
            lr, beta1, beta2, eps, wd, amsgrad, maximize, decoupled_weight_decay,
        )
        adam_step_(
            p_new, _consolidate(grads, R), m_new, v_new, mx_new, s_new,
            lr, beta1, beta2, eps, wd, mask,
            amsgrad=amsgrad, maximize=maximize,
            decoupled_weight_decay=decoupled_weight_decay,
        )

    atol = 1e-5 if device == "cuda" else 1e-6
    torch.testing.assert_close(_consolidate(p_ref, R), p_new, atol=atol, rtol=1e-5)
    torch.testing.assert_close(_consolidate(m_ref, R), m_new, atol=atol, rtol=1e-5)
    torch.testing.assert_close(_consolidate(v_ref, R), v_new, atol=atol, rtol=1e-5)
    if amsgrad:
        torch.testing.assert_close(_consolidate(mx_ref, R), mx_new, atol=atol, rtol=1e-5)
    # Shared per-replica step counter matches the (identical) per-param ref steps.
    torch.testing.assert_close(s_ref[0], s_new, atol=atol, rtol=1e-5)
