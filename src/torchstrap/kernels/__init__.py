"""Kernels for the fused batched optimizers.

`kernels/adam.py`, `kernels/sgd.py` and `kernels/adagrad.py` own
`torchstrap::adam_step_`, `torchstrap::sgd_step_` and `torchstrap::adagrad_step_`. Each schema and both of its hand-written kernels are C++
(`csrc/`) and register themselves the way ATen's `_fused_adam_` / `_fused_sgd_` /
`_fused_adagrad_` do,
so the only Python left is the `register_fake` / `register_vmap` rules.
`adam_step_` / `sgd_step_` / `adagrad_step_` are the operator handles; `torch.ops.torchstrap.*` are
the same objects.
"""

# Spelled `import a.b._C` rather than `from a.b import _C`: both bind `_C` on this
# package, but only the former is resolvable by a type checker, which sees a
# compiled extension with no stub as an unknown symbol of the package.
import torchstrap.kernels._C  # noqa: F401
from torchstrap.kernels.adam import adam_step_
from torchstrap.kernels.sgd import sgd_step_
from torchstrap.kernels.adagrad import adagrad_step_

__all__ = ("adam_step_", "sgd_step_", "adagrad_step_")
