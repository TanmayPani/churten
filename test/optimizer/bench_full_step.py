"""Attribution bench: where does the per-batch time go in the real fit loop?

Mirrors ``omnitrain.fit_ensemble``'s inner step on the multifold MLP shape and times the
components separately: the vmapped ``grad_loss`` (fwd+bwd), the grad scatter
(``_consolidated_grads.update_``), and the fused ``Adam.update`` op. Run:
``uv run python test/optimizer/bench_full_step.py``
"""

import time

import torch
from torch.func import vmap, grad_and_value
from torch.nn.functional import binary_cross_entropy_with_logits

from torchstrap.stateless import StatelessModule
from torchstrap.optimizer import Adam
from torchstrap.optimizer.adam import adam_step_
from torchstrap.utils.nn.archs import MLP


def _sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def _time(fn, dev, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    _sync(dev)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync(dev)
    return (time.perf_counter() - t0) / iters * 1e3


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    R, B, nf = 20, 5000, 8
    layer_sizes = [nf, 256, 256, 256, 1]
    print(f"device={dev}  R={R}  B={B}  MLP={layer_sizes}")

    module, state = StatelessModule.init(
        MLP, Adam, layer_sizes=layer_sizes, num_replicas=R, device=str(dev),
        init_randomness="different",
        optimizer_kwargs=dict(lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8,
                              weight_decay=1e-2),
    )
    T = state.params.shape[1]
    print(f"T={T}  buffer {R}x{T} = {R*T*4/1e6:.1f} MB\n")

    x = torch.randn(R, B, nf, device=dev)
    y = (torch.rand(R, B, 1, device=dev) > 0.5).float()

    def loss_fn(params, buffers, x, y):
        return binary_cross_entropy_with_logits(module(params, buffers, x), y)

    grad_loss = vmap(grad_and_value(loss_fn, argnums=0), in_dims=(0, 0, 0, 0),
                     randomness="different")

    grads, _ = grad_loss(state.params_dict, state.buffers_dict, x, y)

    def step_gradloss():
        grad_loss(state.params_dict, state.buffers_dict, x, y)

    def step_scatter():
        state._consolidated_grads.update_(grads)

    def step_adam():
        Adam.update(state)

    def step_full():
        g, _ = grad_loss(state.params_dict, state.buffers_dict, x, y)
        Adam.apply_gradient(state, g)

    t_gl = _time(step_gradloss, dev)
    t_sc = _time(step_scatter, dev)
    t_ad = _time(step_adam, dev)
    t_full = _time(step_full, dev)

    print("per-batch components (ms/iter):")
    print(f"  grad_loss (fwd+bwd)        : {t_gl:7.3f}")
    print(f"  grad scatter (update_)     : {t_sc:7.3f}")
    print(f"  Adam.update (fused op)     : {t_ad:7.3f}")
    print(f"  full step (gl+apply)       : {t_full:7.3f}")
    print(f"  sum of parts               : {t_gl+t_sc+t_ad:7.3f}")

    # Raw adam_step_ op with explicit (R,T) tensors (isolates kernel/config cost).
    p = state.params.clone()
    g = state.grads.clone()
    m = torch.zeros_like(p); v = torch.zeros_like(p)
    steps = torch.zeros(R, device=dev)
    vecs = [torch.full((R,), c, device=dev) for c in (1e-3, 0.9, 0.999, 1e-8, 1e-2)]
    mask = torch.ones(R, dtype=torch.bool, device=dev)

    def raw_adam():
        adam_step_(p, g, m, v, None, steps, *vecs, mask,
                   amsgrad=False, maximize=False, decoupled_weight_decay=True)

    print(f"\n  raw adam_step_ (R={R}, T={T}) : {_time(raw_adam, dev):7.3f} ms")


if __name__ == "__main__":
    main()
