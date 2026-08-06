"""Attribution bench: does the consolidated ``(R, T)`` param layout slow the vmap forward?

The refactor moved params into one replica-major ``(R, T)`` buffer; ``State.params_dict``
hands ``torch.func.vmap`` **non-contiguous, replica-strided** views
(``buf[:, off:off+n].view(R, *shape)``, dim-0 stride = T). This bench times the
*identical* vmapped ``grad_and_value`` forward+backward over three param layouts that
differ ONLY in memory layout (same numerics):

  A. contiguous stacked  -- HEAD-style ``stack_module_state`` output (batch stride = numel)
  B. strided (R, T)      -- current ``consolidated_dict_view`` views (batch stride = T)
  C. param-segmented     -- per-param contiguous ``(R, n_i)`` blocks (zero-copy contiguous)
  B'. contiguous-copy    -- the minimal fix: ``.contiguous()`` the (B) views each step

If (B) is materially slower than (A)/(C)/(B'), functorch is copying the strided views and
the consolidation is the regression. Run: ``uv run python test/optimizer/bench_vmap_param_layout.py``
"""

import time

import torch
from torch.func import vmap, grad_and_value, stack_module_state
from torch.nn.functional import binary_cross_entropy_with_logits

from torchstrap.utils.nn.archs import MLP
from torchstrap.state import consolidate_params_and_bufffers_dict, consolidated_dict_view


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
    return (time.perf_counter() - t0) / iters * 1e3  # ms/iter


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    R, B, nf = 20, 5000, 8
    layer_sizes = [nf, 256, 256, 256, 1]
    print(f"device={dev}  R={R}  B={B}  MLP={layer_sizes}")

    models = [MLP(layer_sizes=layer_sizes, device=str(dev)) for _ in range(R)]
    params_c, buffers_c = stack_module_state(models)          # contiguous stacked
    params_c = {k: v.detach() for k, v in params_c.items()}
    buffers_c = {k: v.detach() for k, v in buffers_c.items()}

    T = sum(v[0].numel() for v in params_c.values())
    print(f"per-replica numel T={T}  (buffer {R}x{T} = {R*T*4/1e6:.1f} MB fp32)")

    # B: strided views into a replica-major (R, T) cat buffer (current behaviour).
    cons = consolidate_params_and_bufffers_dict((params_c, buffers_c), batch_size=(R,))
    flat = cons["params"]["storage"]
    pmeta = cons["params"]["metadata"]
    params_strided = consolidated_dict_view(flat, pmeta)      # non-contiguous views

    # C: param-segmented contiguous (R, n_i) blocks -> zero-copy contiguous views.
    seg = {k: v.reshape(R, -1).contiguous() for k, v in params_c.items()}
    params_seg = {k: v.view_as(params_c[k]) for k, v in seg.items()}

    x = torch.randn(R, B, nf, device=dev)
    y = (torch.rand(R, B, 1, device=dev) > 0.5).float()

    base = models[0]

    def loss_fn(params, buffers, x, y):
        from torch.func import functional_call
        return binary_cross_entropy_with_logits(
            functional_call(base, (params, buffers), (x,)), y
        )

    gl = vmap(grad_and_value(loss_fn, argnums=0), in_dims=(0, 0, 0, 0))

    # sanity: identical loss across layouts
    _, lA = gl(params_c, buffers_c, x, y)
    _, lB = gl(params_strided, buffers_c, x, y)
    _, lC = gl(params_seg, buffers_c, x, y)
    torch.testing.assert_close(lA, lB)
    torch.testing.assert_close(lA, lC)
    print("loss parity across layouts: OK\n")

    tA = _time(lambda: gl(params_c, buffers_c, x, y), dev)
    tB = _time(lambda: gl(params_strided, buffers_c, x, y), dev)
    tC = _time(lambda: gl(params_seg, buffers_c, x, y), dev)
    tBp = _time(
        lambda: gl({k: v.contiguous() for k, v in params_strided.items()},
                   buffers_c, x, y),
        dev,
    )

    print("vmap grad_and_value fwd+bwd  (ms/iter):")
    print(f"  A  contiguous stacked (HEAD) : {tA:7.3f}")
    print(f"  B  strided (R,T) views (now) : {tB:7.3f}   ({tB/tA:.2f}x vs A)")
    print(f"  C  param-segmented contiguous: {tC:7.3f}   ({tC/tA:.2f}x vs A)")
    print(f"  B' .contiguous() the B views : {tBp:7.3f}   ({tBp/tA:.2f}x vs A)")


if __name__ == "__main__":
    main()
