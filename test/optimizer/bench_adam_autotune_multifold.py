"""Does Helion autotune beat the pinned [1,2048] Adam config at the multifold shape?

The committed/efficient run autotuned the Helion kernel (quick, ~5 min). This compares, for
the consolidated one-launch kernel at the multifold shape (R=20, MLP [nf,256,256,256,1]):
  - the pinned config (block_sizes=[1,2048]),
  - the curated sweep (incl. the autotuner's "trapped" [1,128] pick),
  - a real Helion quick-autotune (if the API is available).
Run: ``uv run python test/optimizer/bench_adam_autotune_multifold.py``
"""

import time
from statistics import median

import torch

from bench_inplace_adam import (
    sweep_consolidated_configs, _consolidated_T, _make_consolidated_state,
    _consolidated_hypers, MLP_SHAPES, NUM_WARMUP, NUM_TIMED, fmt_time,
)


def main():
    if not torch.cuda.is_available():
        print("CUDA required."); return
    dev = torch.device("cuda")
    R = 20
    T = _consolidated_T(MLP_SHAPES)
    print(f"multifold consolidated adam: R={R}, T={T}  (buffer {R}x{T} = {R*T*4/1e6:.1f} MB)")

    # 1) curated sweep (includes pinned [1,2048] @ block_n and the trapped [1,128]).
    best_cfg = sweep_consolidated_configs(R, MLP_SHAPES, dev)

    # 2) real Helion quick autotune on this exact shape.
    import helion
    import helion.language as hl
    from torchstrap.kernels.adam import helion_adam_kernel

    _adam_kernel_helion = helion_adam_kernel()

    p, grad, m, v, s = _make_consolidated_state(R, T, dev)
    lr, beta1, beta2, eps, wd, mask = _consolidated_hypers(R, dev)
    bc1 = 1.0 - beta1.pow(s); bc2 = 1.0 - beta2.pow(s)
    args = (p, grad, m, v, p, lr, beta1, beta2, eps, wd, mask.to(torch.uint8), bc1, bc2,
            hl.constexpr(False), hl.constexpr(False), hl.constexpr(True))
    bound = _adam_kernel_helion.bind(args)

    print("\n=== real Helion quick autotune (this may take a few minutes) ===")
    try:
        from helion.autotuner import DifferentialEvolutionSearch  # noqa: F401
        # Quick effort: small population/generations, like the committed run.
        cfg = bound.autotune(args, effort="quick") if "effort" in \
            getattr(bound.autotune, "__doc__", "") or True else bound.autotune(args)
        print(f"autotune picked: {cfg}")
        bound.set_config(cfg)
        _t = _time(bound, args, dev)
        print(f"autotuned time: {fmt_time(_t)}")
    except Exception as e:
        print(f"(autotune via bound.autotune unavailable: {type(e).__name__}: {e})")
        print("The curated sweep above already brackets what autotune would explore.")

    print(f"\npinned config in adam.py is block_sizes=[1,2048]; sweep best = {best_cfg}")


def _time(bound, args, dev):
    for _ in range(NUM_WARMUP):
        bound(*args)
    torch.cuda.synchronize()
    ts = []
    for _ in range(NUM_TIMED):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        bound(*args)
        torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
    return median(ts)


if __name__ == "__main__":
    main()
