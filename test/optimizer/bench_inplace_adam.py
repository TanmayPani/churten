"""Wall-clock + memory benchmark for the new in-place batched adam_step_.

Compares three paths for training R model replicas in parallel:

  1. vanilla:    R separate `torch.optim.Adam` instances, called in a Python loop.
                 (The path most users would write.)
  2. old:        `torch.optim.adam.adam(fused=True)` functional, called per replica.
                 (The path the previous torchstrap `apply_gradient` took.)
  3. torchstrap:    single `adam_step_(...)` call over the stacked (R, *p) state.

Run as `uv run python test/optimizer/bench_inplace_adam.py`.
"""
import gc
import time
from statistics import median

import torch
from torch.optim.adam import adam as torch_adam_func

from torchstrap.optimizer.adam import adam_step_
from torchstrap.state import _CONSOLIDATION_ALIGNMENT


SHAPES = [(2, 512), (512,), (512, 512), (512,), (512, 1), (1,)]

# Real-workload shapes: an MLP ensemble like jet-angularity-study/multifold.py
# (layer_sizes=[nf, 256, 256, 256, 1]) — Linear weight (out,in) + bias (out,)
# per layer. This is the regime where the v2 1D kernel "didn't improve much".
MULTIFOLD_NF = 8
MLP_SHAPES = [
    (256, MULTIFOLD_NF), (256,),
    (256, 256),          (256,),
    (256, 256),          (256,),
    (1, 256),            (1,),
]
NUM_WARMUP = 5
NUM_TIMED = 20


def _make_replica_state(shapes, device, dtype=torch.float32, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    params  = [torch.randn(*s, device=device, dtype=dtype, generator=g, requires_grad=True)
               for s in shapes]
    grads   = [torch.randn(*s, device=device, dtype=dtype, generator=g) for s in shapes]
    for p, gd in zip(params, grads):
        p.grad = gd.clone()
    return params


def _make_stacked_state(R, shapes, device, dtype=torch.float32, seed=0, amsgrad=False):
    g = torch.Generator(device=device).manual_seed(seed)
    params      = [torch.randn(R, *s, device=device, dtype=dtype, generator=g) for s in shapes]
    grads       = [torch.randn(R, *s, device=device, dtype=dtype, generator=g) for s in shapes]
    exp_avgs    = [torch.zeros_like(p) for p in params]
    exp_avg_sqs = [torch.zeros_like(p) for p in params]
    max_es      = [torch.zeros_like(p) for p in params] if amsgrad else []
    state_steps = [torch.zeros(R, device=device, dtype=dtype) for _ in shapes]
    return params, grads, exp_avgs, exp_avg_sqs, max_es, state_steps


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def bench_vanilla_per_replica(R, shapes, device):
    """R independent torch.optim.Adam instances, called in a Python loop."""
    optims = []
    param_lists = []
    for r in range(R):
        ps = _make_replica_state(shapes, device, seed=r)
        param_lists.append(ps)
        optims.append(torch.optim.Adam(
            ps, lr=1e-2, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2,
        ))

    def one_step():
        for opt in optims:
            opt.step()

    _sync(device)
    for _ in range(NUM_WARMUP):
        one_step()
    _sync(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    times = []
    for _ in range(NUM_TIMED):
        _sync(device)
        t0 = time.perf_counter()
        one_step()
        _sync(device)
        times.append(time.perf_counter() - t0)
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    return times, peak


def bench_old_torchstrap(R, shapes, device):
    """torch.optim.adam.adam(fused=True) per replica — the previous torchstrap path."""
    # Build R independent flat states (one per replica), backed by stacked tensors
    # so the memory profile matches what torchstrap holds today.
    stacked = _make_stacked_state(R, shapes, device, amsgrad=False)
    params_S, grads_S, m_S, v_S, _, s_S = stacked
    use_fused = device.type == "cuda"

    def one_step():
        for r in range(R):
            p   = [t[r] for t in params_S]
            g   = [t[r] for t in grads_S]
            m   = [t[r] for t in m_S]
            v   = [t[r] for t in v_S]
            s   = [t[r] for t in s_S]
            torch_adam_func(
                p, g, m, v, [], s,
                amsgrad=False, has_complex=False,
                beta1=0.9, beta2=0.999, lr=1e-2, weight_decay=1e-2, eps=1e-8,
                maximize=False, foreach=False,
                capturable=use_fused, differentiable=False, fused=use_fused,
                grad_scale=None, found_inf=None, decoupled_weight_decay=True,
            )

    _sync(device)
    for _ in range(NUM_WARMUP):
        one_step()
    _sync(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    times = []
    for _ in range(NUM_TIMED):
        _sync(device)
        t0 = time.perf_counter()
        one_step()
        _sync(device)
        times.append(time.perf_counter() - t0)
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    return times, peak


def _bench_step_runner(call_one_step, R, device):
    _sync(device)
    for _ in range(NUM_WARMUP):
        call_one_step()
    _sync(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    times = []
    for _ in range(NUM_TIMED):
        _sync(device)
        t0 = time.perf_counter()
        call_one_step()
        _sync(device)
        times.append(time.perf_counter() - t0)
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    return times, peak


# --------------------------------------------------------------------------- #
# Consolidated single-(R, T) path — what training actually runs now (one fused
# launch over the whole ensemble). This is the shape the pinned Helion config is
# chosen for; keeping it benchmarked guards against the autotuner-trap regression.
# --------------------------------------------------------------------------- #
def _consolidated_T(shapes):
    """Per-replica width of the one (R, T) buffer, exactly as `State` builds it.

    The rounding is not cosmetic and must not be dropped: `State` pads the cat to
    `_CONSOLIDATION_ALIGNMENT` so the kernel reaches ATen's vectorized `load_store`
    path (the guard is `n % kILP == 0`, and every chunk of every row inherits
    `T mod 4`). Benchmarking the raw sum would measure the ragged path — a shape
    training no longer runs, and ~6% slower.
    """
    t = 0
    for s in shapes:
        n = 1
        for d in s:
            n *= d
        t += n
    return t + (-t % _CONSOLIDATION_ALIGNMENT)


def _make_consolidated_state(R, T, device, dtype=torch.float32, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    p = torch.randn(R, T, device=device, dtype=dtype, generator=g)
    grad = torch.randn(R, T, device=device, dtype=dtype, generator=g)
    m = torch.zeros_like(p)
    v = torch.zeros_like(p)
    s = torch.zeros(R, device=device, dtype=dtype)
    return p, grad, m, v, s


def _consolidated_hypers(R, device):
    return (
        torch.full((R,), 1e-2, device=device),   # lr
        torch.full((R,), 0.9, device=device),    # beta1
        torch.full((R,), 0.999, device=device),  # beta2
        torch.full((R,), 1e-8, device=device),   # eps
        torch.full((R,), 1e-2, device=device),   # weight_decay
        torch.ones(R, dtype=torch.bool, device=device),  # active mask
    )


def bench_consolidated(R, shapes, device, frozen_frac=0.0):
    """Single adam_step_ over one consolidated (R, T) buffer — the training path.

    `frozen_frac` deactivates that fraction of replicas, which is what a
    late-training ensemble under EarlyStopping looks like — the kernel skips those
    rows outright (the mask test is block-uniform on CUDA, a `continue` on CPU) and
    moves none of their bytes.
    """
    T = _consolidated_T(shapes)
    p, grad, m, v, s = _make_consolidated_state(R, T, device)
    lr, beta1, beta2, eps, wd, mask = _consolidated_hypers(R, device)
    if frozen_frac:
        mask[: int(R * frozen_frac)] = False

    def one_step():
        adam_step_(
            p, grad, m, v, None, s,
            lr, beta1, beta2, eps, wd, mask,
            amsgrad=False, maximize=False, decoupled_weight_decay=True,
        )

    return _bench_step_runner(one_step, R, device)


def fmt_time(t):
    if t < 1e-3:
        return f"{t*1e6:8.1f} us"
    if t < 1.0:
        return f"{t*1e3:8.2f} ms"
    return f"{t:8.3f} s"


def fmt_mem(b):
    if b is None:
        return "       -- "
    return f"{b/1024/1024:8.1f} MB"


def run(device_name, replica_counts, shapes=SHAPES):
    device = torch.device(device_name)
    from functools import partial

    benches = [
        ("vanilla (per-replica torch.optim.Adam) ", bench_vanilla_per_replica),
        ("old torchstrap (fused functional)         ", bench_old_torchstrap),
        ("consolidated (R,T) adam_step_            ", bench_consolidated),
        ("consolidated adam_step_, 75% frozen      ",
         partial(bench_consolidated, frozen_frac=0.75)),
    ]
    print()
    print(f"=== device={device_name}  shapes={shapes} ===")
    print(f"{'approach':<41s} {'R':>5s} {'median':>11s} {'p10':>11s} {'p90':>11s} {'peak mem':>11s}  {'speedup_vs_vanilla'}")
    for R in replica_counts:
        baseline_median = None
        for label, fn in benches:
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            times, peak = fn(R, shapes, device)
            times.sort()
            med = median(times)
            p10 = times[int(0.1 * len(times))]
            p90 = times[int(0.9 * len(times))]
            if "vanilla" in label and baseline_median is None:
                baseline_median = med
                speedup = "  1.00x"
            else:
                speedup = f"  {baseline_median / med:5.2f}x" if baseline_median else "        "
            print(f"{label} {R:5d} {fmt_time(med):>11s} {fmt_time(p10):>11s} {fmt_time(p90):>11s} {fmt_mem(peak):>11s}  {speedup}")
        print()


if __name__ == "__main__":
    cpu_R = [1, 8, 32]
    cuda_R = [1, 8, 32, 100] if torch.cuda.is_available() else []

    run("cpu", cpu_R)
    if cuda_R:
        run("cuda", cuda_R)
        run("cuda", [10], shapes=MLP_SHAPES)
