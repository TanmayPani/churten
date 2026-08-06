"""Wall-clock benchmark for the consolidated batched ``adagrad_step_``.

Same shape as ``bench_sgd.py``: R separate ``torch.optim.Adagrad`` instances in a
Python loop, versus one ``adagrad_step_`` over a single ``(R, T)`` buffer, versus the
same at 75% frozen (what a late-training ensemble under EarlyStopping looks like).

**The CUDA number carries a large fp64 tax, and it is bigger than it looks.** ATen's
`adagrad_math` takes its hyperparameters as `const double&`, so the final update — a
double *division* — runs in fp64 on the device; on a consumer card (4070: 1/64 fp64
rate) that dominates. Measured at R=1, T=264708 on a 4070 Laptop: **335 us for
adagrad_step_ against 39 us for sgd_step_ (8.6x)**, even though the two move exactly
the same bytes (param r+w, grad r, third buffer r+w = 20 bytes/element). And the
kernel is **not memory-bound**: at R=1 the time barely moves from T=264708 to
T=1058832 (335 -> 339 us), because with so few blocks each thread is serialising fp64
latency it has no parallelism to hide. At R=100 it is 5.1 ms against SGD's 2.4 ms and
Adam's 3.2 ms, despite Adam moving 28 bytes/element to Adagrad's 20.

Consequence worth knowing before choosing this optimizer: **at small R the fused path
is *slower* than the Python loop** (R=1: 0.31x). It only wins once there are enough
replicas to fill the machine.

None of this is fixable without giving up the guarantee. The fp64 is ATen's own
`adagrad_math`, included verbatim, and narrowing it to float would be editing upstream
inside our copy — `test_aten_adagrad_parity.py` would fail immediately. So the number
is recorded rather than optimised away; see `csrc/cuda/adagrad.cu`. A separate
float32 variant, sold as such and not bit-exact, is a legitimate thing to want, but it
is a new feature and a deliberate choice, not a fix to this one.

Not collected by pytest (the ``bench_*`` convention). Run as
``uv run python test/optimizer/bench_adagrad.py``.
"""

import gc
import time
from functools import partial
from statistics import median

import torch

from torchstrap.optimizer.adagrad import adagrad_step_
from torchstrap.state import _CONSOLIDATION_ALIGNMENT

# Same shapes as bench_inplace_adam.py / bench_sgd.py, so all three are comparable.
SHAPES = [(2, 512), (512,), (512, 512), (512,), (512, 1), (1,)]
NUM_WARMUP = 5
NUM_TIMED = 20


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _consolidated_T(shapes):
    """Per-replica width of the one ``(R, T)`` buffer, exactly as ``State`` builds it.

    The rounding is not cosmetic: ``State`` pads the cat to
    ``_CONSOLIDATION_ALIGNMENT`` so the kernel reaches ATen's vectorized
    ``load_store`` path. Benchmarking the raw sum would measure the ragged path — a
    shape training no longer runs.
    """
    t = 0
    for s in shapes:
        n = 1
        for d in s:
            n *= d
        t += n
    return t + (-t % _CONSOLIDATION_ALIGNMENT)


def _bench(call_one_step, device):
    # Let the GPU settle first; see the note in bench_sgd.py. Without it a row late
    # in the sweep is measured on a downclocked card.
    if device.type == "cuda":
        _sync(device)
        time.sleep(0.5)

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


def bench_vanilla(R, shapes, device):
    optims = []
    for r in range(R):
        g = torch.Generator(device=device).manual_seed(r)
        ps = [
            torch.randn(*s, device=device, generator=g, requires_grad=True)
            for s in shapes
        ]
        for p in ps:
            p.grad = torch.randn(*p.shape, device=device, generator=g)
        optims.append(
            torch.optim.Adagrad(ps, lr=1e-2, lr_decay=1e-2, weight_decay=1e-2)
        )

    def one_step():
        for opt in optims:
            opt.step()

    return _bench(one_step, device)


def bench_consolidated(R, shapes, device, frozen_frac=0.0):
    T = _consolidated_T(shapes)
    g = torch.Generator(device=device).manual_seed(0)
    p = torch.randn(R, T, device=device, generator=g)
    grad = torch.randn(R, T, device=device, generator=g)
    sums = torch.zeros_like(p)
    steps = torch.zeros(R, device=device)

    full = lambda x: torch.full((R,), x, device=device)
    lr, lrd, wd, eps = full(1e-2), full(1e-2), full(1e-2), full(1e-10)
    mask = torch.ones(R, dtype=torch.bool, device=device)
    if frozen_frac:
        mask[: int(R * frozen_frac)] = False

    def one_step():
        adagrad_step_(p, grad, sums, steps, lr, lrd, wd, eps, mask, maximize=False)

    return _bench(one_step, device)


def fmt_time(t):
    if t < 1e-3:
        return f"{t * 1e6:8.1f} us"
    if t < 1.0:
        return f"{t * 1e3:8.2f} ms"
    return f"{t:8.3f} s"


def fmt_mem(b):
    return "       -- " if b is None else f"{b / 1024 / 1024:8.1f} MB"


def run(device_name, replica_counts, shapes=SHAPES):
    device = torch.device(device_name)
    benches = [
        ("vanilla (per-replica torch.optim.Adagrad)", bench_vanilla),
        ("consolidated (R,T) adagrad_step_         ", bench_consolidated),
        ("consolidated adagrad_step_, 75% frozen   ",
         partial(bench_consolidated, frozen_frac=0.75)),
    ]
    print()
    print(f"=== device={device_name}  T={_consolidated_T(shapes)} ===")
    print(f"{'approach':<41s} {'R':>5s} {'median':>11s} {'p10':>11s} "
          f"{'p90':>11s} {'peak mem':>11s}  speedup_vs_vanilla")
    for R in replica_counts:
        baseline = None
        for label, fn in benches:
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            times, peak = fn(R, shapes, device)
            times.sort()
            med = median(times)
            if "vanilla" in label:
                baseline = med
                speedup = "  1.00x"
            else:
                speedup = f"  {baseline / med:5.2f}x" if baseline else "        "
            print(f"{label} {R:5d} {fmt_time(med):>11s} "
                  f"{fmt_time(times[int(0.1 * len(times))]):>11s} "
                  f"{fmt_time(times[int(0.9 * len(times))]):>11s} "
                  f"{fmt_mem(peak):>11s}  {speedup}")
        print()


if __name__ == "__main__":
    run("cpu", [1, 8, 32])
    if torch.cuda.is_available():
        run("cuda", [1, 8, 32, 100])
