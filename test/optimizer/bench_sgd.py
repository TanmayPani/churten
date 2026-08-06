"""Wall-clock benchmark for the consolidated batched ``sgd_step_``.

Compares, for an ensemble of R replicas:

  1. vanilla:      R separate ``torch.optim.SGD`` instances, called in a Python loop
                   (the path most users would write).
  2. consolidated: a single ``sgd_step_`` over one ``(R, T)`` buffer — the training
                   path, one kernel launch for the whole ensemble.
  3. the same at 75% frozen, which is what a late-training ensemble under
     EarlyStopping looks like: those rows exit before touching memory (the mask test
     is block-uniform on CUDA, a ``continue`` on CPU), so they move zero bytes.

Not collected by pytest (the ``bench_*`` convention): it reports numbers rather than
asserting them. Run as ``uv run python test/optimizer/bench_sgd.py``.
"""

import gc
import time
from functools import partial
from statistics import median

import torch

from torchstrap.optimizer.sgd import sgd_step_
from torchstrap.state import _CONSOLIDATION_ALIGNMENT

# Same shapes as bench_inplace_adam.py, so the two are directly comparable.
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
    # Let the GPU settle before warming up. Without this, a row late in a long
    # sweep is measured on a downclocked card: R=100 with momentum reported 10.4 ms
    # (bimodal, p10 3.1 ms) at the tail of the full sweep versus a steady 2.66 ms
    # run on its own. The numbers below are throughput claims, so they have to be
    # taken on a card in the same state each time.
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


def bench_vanilla(R, shapes, device, momentum=0.9):
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
            torch.optim.SGD(ps, lr=1e-2, momentum=momentum, weight_decay=1e-2)
        )

    def one_step():
        for opt in optims:
            opt.step()

    return _bench(one_step, device)


def bench_consolidated(R, shapes, device, momentum=0.9, frozen_frac=0.0):
    T = _consolidated_T(shapes)
    g = torch.Generator(device=device).manual_seed(0)
    p = torch.randn(R, T, device=device, generator=g)
    grad = torch.randn(R, T, device=device, generator=g)
    mb = torch.zeros_like(p) if momentum else None
    steps = torch.zeros(R, device=device)

    full = lambda x: torch.full((R,), x, device=device)
    lr, mom, damp, wd = full(1e-2), full(momentum), full(0.0), full(1e-2)
    mask = torch.ones(R, dtype=torch.bool, device=device)
    if frozen_frac:
        mask[: int(R * frozen_frac)] = False

    def one_step():
        sgd_step_(p, grad, mb, steps, lr, mom, damp, wd, mask,
                  nesterov=False, maximize=False)

    return _bench(one_step, device)


def fmt_time(t):
    if t < 1e-3:
        return f"{t * 1e6:8.1f} us"
    if t < 1.0:
        return f"{t * 1e3:8.2f} ms"
    return f"{t:8.3f} s"


def fmt_mem(b):
    return "       -- " if b is None else f"{b / 1024 / 1024:8.1f} MB"


def run(device_name, replica_counts, momentum, shapes=SHAPES):
    device = torch.device(device_name)
    benches = [
        ("vanilla (per-replica torch.optim.SGD)   ", bench_vanilla),
        ("consolidated (R,T) sgd_step_            ", bench_consolidated),
        ("consolidated sgd_step_, 75% frozen      ",
         partial(bench_consolidated, frozen_frac=0.75)),
    ]
    print()
    print(f"=== device={device_name}  momentum={momentum}  T={_consolidated_T(shapes)} ===")
    print(f"{'approach':<41s} {'R':>5s} {'median':>11s} {'p10':>11s} "
          f"{'p90':>11s} {'peak mem':>11s}  speedup_vs_vanilla")
    for R in replica_counts:
        baseline = None
        for label, fn in benches:
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            times, peak = fn(R, shapes, device, momentum=momentum)
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
    for momentum in (0.0, 0.9):
        run("cpu", [1, 8, 32], momentum)
        if torch.cuda.is_available():
            run("cuda", [1, 8, 32, 100], momentum)
