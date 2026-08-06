# torchstrap

**Train an ensemble of N model replicas in parallel on a single GPU — vectorized, not looped.**

`torchstrap` is a PyTorch training framework for *model ensembles*. Instead of looping
over N models, every replica is a slice along a leading "batch" dimension and the
entire ensemble is trained in one vectorized pass with [`torch.func.vmap`](https://pytorch.org/docs/stable/func.html).
Models run **statelessly** — parameters and buffers live outside the `nn.Module` and
are threaded in explicitly via `functional_call` — which makes per-replica
early-stopping, checkpointing, and bootstrap resampling first-class.

The headline component is a **fused, batched optimizer** written in C++ and CUDA C++.
Every parameter of all N replicas is packed into one consolidated `(N, T)` buffer, so it
updates the whole ensemble in a **single GPU launch total** and is **up to 4.5× faster**
than the loop most people would write, with **no extra memory**. Each kernel ports the math
of ATen's own `Fused*` counterpart **for the device it runs on**, term for term, so an active
replica is **bit-identical** to `torch.optim.<Opt>(fused=True)` — on CPU and on CUDA alike.
**`Adam`/`AdamW`, `SGD` and `Adagrad`** are implemented today — which is ATen's entire fused family.

---

## Benchmark: fused batched Adam vs. vanilla PyTorch

The baseline ("vanilla") is the obvious approach: `N` independent
`torch.optim.Adam` instances, each `.step()`-ed in a Python loop. `torchstrap` replaces
that with one `adam_step_` call over the stacked `(N, *param)` state, dispatched to a
fused CUDA kernel.

Single optimizer step, fp32, NVIDIA RTX 4070 Laptop GPU (torch 2.13 / CUDA 13.0),
median of 20 runs:

| Workload                                  |  N  | vanilla (loop) | **torchstrap (fused)** | speedup | peak mem (vanilla → torchstrap) |
| ----------------------------------------- | --: | -------------: | ------------------: | :-----: | :--------------------------: |
| 6 tensors · 265k params/replica           |   8 |        796 µs  |          **140 µs** | **5.7×**|        33.3 → 32.3 MB        |
| 6 tensors · 265k params/replica           |  32 |       3.22 ms  |         **1.18 ms** | **2.7×**|       130.3 → 129.3 MB       |
| 6 tensors · 265k params/replica           | 100 |       9.38 ms  |         **3.23 ms** | **2.9×**|       405.1 → 403.9 MB       |

At every size the baseline pays `N` Python-level `.step()` calls where `torchstrap` issues
one fused launch for the whole ensemble; the ratio peaks at small `N`, where that launch
overhead is the entire cost, and settles to the bandwidth ratio once the ensemble is large
enough to saturate memory. Peak
memory **matches vanilla** because the kernel fuses the whole Adam update (moment
updates, bias correction, `sqrt`, decoupled weight decay) and allocates **zero
`(N, *param)` temporaries**.

Reproduce: `uv run python test/optimizer/bench_inplace_adam.py`

---

## Highlights

- **Vectorized ensembles.** N replicas train in one `vmap`-ed forward/backward pass;
  no Python loop over models.
- **Fused batched Adam, SGD and Adagrad.** Custom `torch.library` ops
  (`torchstrap::adam_step_`, `torchstrap::sgd_step_`, `torchstrap::adagrad_step_`) with
  hand-written kernels on both devices, compiled ahead of time into one extension
  (`torchstrap.kernels._C`) and each claiming its dispatch key from C++, that update all replicas in
  one pass. A frozen replica is skipped
  outright rather than recomputed, so it moves **zero bytes** and its rows stay
  bit-identical, with no snapshot/restore. Each kernel uses the `adam_math` of ATen's fused
  Adam **for its own device** — the CUDA one includes ATen's header and calls it verbatim —
  so results are **bit-identical to `torch.optim.Adam(fused=True)` on that device** (`torch.equal`, across the amsgrad / maximize / decoupled-weight-decay
  matrix). `SGD` gets the same guarantee against `torch.optim.SGD(fused=True)` across the
  momentum / dampening / nesterov / maximize matrix, and adds two things ATen structurally
  cannot: a **per-replica `is_first_step`** (upstream's is one host bool, which would hand a
  replica frozen before its first step an uninitialised momentum buffer on thaw), and a
  per-replica `momentum` value, so a sweep that includes `momentum=0` is expressible.
  `Adagrad` completes ATen's fused family and gets the same treatment — though note its
  update runs in **fp64 on the device** (that is upstream's `adagrad_math`, and it is
  what the bit-exactness *is*), which makes it compute-bound rather than memory-bound
  and slower than the Python loop at small `R`; see `bench_adagrad.py` for numbers. Upstream deliberately gives CPU and CUDA different formulations — the CPU one
  computes bias corrections in `double` and uses a cancellation-safe lerp — so CPU and CUDA
  agree with each other only to `atol=rtol=1e-5`, and torchstrap keeps that split rather
  than picking one formulation for both.
- **Per-replica everything.** Each replica can have its own learning rate,
  early-stopping schedule, and checkpoint — useful for bootstrap ensembles and
  hyperparameter sweeps.
- **Stateless by design.** Weights live in an explicit `State` pytree, not inside the
  module, enabling clean functional transforms and `torch.compile`.
- **skorch-style callbacks.** `EarlyStopping`, `Checkpoint`, `LRScheduler`,
  scoring, and logging hooks — all replica-aware. The `LRScheduler` is fully
  vectorized over the ensemble: each replica carries its own freeze-aware schedule
  clock, and `ReduceLROnPlateau` runs one plateau detector **per replica** (one
  scalar metric can't — torch's can't do this).
- **Runtime type safety.** The whole package is `beartype`-checked at runtime.

---

## Install

Requires Python ≥ 3.13. Uses [`uv`](https://docs.astral.sh/uv/) (the lockfile pins
torch's CPU/CUDA wheel automatically via `UV_TORCH_BACKEND=auto`):

```bash
uv sync
```

That compiles `torchstrap.kernels._C` — **one** extension holding both kernels, built ahead of time
by `setup.py`. Rebuild it after editing anything under `kernels/csrc/`:

```bash
uv sync --reinstall-package torchstrap
```

| | source | needed to build | included when |
|---|---|---|---|
| CPU | `cpu/adam.cpp` | a host `c++` | always |
| CUDA | `cuda/adam.cu` | `nvcc` matching torch's CUDA major, + a host compiler it accepts (CUDA 13.x caps at GCC 15) | a matching toolkit is found |

The toolkit is located automatically: `setup.py` picks the installed `nvcc` whose **major**
matches `torch.version.cuda` rather than following the `/usr/local/cuda` symlink, and reads
the GCC ceiling out of that toolkit's own `crt/host_config.h` to choose a `-ccbin`. It
prints what it settled on as it builds. The build is `--no-build-isolation` on purpose: the
extension links against the torch already in the environment, and a torch downloaded into
an isolated build env would be a different build.

Without a usable toolkit the extension is CPU-only, and the op raises from the dispatcher on
a CUDA tensor — the same thing `torch._fused_adam_` does on a device ATen has no kernel for.

---

## Quickstart

Train a 100-member bootstrap ensemble of MLP classifiers in parallel:

```python
import torch
from torch.nn import Sequential, Linear, ReLU
from torch.nn.functional import binary_cross_entropy_with_logits

from torchstrap.stateless import StatelessModule
from torchstrap.optimizer import Adam
from torchstrap.callbacks import Checkpoint, EarlyStopping, LRScheduler

def make_mlp(*sizes):
    layers = []
    for a, b in zip(sizes[:-2], sizes[1:-1]):
        layers += [Linear(a, b), ReLU()]
    layers += [Linear(sizes[-2], sizes[-1])]
    return Sequential(*layers)

# Deep-copies the model into 100 independently-initialized replicas, stacks their
# params/buffers, and builds the optimizer + State. `optimizer` is the Adam *class*.
ensemble, optimizer, state = StatelessModule.init(
    make_mlp,
    Adam,
    model_init_args=(2, 512, 512, 1),
    num_replicas=100,
    device="cuda",
    init_randomness="different",          # each replica gets distinct initial weights
)

# `data_iterator` yields (input, target, sample_weight) with a leading replica dim,
# e.g. bootstrap-resampled minibatches of shape (num_replicas, batch, *features).
history = ensemble.fit(
    optimizer,
    binary_cross_entropy_with_logits,
    state,
    data_iterator,
    callbacks=[
        ("checkpoint",     Checkpoint(monitor="train_loss_best")),  # per-replica save
        ("early_stopping", EarlyStopping(monitor="train_loss")),    # per-replica freeze
        # Vectorized schedule; lr is written in place into the (N,) lr vector.
        ("lr",             LRScheduler(policy="CosineAnnealingLR", T_max=100)),
    ],
)
```

Per-replica predictions on a grid (the ensemble mean is a calibrated probability):

```python
from functools import partial
from torch.func import vmap, functional_call
from torch.nn.functional import sigmoid

def predict(model, params, buffers, x):
    return sigmoid(functional_call(model, (params, buffers), x))

with torch.inference_mode():
    # points: (num_replicas, num_points, 2)
    probs = vmap(partial(predict, ensemble._base_model))(
        state.param_dict, state.buffer_dict, points
    )
    ensemble_mean = probs.mean(dim=0)     # average over the 100 replicas
```

A full runnable version (with plots of the loss curve and decision boundary) lives
in [`examples/spirals/spirals_parallel.py`](examples/spirals/spirals_parallel.py):

```bash
uv run python examples/spirals/spirals_parallel.py
```

It trains 100 bootstrap replicas on a noisy two-spirals dataset and renders the
ensemble's averaged decision boundary — a cheap, well-calibrated uncertainty estimate.

For a heavier workout of the consolidated optimizer state, a parallel **CNN**
ensemble lives in
[`examples/cnn/cnn_parallel.py`](examples/cnn/cnn_parallel.py):

```bash
uv run python examples/cnn/cnn_parallel.py
```

A CNN has many parameter tensors of very different shapes (conv kernels, biases,
large FC weights), so it stresses the way `State` packs every parameter of all `N`
replicas into a **single contiguous `(N, T)` buffer** and runs the fused Adam in one
launch per step regardless of the layer count. The script prints the consolidation
footprint (parameter-tensor count → one `(N, T)` buffer and its size), trains 64
bootstrap CNN replicas on a synthetic image task, and reports throughput (steps/s)
plus held-out ensemble accuracy.

---

## How it fits together

Four small abstractions interlock:

| Component          | Role                                                                                                          |
| ------------------ | ------------------------------------------------------------------------------------------------------------- |
| `StatelessModule`  | The trainer. Holds the base model on the `meta` device (no weights) and runs the `vmap`-ed forward/backward.  |
| `State`            | All training state as a pytree: stacked `param_dict` / `buffer_dict` + optimizer state, with a leading `N` dim. Unbinds into per-replica views for masking/checkpointing. |
| `GradientTransformation` | Optimizers are **classes**, not instances (a metaclass). `Adam` defines an `AdamState` and an `update` classmethod; `apply_gradient` drives the fused kernel. |
| `Callback`         | skorch-style hooks (`on_epoch_end`, `on_grad_computed`, …), all replica-aware.                                |

The training step computes per-replica gradients with
`vmap(grad_and_value(forward))`, stores them on the `State`, and calls the fused
`adam_step_` over the whole stacked ensemble.

---

## Testing & benchmarking

```bash
uv run pytest                     # the whole suite, ~2 s
uv run pytest -k cuda             # only the CUDA half
```

`test/conftest.py` provides a parametrized `device` fixture, so each kernel test runs once
per device and the CUDA cases **skip** — rather than silently disappear — on a machine
without a GPU. The headline gates are `test_aten_fused_parity.py` / `test_aten_sgd_parity.py` (bit-exact vs
`torch._fused_adam_` / `torch._fused_sgd_` / `torch._fused_adagrad_` on each device) and
the matching `test_cpu_cuda_*_parity.py` files (CPU vs CUDA, plus `torch.library.opcheck`).

`bench_*.py` files are not collected — they report numbers rather than asserting them, and
are run directly:

```bash
uv run python test/optimizer/bench_inplace_adam.py
uv run python test/optimizer/bench_sgd.py
uv run python test/optimizer/bench_adagrad.py
```

### Why the kernel is hand-written, and why there is nothing to autotune

The update is purely **memory-bound**: it moves ~7 × N × T × 4 bytes per step and does a
handful of FMAs on each. Measured at N=100, T=265k on a 4070 Laptop, block sizes of 128,
256, 512 and 1024 all land within 1% of each other, ~192 GB/s against a ~256 GB/s part.
There is simply no launch-shape decision worth searching for, so the kernel just uses
ATen's own (512 threads, 65536-element chunks).

That is what makes a search actively harmful. Autotuning this single consolidated `(N, T)`
launch used to converge on a pathological tiny-block config several× off the roofline,
because the bandwidth-saturating candidates time out at Helion's 60 s compile limit and
get eliminated — so the Helion kernel it replaced had to be pinned by hand anyway. The
CUDA kernel matches its throughput with a fixed launch config and no search at all:

| N=100, T=265k | Helion | CUDA kernel | |
|---|---|---|---|
| all replicas active | 3.29 ms | 3.23 ms | parity |
| 75% frozen | 3.22 ms | **800 µs** | **4.0×** |

Those CUDA figures depend on `State` rounding the consolidated width `T` up to a multiple
of `kILP`. ATen's vectorized guard is `n % kILP == 0` with `n = T - chunk_offset`, and
every chunk offset is a multiple of `kChunkSize`, so an unaligned `T` puts the *entire*
ensemble on the ragged element-by-element path — not just the last chunk. ATen cannot fix
that (it does not own the allocation); torchstrap can, because it builds the consolidation.
Unpadded, the same two rows measure 3.44 ms and 870 µs.

Where it wins outright is that second row — a **partly frozen ensemble**, the normal state
late in training under `EarlyStopping` — because frozen blocks retire before touching
memory rather than reading and writing back unchanged values.

The CPU kernel is one fused pass with no temporaries, against ~15 passes and ~8 `(R, T)`
allocations for the PyTorch path:

| CPU workload | PyTorch path | C++ kernel | |
|---|---|---|---|
| R=32, T=50k | 8.39 ms | 0.70 ms | **12.0×** |
| R=32, T=50k, 75% frozen | 8.70 ms | 0.28 ms | **31.1×** |
| R=100, T=265k | 295.7 ms | 22.3 ms | **13.3×** |
| R=100, T=265k, 75% frozen | 296.0 ms | 4.91 ms | **60.3×** |

Both kernels register themselves the way ATen registers `_fused_adam_` — the schema comes
from a `TORCH_LIBRARY` in `kernels/csrc/stubs.cpp`, the kernels from a `TORCH_LIBRARY_IMPL` each, and
`torch.ops.torchstrap.adam_step_(...)` dispatches by device with no Python in the path.

---

## Roadmap

The destination: **a batched, vmap-native drop-in for `torch.optim` and friends** —
anything you'd train one model with, you can train an ensemble of, fused.

- **The full optimizer family.** `Adam`/`AdamW`, `SGD` and `Adagrad` ship today —
  ATen's *entire* fused family, and therefore everything that can be held to the
  bit-exactness bar. `RMSprop`, `NAdam`, `RAdam`, `Adadelta`, `Adamax`, `Rprop` and
  `ASGD` have no fused reference on either device, so they get a batched op cut from
  the same cloth but a tolerance-based gate against `torch.optim`. `LBFGS` deliberately gets no kernel: its work is reductions rather than
  elementwise math (a `bmm` cuBLAS already runs at roofline), and its line search has
  data-dependent trip counts per replica.
- **Fused kernels everywhere.** Roll the `csrc/cuda/*.cu` treatment out to every
  optimizer — each one a port of its ATen `fused_*` counterpart, reshaped for the
  consolidated `(N, T)` layout — then close the remaining gap: `bf16`/`fp16` optimizer
  *states* (ATen's `FusedAdamMathFunctorMP` keeps fp32 params with bf16 moments, which
  cuts per-element traffic ~29% on a memory-bound kernel and halves the moment footprint).
- **Deeper vmap composition.** Support `R > 1` replicas *inside* an outer `vmap` —
  ensembles of ensembles, for nested resampling and hierarchical sweeps.
- **Beyond bootstrap.** Lean into what per-replica state unlocks: built-in
  cross-validation folds, hyperparameter sweeps as replicas, and SWA-style weight
  averaging across the ensemble.
