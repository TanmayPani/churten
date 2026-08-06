"""Parallel CNN ensemble — a consolidation stress test.

Where `examples/spirals/` trains a tiny MLP (a handful of same-ish parameter
tensors), this trains an **ensemble of small CNNs** in parallel. A CNN has many
parameter tensors of very different shapes — conv kernels `(out, in, k, k)`, conv
biases `(out,)`, large FC weights `(hidden, C*H*W)`, FC biases — so it exercises
the consolidated `(N, T)` optimizer state much harder:

  * `State` packs all those shapes into ONE contiguous `(N, T)` buffer; this script
    prints `T`, the number of distinct parameter tensors, and the buffer size.
  * every training step reconstructs the per-name `(N, *p)` views, differentiates
    w.r.t. the flat buffer, and runs the fused Adam in a SINGLE kernel launch over
    the whole ensemble — regardless of how many conv/FC tensors the model has.

The task is synthetic but learnable (per-class image templates + noise), so the
ensemble's train loss/accuracy actually improve — i.e. the consolidated path is
exercised end to end (vmapped conv forward + grad + fused step + callbacks), not
just constructed.

Run: `uv run python examples/cnn/cnn_parallel.py`
"""

import argparse
from pathlib import Path
from time import perf_counter

import torch
from torch.func import vmap, functional_call, grad_and_value
from torch.nn import Sequential, Conv2d, ReLU, MaxPool2d, Flatten, Linear
from torch.nn.functional import cross_entropy

from torchstrap.stateless import StatelessModule
from torchstrap.optimizer import Adam


# --------------------------------------------------------------------------- #
# Synthetic image classification: per-class templates + noise.
# --------------------------------------------------------------------------- #
def make_image_dataset(
    n_samples,
    in_channels,
    height,
    width,
    num_classes,
    *,
    noise_std=1.6,
    seed=0,
    device="cpu",
    dtype=torch.float32,
):
    g = torch.Generator().manual_seed(seed)
    templates = torch.randn(num_classes, in_channels, height, width, generator=g)
    labels = torch.randint(0, num_classes, (n_samples,), generator=g)
    images = templates[labels] + noise_std * torch.randn(
        n_samples, in_channels, height, width, generator=g
    )
    return (
        images.to(device=device, dtype=dtype),
        labels.to(device=device),
    )


class ParallelImageLoader:
    """Re-iterable (one fresh pass per epoch) loader yielding per-replica
    bootstrap minibatches shaped ``(N, B, C, H, W)`` / ``(N, B)``.

    Each replica draws its own bootstrap resample of the dataset (distinct rows),
    which is exactly the ensemble-of-replicas setting torchstrap targets. Indices
    are fixed at construction so every epoch sees the same per-replica stream.
    """

    def __init__(self, X, y, *, num_replicas, batch_size, num_batches, seed=0):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.num_batches = num_batches
        total = batch_size * num_batches
        g = torch.Generator().manual_seed(seed)
        # (N, total) independent bootstrap indices into the dataset.
        self._indices = torch.randint(0, X.shape[0], (num_replicas, total), generator=g)

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        bs, n = self.batch_size, self._indices.shape[0]
        feat = self.X.shape[1:]
        for i in range(0, bs * self.num_batches, bs):
            bi = self._indices[:, i : i + bs].to(self.X.device)  # (N, B)
            flat = bi.reshape(-1)
            Xb = self.X[flat].reshape(n, bs, *feat)  # (N, B, C, H, W)
            yb = self.y[flat].reshape(n, bs)  # (N, B) long
            yield Xb, yb, None


# --------------------------------------------------------------------------- #
# A small CNN: several conv blocks + FC head → many varied-shape param tensors.
# --------------------------------------------------------------------------- #
def make_cnn(
    in_channels,
    num_classes,
    *,
    height,
    width,
    channels=(16, 32),
    hidden=64,
    device="cpu",
    dtype=torch.float32,
):
    layers = []
    c, h, w = in_channels, height, width
    for oc in channels:
        layers += [Conv2d(c, oc, kernel_size=3, padding=1), ReLU(), MaxPool2d(2)]
        c, h, w = oc, h // 2, w // 2
    layers += [Flatten(), Linear(c * h * w, hidden), ReLU(), Linear(hidden, num_classes)]
    return Sequential(*layers).to(device=device, dtype=dtype)


def predict_fn(model, params, buffers, x):
    return functional_call(model, (params, buffers), (x,))


@torch.no_grad()
def ensemble_accuracy(ensemble, state, X, y):
    """Mean per-replica accuracy over the whole dataset (X broadcast to all N)."""
    n = state.batch_size[0]
    Xb = X.unsqueeze(0).expand(n, *X.shape)  # (N, M, C, H, W)
    logits = vmap(predict_fn, in_dims=(None, 0, 0, 0))(
        ensemble._base_model, state.params_dict, state.buffers_dict, Xb
    )  # (N, M, num_classes)
    preds = logits.argmax(dim=-1)  # (N, M)
    return (preds == y.unsqueeze(0)).float().mean(dim=1)  # (N,)


def main(outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Stress knobs.
    num_replicas = 64
    in_channels, height, width = 1, 16, 16
    num_classes = 3
    channels, hidden = (16, 32), 64
    n_samples = 2000
    n_test = 512
    batch_size = 64
    num_batches = 50
    num_epochs = 4

    torch.manual_seed(0)
    print(f"device = {device}")

    # One dataset (shared per-class templates); split into a train pool the replicas
    # bootstrap-resample and a held-out test set none of them ever see — so accuracy
    # below reflects generalization (with real ensemble spread), not memorization.
    X_all, y_all = make_image_dataset(
        n_samples + n_test, in_channels, height, width, num_classes, device=device
    )
    X, y = X_all[:n_samples], y_all[:n_samples]
    Xe, ye = X_all[n_samples:], y_all[n_samples:]
    loader = ParallelImageLoader(
        X, y, num_replicas=num_replicas, batch_size=batch_size, num_batches=num_batches
    )
    print(
        f"dataset: {n_samples} images {tuple(X.shape[1:])}, {num_classes} classes; "
        f"{num_replicas} replicas × {num_batches} batches/epoch × {batch_size}"
    )

    ensemble, state = StatelessModule.init(
        make_cnn,
        Adam,
        in_channels,
        num_classes,
        height=height,
        width=width,
        channels=channels,
        hidden=hidden,
        num_replicas=num_replicas,
        device=device,
        init_randomness="different",
        optimizer_kwargs=dict(lr=3e-3),
    )

    # --- consolidation footprint: what the (N, T) buffer actually packs ---
    param_names = list(state.params_dict.keys())
    n_tensors = len(param_names)
    n_rep, T = tuple(state.flat_params.shape)
    bytes_per = state.flat_params.element_size()
    print(
        f"consolidated state: {n_tensors} parameter tensors -> ONE ({n_rep}, {T}) "
        f"buffer ({n_rep * T * bytes_per / 1e6:.2f} MB/buffer; params+grad+2 moments)"
    )
    print(f"  per-replica params T = {T:,}; param tensors: {param_names}")

    # Held-out test accuracy (one vmapped forward broadcasts the test set to all N).
    acc0 = ensemble_accuracy(ensemble, state, Xe, ye).mean().item()

    # Manual vmap training loop. cross_entropy needs integer class indices, so the
    # per-replica targets `yb` stay long (no compute-dtype cast — the manual-loop
    # analogue of the old `target_dtype=None`).
    def loss_fn(params, buffers, x, y):
        return cross_entropy(ensemble(params, buffers, x), y)

    grad_loss = vmap(grad_and_value(loss_fn, argnums=0), in_dims=(0, 0, 0, 0))

    batch_curve = []  # per-batch mean-over-replicas loss
    t0 = perf_counter()
    for _ in range(num_epochs):
        for Xb, yb, _ in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            grads, loss = grad_loss(state.params_dict, state.buffers_dict, Xb, yb)
            Adam.apply_gradient(state, grads)
            batch_curve.append(loss.detach().mean())
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = perf_counter() - t0

    acc1 = ensemble_accuracy(ensemble, state, Xe, ye)
    steps = num_epochs * num_batches
    print(
        f"\ntrained {num_epochs} epochs ({steps} fused steps) in {elapsed:.2f}s "
        f"= {steps / elapsed:.1f} steps/s  ({num_replicas} replicas/step)"
    )
    print(
        f"held-out test accuracy: {acc0:.3f} (init) -> "
        f"{acc1.mean().item():.3f} ± {acc1.std().item():.3f} (final, across replicas)"
    )

    # Optional loss curve.
    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt

        curve = torch.stack(batch_curve).cpu()
        fig, ax = plt.subplots()
        ax.set_title("Parallel CNN ensemble — mean cross-entropy", weight="bold")
        ax.set_xlabel("minibatch iteration")
        ax.set_ylabel("loss")
        ax.plot(curve.numpy())
        out = outdir / "cnn_loss.png"
        fig.savefig(out)
        print(f"saved {out}")
    except Exception as e:  # plotting is optional; never fail the stress test on it
        print(f"(skipped loss plot: {e})")


if __name__ == "__main__":
    # Outputs go next to this script, never into whatever directory you happen to
    # have run from — `out/` is gitignored, so the repo stays clean.
    default_out = Path(__file__).resolve().parent / "out"
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--outdir", type=Path, default=default_out,
        help=f"where to write the loss plot (default: {default_out})",
    )
    main(ap.parse_args().outdir)
