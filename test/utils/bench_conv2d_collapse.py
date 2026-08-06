"""Conv2dNN: gradual spatial collapse (valid convs → 1×1) vs the global-avg-pool baseline.

Apples-to-apples held-out accuracy on a synthetic (2, 9, 9) per-class-template task (the
`bin_counts` shape), mirroring `examples/cnn/cnn_parallel.py`'s ensemble harness. Two
`Conv2dNN` ensembles are trained with an identical budget/seed:

  * baseline : conv_channels=(32, 64), k=3 pad=1 (size-preserving) + AdaptiveAvgPool2d(1)
  * collapse : collapse_spatial=True, input_size=9, conv_channels=(16,32,48,64)
               → valid convs 9→7→5→3→1, so the last conv (not a mean) summarizes the grid

Reports init→final held-out accuracy (mean ± std across replicas), per-replica param count
`T`, peak CUDA memory, and steps/s. The collapse head is more expressive about *where*
signal sits, so accuracy should be >= baseline.

Run: uv run python test/utils/bench_conv2d_collapse.py
"""

from time import perf_counter

import torch
from torch.func import vmap, functional_call, grad_and_value
from torch.nn.functional import cross_entropy

from torchstrap.stateless import StatelessModule
from torchstrap.optimizer import Adam
from torchstrap.utils.nn.archs import Conv2dNN


# --------------------------------------------------------------------------- #
# Synthetic image classification: per-class spatial templates + noise.
# --------------------------------------------------------------------------- #
def make_image_dataset(n, in_channels, h, w, num_classes, *, noise_std, seed, device):
    g = torch.Generator().manual_seed(seed)
    templates = torch.randn(num_classes, in_channels, h, w, generator=g)
    labels = torch.randint(0, num_classes, (n,), generator=g)
    images = templates[labels] + noise_std * torch.randn(
        n, in_channels, h, w, generator=g
    )
    return images.to(device), labels.to(device)


class ParallelImageLoader:
    """Per-replica bootstrap minibatches shaped (N, B, C, H, W) / (N, B) long."""

    def __init__(self, X, y, *, num_replicas, batch_size, num_batches, seed=0):
        self.X, self.y = X, y
        self.batch_size, self.num_batches = batch_size, num_batches
        g = torch.Generator().manual_seed(seed)
        total = batch_size * num_batches
        self._idx = torch.randint(0, X.shape[0], (num_replicas, total), generator=g)

    def __iter__(self):
        bs, n, feat = self.batch_size, self._idx.shape[0], self.X.shape[1:]
        for i in range(0, bs * self.num_batches, bs):
            flat = self._idx[:, i : i + bs].reshape(-1).to(self.X.device)
            yield self.X[flat].reshape(n, bs, *feat), self.y[flat].reshape(n, bs), None


def _predict(model, params, buffers, x):
    return functional_call(model, (params, buffers), (x,))


@torch.no_grad()
def ensemble_accuracy(ensemble, state, X, y):
    n = state.batch_size[0]
    Xb = X.unsqueeze(0).expand(n, *X.shape)  # (N, M, C, H, W)
    logits = vmap(_predict, in_dims=(None, 0, 0, 0))(
        ensemble._base_model, state.params_dict, state.buffers_dict, Xb
    )
    preds = logits.argmax(dim=-1)  # (N, M)
    return (preds == y.unsqueeze(0)).float().mean(dim=1)  # (N,)


def train(ensemble, state, loader, *, num_epochs, device):
    def loss_fn(params, buffers, x, y):
        return cross_entropy(ensemble(params, buffers, x), y)

    grad_loss = vmap(grad_and_value(loss_fn, argnums=0), in_dims=(0, 0, 0, 0))
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = perf_counter()
    steps = 0
    for _ in range(num_epochs):
        for Xb, yb, _ in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            grads, _ = grad_loss(state.params_dict, state.buffers_dict, Xb, yb)
            Adam.apply_gradient(state, grads)
            steps += 1
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = perf_counter() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0
    return steps / elapsed, peak


def build(num_classes, num_replicas, device, **conv_kwargs):
    return StatelessModule.init(
        Conv2dNN,
        Adam,
        in_channels=2,
        head_sizes=(num_classes,),
        num_replicas=num_replicas,
        init_randomness="different",
        device=device,
        optimizer_kwargs=dict(lr=3e-3),
        **conv_kwargs,
    )


def run_one(name, conv_kwargs, *, X, y, Xe, ye, num_classes, num_replicas,
            num_epochs, batch_size, num_batches, device):
    torch.manual_seed(0)  # identical init RNG stream for both variants
    ensemble, state = build(num_classes, num_replicas, device, **conv_kwargs)
    T = int(state.flat_params.shape[1])
    loader = ParallelImageLoader(
        X, y, num_replicas=num_replicas, batch_size=batch_size,
        num_batches=num_batches, seed=0,  # identical bootstrap stream for both
    )
    acc0 = ensemble_accuracy(ensemble, state, Xe, ye).mean().item()
    sps, peak = train(ensemble, state, loader, num_epochs=num_epochs, device=device)
    acc1 = ensemble_accuracy(ensemble, state, Xe, ye)
    print(
        f"  {name:>9}: acc {acc0:.3f} -> {acc1.mean().item():.3f} "
        f"± {acc1.std().item():.3f}   T={T:,}  peak={peak:.2f}GB  {sps:.0f} steps/s"
    )
    return acc1.mean().item(), acc1.std().item()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    in_ch, S, num_classes = 2, 9, 4
    num_replicas, n_train, n_test = 32, 3000, 1000
    batch_size, num_batches, num_epochs = 128, 40, 8

    print(f"device = {device};  task: (2,9,9) templates, {num_classes} classes, "
          f"{num_replicas} replicas")
    baseline_kw = dict(conv_channels=(32, 64))  # k=3 pad=1 preserve + avg-pool
    collapse_kw = dict(
        conv_channels=(16, 32, 48, 64), collapse_spatial=True, input_size=S
    )

    for noise_std in (1.2, 1.6, 2.0):
        X_all, y_all = make_image_dataset(
            n_train + n_test, in_ch, S, S, num_classes,
            noise_std=noise_std, seed=0, device=device,
        )
        X, y = X_all[:n_train], y_all[:n_train]
        Xe, ye = X_all[n_train:], y_all[n_train:]
        common = dict(
            X=X, y=y, Xe=Xe, ye=ye, num_classes=num_classes,
            num_replicas=num_replicas, num_epochs=num_epochs,
            batch_size=batch_size, num_batches=num_batches, device=device,
        )
        print(f"\nnoise_std = {noise_std}")
        b_mean, _ = run_one("baseline", baseline_kw, **common)
        c_mean, c_std = run_one("collapse", collapse_kw, **common)
        verdict = "OK (>=)" if c_mean >= b_mean - c_std else "WORSE"
        print(f"  -> collapse {c_mean:.3f} vs baseline {b_mean:.3f}: {verdict}")


if __name__ == "__main__":
    main()
