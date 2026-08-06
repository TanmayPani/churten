import argparse
from functools import partial
from itertools import chain
from math import pi
from pathlib import Path

from matplotlib import cm as cm
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

import torch
from torch.func import vmap, functional_call, grad_and_value
from torch.nn import Sequential, Linear, ReLU
from torch.nn.functional import binary_cross_entropy_with_logits, sigmoid

from torch.distributions import Categorical

from torchstrap.stateless import StatelessModule
from torchstrap.optimizer import Adam
from torchstrap.callbacks import (
    EpochScore, EpochTimer, PrintLog, LRScheduler, Checkpoint, EarlyStopping,
)

def make_spirals(n_samples, noise_std=0., rotations=1.):
    ts = torch.linspace(0, 1, n_samples)
    rs = ts ** 0.5
    thetas = rs * rotations * 2 * pi
    signs = torch.randint(0, 2, (n_samples,)) * 2 - 1

    labels = (signs > 0).to(torch.int8)

    xs = rs * signs * torch.cos(thetas) + torch.randn(n_samples) * noise_std
    ys = rs * signs * torch.sin(thetas) + torch.randn(n_samples) * noise_std
    points = torch.stack([xs, ys], dim=1)
    return points, labels

def make_classifier_module(*layer_sizes, device = "cpu", dtype = torch.float32):
    layer_sizes, output_size = layer_sizes[:-1], layer_sizes[-1]
    module = Sequential()
    for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:]):
        module.extend(
            Sequential(
                Linear(in_size, out_size), 
                ReLU(),
            )
        )
    module.append(Linear(layer_sizes[-1], output_size))
    return module.to(device=device, dtype=dtype)

def make_batched_indices(seed, *, dataset_size, sample_size=()):
    dist = Categorical(torch.ones(dataset_size))
    return dist.sample(sample_size)

def parallel_batch_iterator(
    X, y, *, 
    num_replicas = 1,
    batch_size = 32,
    num_batches = 100,
):
    total_samples = batch_size*num_batches

    indices = vmap(
        make_batched_indices, 
        randomness="different"
    )(
        torch.arange(num_replicas), 
        dataset_size=X.shape[0], 
        sample_size=(total_samples,),
    )

    for i in range(0, total_samples, batch_size):
        batch_indices = indices[:, i:i+batch_size]
        yield (
            X[batch_indices.ravel()].reshape(*(batch_indices.shape), *(X.shape[1:])), 
            y[batch_indices.ravel()].reshape(*(batch_indices.shape), 1), 
            None,
        )

def predict_fn(model, params, buffers, inputs):
    return sigmoid(functional_call(model, (params, buffers), inputs))

def predict_on_mesh(ensemble, state, width=1.5, steps=50):
    with torch.inference_mode():
        n = state.batch_size[0]
        xs = torch.linspace(-width, width, steps=steps, device=state.device)
        ys = torch.linspace(-width, width, steps=steps, device=state.device)
        xx, yy = torch.meshgrid(xs, ys, indexing="xy")

        points = torch.stack([xx.ravel(), yy.ravel()], dim=1).expand(n, xx.numel(), 2)
        fpred = vmap(partial(predict_fn, ensemble._base_model))
        z = fpred(state.params_dict, state.buffers_dict, points)

        z_mean = z.mean(dim=0).reshape_as(xx)

        return xx.detach().cpu(), yy.detach().cpu(), z_mean.detach().cpu()
    
def plot_predictions(ax, xx, yy, z):
    return ax.imshow(
        z, 
        extent=(
            xx.min(), xx.max(), 
            yy.min(), yy.max(),
        ), 
        origin="lower", 
        cmap=LinearSegmentedColormap.from_list(
            "blueorange", 
            ["xkcd:darkblue", "white", "xkcd:orangered"],
            #["tab:blue", "white", "tab:orange"],
        ), 
        vmin=0, vmax=1, 
        aspect="equal",
    )


def plot_spirals(ax, points, labels):
    return ax.scatter(
        points[:, 0], 
        points[:, 1], 
        c=labels,
        cmap = ListedColormap([
            #"xkcd:darkblue", "xkcd:orangered", 
            "tab:blue", "tab:orange",
        ]),
        edgecolors = "white",
    )

def parse_args():
    # Outputs go next to this script, never into whatever directory you happen to
    # have run from — `out/` is gitignored, so the repo stays clean.
    default_out = Path(__file__).resolve().parent / "out"
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--outdir", type=Path, default=default_out,
        help=f"where to write plots and checkpoints (default: {default_out})",
    )
    ap.add_argument(
        "--show", action=argparse.BooleanOptionalAction, default=True,
        help="open the figures in a window when done (default: --show)",
    )
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_replicas = 100
    num_samples = 100
    batch_size = 32
    num_epochs = 20
    batches_per_epoch = 20

    torch.manual_seed(0)

    points, labels = make_spirals(num_samples, noise_std=0.05)

    print("Initialized dataset ...")

    ensemble, state = StatelessModule.init(
        make_classifier_module,
        Adam,
        2, 512, 512, 1,
        num_replicas=num_replicas,
        device=device,
        init_randomness="different",
    )

    print("Initialized ensemble for bootstrapping ...")

    # Manual vmap training loop: differentiate the per-replica loss w.r.t. the
    # per-name param views, then run the fused Adam over the whole ensemble.
    def loss_fn(params, buffers, x, y):
        return binary_cross_entropy_with_logits(ensemble(params, buffers, x), y)

    grad_loss = vmap(grad_and_value(loss_fn, argnums=0), in_dims=(0, 0, 0, 0))

    # Callbacks are just callables dropped into the loop: schedule the per-replica
    # lr, snapshot the best rows, freeze plateaued replicas, time & print epochs.
    score = EpochScore()
    timer = EpochTimer()
    sched = LRScheduler("CosineAnnealingLR", T_max=num_epochs)
    ckpt = Checkpoint(root_dir=args.outdir / "spirals_ckpt", verbose=False)
    early = EarlyStopping(patience=8, threshold=1e-3, verbose=False)
    log = PrintLog()

    epoch_losses = []  # per-epoch (N,) score, on host
    for epoch in range(num_epochs):
        timer.tic()
        data_iterator = parallel_batch_iterator(
            points, labels,
            num_replicas=num_replicas, batch_size=batch_size,
            num_batches=batches_per_epoch,
        )
        batch_losses = []  # per-batch (N,) loss, on-device
        for X, Y, _ in data_iterator:
            X, Y = X.to(device), Y.to(device)
            grads, loss = grad_loss(state.params_dict, state.buffers_dict, X, Y)
            Adam.apply_gradient(state, grads)
            batch_losses.append(loss.detach())

        s = score(batch_losses)                 # (N,) epoch loss
        sched(state, s)                          # advance lr in place
        improved = ckpt(state, s)                # snapshot best rows
        epoch_losses.append(s.detach().cpu())
        log(epoch=epoch, train_loss=s, train_loss_best=bool(improved.any()),
            lr=state.optimizer_state["lr"], dur=timer.toc())
        if early(state, s):                      # all replicas frozen?
            break

    early.restore_best(state)                    # roll frozen replicas back to best
    ckpt.load_best(state)
    print("Training ensemble with bootstrapping done ...")

    losses = torch.stack(epoch_losses)  # (num_epochs, N)
    dy, y = torch.std_mean(losses, dim=1)
    x = torch.arange(losses.shape[0])

    fig0, ax0 = plt.subplots()
    ax0.set_title("Cross entropy loss vs epoch", weight="bold")
    ax0.set_xlabel("epoch", weight="bold")
    ax0.set_ylabel("loss", weight="bold")
    ax0.plot(x, y, "-", label="loss")
    ax0.fill_between(x, y-dy, y+dy, alpha=0.2, label=r"$\Delta$(loss)")
    ax0.legend()
    fig0.savefig(args.outdir / "loss.png")


    xx, yy, z = predict_on_mesh(ensemble, state)

    fig = plt.figure()
    #fig.suptitle("Predictions from bootstraped ensemble", weight="bold")
    ax = fig.add_subplot()
    #fig.subplots_adjust(top=0.8)
    ax.set_title(f"Predictions from bootstrap ({num_replicas} resamples)", weight="bold", y=0.95, pad = 30)

    ax.set_ylim(-1.5, 1.5)
    ax.set_xlim(-1.5, 1.5)
    
    ax.set_xlabel("x", weight="bold")
    ax.set_ylabel("y", weight="bold")

    sc = plot_spirals(ax, points, labels)
    im = plot_predictions(ax, xx, yy, z)
    ax.legend(
        sc.legend_elements()[0], 
        ["0", "1"], 
        title = "label",
        title_fontproperties = FontProperties(weight="bold"), 
        loc="lower right",
        #frameon=False,
        alignment = "center",
    )
    fig.colorbar(im, ax=ax, label="mean prediction")
    fig.savefig(args.outdir / "predictions.png")
    
    print(f"wrote loss.png, predictions.png and spirals_ckpt/ to {args.outdir}")

    if args.show:
        try:
            plt.show()
        except KeyboardInterrupt:
            exit()



