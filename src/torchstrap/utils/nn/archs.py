from beartype.typing import Union, Protocol, Optional, Any
from beartype.typing import runtime_checkable

import torch
from torch import Tensor
from torch.nn import Module

from torchstrap.utils._utils import params_for


@runtime_checkable
class TensorCallable(Protocol):
    def __call__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Tensor | tuple[Tensor, ...]: ...


type ModuleLike = Union[Module, type[Module], TensorCallable, type[TensorCallable]]


@torch.no_grad
def init_to_zero(module):
    for param in module.parameters():
        param.zero_()


class Lambda(Module):
    def __init__(self, fn: TensorCallable):
        super().__init__()
        self.forward = fn


def initialize_layer(layer: ModuleLike, *args, **kwargs) -> Module:
    if isinstance(layer, Module):
        return layer

    if isinstance(layer, TensorCallable) and not isinstance(layer, type):
        return Lambda(layer)

    if isinstance(layer, type) and issubclass(layer, Module):
        return layer(*args, **kwargs)

    return Lambda(layer(*args, **kwargs))


class Layer(Module):
    def __init__(
        self,
        module: ModuleLike,
        input_size: int,
        output_size: int,
        dropout_prob: Optional[float] = None,
        batch_norm: Optional[type[Module]] = None,
        activation: Optional[ModuleLike] = None,
        **kwargs,
    ):
        super().__init__()

        layer_kwargs = params_for("layer", kwargs)

        self.layeritos = torch.nn.ModuleDict()

        if dropout_prob is not None:
            self.layeritos["dropout"] = torch.nn.Dropout(p=dropout_prob)

        if batch_norm is not None:
            layer_kwargs["bias"] = False

        self.layeritos["layer"] = initialize_layer(
            module, input_size, output_size, **layer_kwargs
        )

        if batch_norm is not None:
            batch_norm_kwargs = params_for("batch_norm", kwargs)
            self.layeritos["batch_norm"] = initialize_layer(
                batch_norm, output_size, **batch_norm_kwargs
            )

        if activation is not None:
            activation_kwargs = params_for("activation", kwargs)
            activation_args = activation_kwargs.pop("args", ())
            self.layeritos["activation"] = initialize_layer(
                activation, *activation_args, **activation_kwargs
            )

    def forward(self, x, training=False):
        # x = F.dropout(x, p = self.dropout_p, training=training)
        for _, module in self.layeritos.items():
            x = module(x)
        return x


class MLP(Module):
    def __init__(
        self,
        layer: ModuleLike | list[ModuleLike] = torch.nn.Linear,
        activation: Optional[ModuleLike | list[Optional[ModuleLike]]] = torch.nn.ReLU,
        layer_sizes=[10, 1],
        output_activation: Optional[ModuleLike] = None,
        input_transform: Optional[ModuleLike] = None,
        device="cpu",
        dtype=torch.float32,
        **kwargs,
    ):

        super().__init__()

        nlayers = len(layer_sizes) - 1
        assert nlayers > 0

        if isinstance(layer, (list)):
            assert len(layer) == nlayers
            layers = layer
        else:
            layers = [layer] * nlayers

        if isinstance(activation, (list)):
            assert len(activation) == nlayers
            activations = list(activation)
        else:
            activations = [activation] * nlayers
        activations[-1] = output_activation

        for key, value in kwargs.items():
            if isinstance(value, list):
                assert len(value) == nlayers
            else:
                kwargs[key] = [value] * nlayers

        list_kwargs = [
            {key: val[i] for key, val in kwargs.items()} for i in range(nlayers)
        ]

        self.layers = torch.nn.ModuleList()
        if input_transform is not None:
            input_transform_kwargs = params_for("input_transform", kwargs)
            input_transform_args = input_transform_kwargs.pop("args", ())
            self.layers.append(
                initialize_layer(
                    input_transform, *input_transform_args, **input_transform_kwargs
                )
            )

        for ilayer, (input_size, output_size) in enumerate(
            zip(layer_sizes[:-1], layer_sizes[1:])
        ):
            self.layers.append(
                Layer(
                    layers[ilayer],
                    input_size,
                    output_size,
                    activation=activations[ilayer],
                    **(list_kwargs[ilayer]),
                )
            )

        self.to(device=device, dtype=dtype)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class DNN(Module):
    def __init__(
        self,
        layer_sizes=[100, 1],
        dropout_prob=None,
        do_batch_norm=False,
        output_activation=None,
        input_transform: Optional[torch.nn.Module] = None,
        **kwargs,
    ):
        super().__init__()

        self.layers = torch.nn.ModuleList()
        if input_transform is not None:
            self.layers.append(input_transform)

        self.layer_sizes = layer_sizes[:-1]
        self.output_size = layer_sizes[-1]
        for input_size, output_size in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            if dropout_prob is not None:
                self.layers.append(torch.nn.Dropout(dropout_prob))
            self.layers.append(torch.nn.Linear(input_size, output_size))
            if do_batch_norm:
                self.layers.append(torch.nn.BatchNorm1d(output_size))
            self.layers.append(torch.nn.ReLU())

        self.layers.append(torch.nn.Linear(self.layer_sizes[-1], self.output_size))
        if output_activation is not None:
            if do_batch_norm:
                self.layers.append(torch.nn.BatchNorm1d(self.output_size))
            self.layers.append(output_activation)

    def forward(self, x, training=False):
        # print(training)
        self.train(mode=training)
        # print(x[0])
        # print(x)
        for layer in self.layers:
            x = layer(x)
            # print(x)
        return x


class Conv1dNN(Module):
    def __init__(
        self,
        sizes,
        output_activation=None,
        dropout_prob=None,
        do_batch_norm=False,
        **kwargs,
    ):
        super().__init__()

        assert len(sizes) > 1
        self.layers = torch.nn.ModuleList()
        self.layer_sizes = sizes[:-1]
        self.output_size = sizes[-1]
        for input_size, output_size in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            if dropout_prob is not None:
                self.layers.append(torch.nn.Dropout(dropout_prob))
            self.layers.append(torch.nn.Conv1d(input_size, output_size, kernel_size=1))
            if do_batch_norm:
                self.layers.append(torch.nn.BatchNorm1d(output_size))
            self.layers.append(torch.nn.ReLU())

        self.layers.append(
            torch.nn.Conv1d(self.layer_sizes[-1], self.output_size, kernel_size=1)
        )
        if output_activation is not None:
            if do_batch_norm:
                self.layers.append(torch.nn.BatchNorm1d(self.output_size))
            self.layers.append(output_activation)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class MatmulConv2d(torch.nn.Conv2d):
    """A `torch.nn.Conv2d` drop-in whose forward is a sum of per-kernel-offset matmuls.

    Mathematically identical to `nn.Conv2d`, but the compute is a handful of GEMMs
    instead of a cuDNN conv. Intended for the ensemble setting, where `torch.func.vmap`
    over per-replica weights lowers `nn.Conv2d` (via PyTorch's C++ convolution batching
    rule) to a **grouped** convolution with `groups = num_replicas`. On some GPU /
    cuDNN combinations that grouped conv is slow; this matmul form sidesteps it.

    **Caveat — benchmark before adopting.** On the hardware tested (RTX 4070, image
    `(2,9,9)`, channels 32→64, R=10–100) the *grouped conv was actually 4–8× FASTER*
    and lighter than this layer — cuDNN handles the small grouped conv well, while the
    matmul form pays for strided-slice copies and tiny-`K` GEMMs. Treat this as an
    opt-in tool for GPUs whose grouped conv is genuinely slow, not a universal win.

    Implementation: pad once, then for each of the `kh·kw` kernel offsets take the
    (strided) window — a **view** of the padded input — and accumulate
    `weight[:, :, i, j] @ window`. Because the windows are views, autograd saves the
    (small) padded input, not a replicated im2col buffer, so **both** the forward and
    the backward stay memory-light (unlike `F.unfold`, whose patches tensor is `kh·kw`×
    the input and is retained for the gradient). When the output is a single spatial
    cell (e.g. `kernel_size == input_size`, the user's "full-image conv"), it collapses
    to one GEMM over the flattened receptive field — a batched dense layer.

    Registers the *identical* params (`weight (O,C,kh,kw)`, `bias (O,)`), init, and
    `state_dict` keys as `nn.Conv2d`, so it is a transparent swap for the stateless
    `functional_call`/vmap path (and old checkpoints still load).

    Only `groups == 1` is supported (the ensemble dim comes from vmap, not the layer).
    `batch_chunk` (if set) splits a no-grad forward over sub-batches to bound peak
    memory during inference (`predict`); it does not reduce training peak (autograd
    retains every chunk's activations), where the view-based accumulation already keeps
    memory low.
    """

    def __init__(self, *args, batch_chunk: Optional[int] = None, **kwargs):
        super().__init__(*args, **kwargs)
        if self.groups != 1:
            raise ValueError("MatmulConv2d supports groups=1 only")
        self.batch_chunk = batch_chunk

    def _conv_matmul(self, x: Tensor) -> Tensor:
        # x: (N, C, H, W). Returns (N, O, H_out, W_out) == nn.Conv2d(x).
        n, c, h, w = x.shape
        o = self.out_channels
        if isinstance(self.padding, str):
            raise ValueError("MatmulConv2d does not support string padding")
        kh, kw = int(self.kernel_size[0]), int(self.kernel_size[1])
        sh, sw = int(self.stride[0]), int(self.stride[1])
        ph, pw = int(self.padding[0]), int(self.padding[1])
        dh, dw = int(self.dilation[0]), int(self.dilation[1])
        h_out = (h + 2 * ph - dh * (kh - 1) - 1) // sh + 1
        w_out = (w + 2 * pw - dw * (kw - 1) - 1) // sw + 1

        xp = x
        if ph or pw:
            xp = torch.nn.functional.pad(x, (pw, pw, ph, ph))

        if h_out == 1 and w_out == 1:
            # Single output cell (e.g. full-image kernel): one GEMM over the flattened
            # (strided) receptive field — order matches weight.reshape(O, C·kh·kw).
            rf = xp[:, :, : dh * (kh - 1) + 1 : dh, : dw * (kw - 1) + 1 : dw]
            out = self.weight.reshape(o, -1) @ rf.reshape(n, c * kh * kw, 1)  # (N,O,1)
        else:
            # Accumulate over kernel offsets; the windows are views of `xp`, so the
            # gradient saves `xp` once instead of a replicated im2col buffer.
            r0, r1 = 0, sh * (h_out - 1) + 1
            c0, c1 = 0, sw * (w_out - 1) + 1
            out = self.weight[:, :, 0, 0] @ xp[:, :, r0:r1:sh, c0:c1:sw].reshape(
                n, c, h_out * w_out
            )
            for i in range(kh):
                for j in range(kw):
                    if i == 0 and j == 0:
                        continue
                    ri, ci = i * dh, j * dw
                    win = xp[
                        :, :, ri : ri + r1 : sh, ci : ci + c1 : sw
                    ].reshape(n, c, h_out * w_out)
                    out = out + self.weight[:, :, i, j] @ win  # (O,C)@(N,C,L)->(N,O,L)

        out = out.reshape(n, o, h_out, w_out)
        if self.bias is not None:
            out = out + self.bias[..., :, None, None]
        return out

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[reportIncompatibleMethodOverride]
        # Promote an unbatched (C, H, W) (e.g. a per-replica image inside vmap) and drop
        # the axis again afterwards, mirroring nn.Conv2d's unbatched-input handling.
        if x.dim() == 3:
            return self._conv_matmul(x.unsqueeze(0)).squeeze(0)
        if self.batch_chunk is None or x.shape[0] <= self.batch_chunk:
            return self._conv_matmul(x)
        n = self.batch_chunk
        return torch.cat(
            [self._conv_matmul(x[i : i + n]) for i in range(0, x.shape[0], n)],
            dim=0,
        )


def _collapse_kernels(input_size, n_layers):
    """Per-layer **valid-conv** kernel sizes that collapse a square/rect grid to 1×1.

    Valid (padding=0) convs reduce each spatial axis by ``k-1`` per layer, so to land
    on exactly 1×1 the per-axis reductions must sum to ``S-1``. Spread that total as
    evenly as possible over ``n_layers`` (the first ``S-1 mod n`` layers take one extra),
    giving e.g. ``S=9, n=4 -> k=[3,3,3,3]`` (9→7→5→3→1) and ``S=9, n=2 -> k=[5,5]``
    (9→5→1). Returns a list of ``(kh, kw)`` tuples; raises if a schedule cannot reach
    exactly 1×1 without an axis going below 1 (so a bad depth/size fails loudly at
    construction, not as a shape error mid-forward).
    """
    if isinstance(input_size, int):
        h = w = input_size
    else:
        h, w = int(input_size[0]), int(input_size[1])

    def per_axis(size, axis):
        red = size - 1
        if red < 0:
            raise ValueError(f"input_size {axis}={size} must be >= 1")
        base, rem = divmod(red, n_layers)
        # First `rem` layers reduce by base+1, the rest by base; kernel = reduction + 1.
        reductions = [base + 1 if i < rem else base for i in range(n_layers)]
        kernels, cur = [], size
        for r in reductions:
            cur -= r
            if cur < 1:
                raise ValueError(
                    f"collapse_spatial: cannot reduce axis from {size} to 1 over "
                    f"{n_layers} valid convs (axis hit {cur})"
                )
            kernels.append(r + 1)
        if cur != 1:
            raise ValueError(
                f"collapse_spatial: axis ends at {cur}, not 1 "
                f"(input {size}, {n_layers} layers)"
            )
        return kernels

    kh, kw = per_axis(h, "H"), per_axis(w, "W")
    return list(zip(kh, kw))


class Conv2dNN(Module):
    """Small 2D-CNN classifier for grid-image inputs (e.g. (C, H, W) jet images).

    Mirrors the `Conv1dNN`/`DNN` idiom: an optional `input_transform` first, then
    `Conv2d -> ReLU` blocks (with optional `Dropout2d`), a global average pool to
    collapse the spatial grid, and a `Linear` head to `head_sizes[-1]` logits.

    Deliberately uses no BatchNorm: running-stat buffer mutation is fragile under
    the stateless `functional_call`/`vmap` training path, and inputs are expected
    to be standardized by `input_transform`.

    `collapse_spatial=True` (with `input_size`) makes the conv stack itself reduce the
    grid to **exactly 1×1** by the last conv via **valid** convolutions (per-layer
    kernels derived by `_collapse_kernels`), instead of preserving the grid and taking
    an unlearned global mean. Each valid conv integrates a growing receptive field, so
    the final 1×1 is a learned, position-aware summary of the whole grid (the trailing
    `AdaptiveAvgPool2d(1)` is then a no-op identity and is kept only as a safety net).
    """

    def __init__(
        self,
        in_channels,
        conv_channels=(32, 64),
        head_sizes=(1,),
        kernel_size=3,
        padding=1,
        dropout_prob=None,
        output_activation=None,
        input_transform: Optional[Module] = None,
        conv_impl: str = "conv",
        batch_chunk: Optional[int] = None,
        input_size: Optional[Union[int, tuple[int, int]]] = None,
        collapse_spatial: bool = False,
        device="cpu",
        dtype=torch.float32,
        **kwargs,
    ):
        super().__init__()

        # `conv_impl` selects the conv layer used inside the vmapped ensemble:
        #   "conv"   — stock `nn.Conv2d`; vmap lowers it to a grouped conv. On the
        #              GPUs tested this is the *fastest* path, so it is the default.
        #   "matmul" — `MatmulConv2d` (sum-of-offsets GEMMs); only worth trying on
        #              hardware whose grouped-conv kernels are slow. Benchmark first:
        #              on an RTX 4070 at (2,9,9)/32-64 it was ~4-8x SLOWER than "conv".
        if conv_impl not in ("conv", "matmul"):
            raise ValueError(f"conv_impl must be 'conv' or 'matmul', got {conv_impl!r}")

        # Gradual spatial collapse: derive per-layer valid-conv kernels (padding 0) that
        # reduce `input_size` to exactly 1×1 across the conv layers. Default off keeps
        # the size-preserving (kernel_size/padding scalar) behavior unchanged.
        if collapse_spatial:
            if input_size is None:
                raise ValueError("collapse_spatial=True requires input_size")
            conv_kernels = _collapse_kernels(input_size, len(conv_channels))
            conv_padding = 0
        else:
            conv_kernels = [kernel_size] * len(conv_channels)
            conv_padding = padding

        self.layers = torch.nn.ModuleList()
        if input_transform is not None:
            self.layers.append(input_transform)

        ch = in_channels
        for out_ch, k in zip(conv_channels, conv_kernels):
            if conv_impl == "matmul":
                conv = MatmulConv2d(
                    ch, out_ch, kernel_size=k, padding=conv_padding,
                    batch_chunk=batch_chunk,
                )
            else:
                conv = torch.nn.Conv2d(
                    ch, out_ch, kernel_size=k, padding=conv_padding
                )
            self.layers.append(conv)
            self.layers.append(torch.nn.ReLU())
            # Dropout AFTER Conv->ReLU (standard post-activation placement). Placing it
            # *before* the conv would zero whole input channels — for the (2,9,9)
            # bin-count image that randomly drops an entire physics channel.
            if dropout_prob is not None:
                self.layers.append(torch.nn.Dropout2d(dropout_prob))
            ch = out_ch

        # Collapse the spatial grid (C, H, W) -> (C,); negative start_dim keeps
        # this correct under vmap (the replica dim is mapped away).
        self.layers.append(torch.nn.AdaptiveAvgPool2d(1))
        self.layers.append(torch.nn.Flatten(start_dim=-3))

        head = [ch] + list(head_sizes)
        for i, (in_f, out_f) in enumerate(zip(head[:-1], head[1:])):
            self.layers.append(torch.nn.Linear(in_f, out_f))
            if i != len(head) - 2:
                self.layers.append(torch.nn.ReLU())
        if output_activation is not None:
            self.layers.append(output_activation)

        self.to(device=device, dtype=dtype)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class SumPooling(Module):
    def __init__(self, dim, mask_dim=1):
        super().__init__()
        self.dim = dim
        self.mask_dim = mask_dim

    def _resize_mask(self, mask, size, dim):
        return mask.select(dim, 0).unsqueeze_(dim).expand(size)

    def forward(self, x, mask=None):
        if mask is not None:
            _new_mask = self._resize_mask(mask, x.size(), self.mask_dim)
            x = torch.where(_new_mask, x, torch.zeros_like(x))
        return torch.sum(x, dim=self.dim)


class PFN(Module):
    def __init__(
        self,
        phi_sizes,
        f_sizes,
        output_activation=None,
        phi_dropout_prob=None,
        phi_do_batch_norm=False,
        f_dropout_prob=None,
        f_do_batch_norm=False,
    ):
        super().__init__()

        self.phi_sizes = phi_sizes
        self.f_sizes = [phi_sizes[-1]] + f_sizes

        assert len(self.phi_sizes) > 1 and len(self.f_sizes) > 1

        self.phi_layers = torch.nn.ModuleList()
        self.phi_layers.append(
            Conv1dNN(
                phi_sizes,
                output_activation=torch.nn.ReLU(),
                dropout_prob=phi_dropout_prob,
                do_batch_norm=phi_do_batch_norm,
            )
        )

        self.pooling_layer = SumPooling(-1)

        self.f_layers = torch.nn.ModuleList()
        self.f_layers.append(
            DNN(
                self.f_sizes,
                output_activation=output_activation,
                dropout_prob=f_dropout_prob,
                do_batch_norm=f_do_batch_norm,
            )
        )

    def forward(self, x, mask):
        for layer in self.phi_layers:
            x = layer(x)
        x = self.pooling_layer(x, mask)
        for layer in self.f_layers:
            x = layer(x)
        return x


class JetNN(Module):
    def __init__(
        self,
        phi_sizes,
        f_sizes,
        jet_dnn_sizes,
        output_dnn_sizes,
        dropout_prob=None,
        do_batch_norm=False,
    ):
        super().__init__()
        self.pfn = PFN(
            phi_sizes,
            f_sizes,
            phi_dropout_prob=dropout_prob,
            phi_do_batch_norm=do_batch_norm,
            f_dropout_prob=dropout_prob,
            f_do_batch_norm=do_batch_norm,
            output_activation=torch.nn.ReLU(),
        )
        self.jet_dnn = DNN(
            jet_dnn_sizes,
            dropout_prob=dropout_prob,
            do_batch_norm=do_batch_norm,
            output_activation=torch.nn.ReLU(),
        )
        self.output_dnn_sizes = [jet_dnn_sizes[-1] + f_sizes[-1]] + output_dnn_sizes
        self.output_dnn = DNN(
            self.output_dnn_sizes,
            dropout_prob=dropout_prob,
            do_batch_norm=do_batch_norm,
        )

    def forward(self, jet, constits, mask):
        jet_dnn_output = self.jet_dnn(jet)
        pfn_output = self.pfn(constits, mask)
        output = torch.cat([jet_dnn_output, pfn_output], dim=-1)
        output = self.output_dnn(output)
        return output
