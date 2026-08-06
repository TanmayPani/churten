import math
import itertools
from typing import Optional
from collections.abc import Sequence, Sized, Iterator

import torch
from torch import Tensor, LongTensor
from torch import Generator

from torch.nn.functional import one_hot

from torch.func import vmap

from torch.utils.data import Sampler


def random_split(
    *inputs,
    dim=0,
    stratify: Optional[Tensor] = None,
    num_replicas: int = 1,
    vmap_in_dims: int | tuple = (0, None),
    vmap_randomness: str = "different",
    **kwargs,
):
    lengths = [input.shape[dim] for input in inputs]

    label = None
    num_unique_label = 1
    label_counts = []

    if stratify is not None:
        unique_label, label, label_count_tensor = torch.unique(
            stratify,
            dim=0,
            return_inverse=True,
            return_counts=True,
        )
        num_unique_label = unique_label.shape[0]
        label_counts.extend(label_count_tensor.tolist())

    if num_replicas > 1:
        index_splits = vmap(
            _random_split,
            in_dims=vmap_in_dims,
            randomness=vmap_randomness,
        )(
            torch.arange(lengths[0]).expand(num_replicas, lengths[0]),
            dim=dim,
            label=label,
            num_unique_label=num_unique_label,
            label_counts=label_counts,
            **kwargs,
        )
        return [
            vmap(input.take_along_dim)(idx, dim=dim)
            for idx in index_splits
            for input in inputs
        ]

    index_splits = _random_split(
        torch.arange(lengths[0]),
        dim=dim,
        label=label,
        num_unique_label=num_unique_label,
        label_counts=label_counts,
        **kwargs,
    )
    return [
        input.take_along_dim(idx, dim=dim) for idx in index_splits for input in inputs
    ]


def undersample_and_random_split(
    input: Tensor,
    *,
    undersample_size: int | float = 0.5,
    split_sizes: Sequence[int | float] = (0.5, 0.5),
    dim: int = 0,
    stratify: Optional[Tensor] = None,
    generator: Optional[Generator] = None,
    num_replicas: int = 1,
    vmap_in_dims: int | tuple = 0,
    vmap_randomness: str = "different",
):
    label = None
    num_unique_label = 1
    label_counts = []

    if stratify is not None:
        unique_label, label, label_count_tensor = torch.unique(
            stratify,
            dim=0,
            return_inverse=True,
            return_counts=True,
        )
        num_unique_label = unique_label.shape[0]
        label_counts.extend(label_count_tensor.tolist())

    if num_replicas > 1:
        vmap_in_dims = (
            (vmap_in_dims, None) if isinstance(vmap_in_dims, int) else vmap_in_dims
        )
        subset_idx = vmap(
            _undersample,
            in_dims=vmap_in_dims,
            randomness=vmap_randomness,
        )(
            torch.arange(input.shape[dim]).expand(num_replicas, -1),
            label,
            size=undersample_size,
            num_unique_label=num_unique_label,
            label_counts=label_counts,
            generator=generator,
        )

        subset_label = (
            vmap(label.__getitem__)(subset_idx) if label is not None else None
        )

        split_vmap_in_dims = vmap_in_dims
        subset_label_counts = []
        if subset_label is not None:
            subset_label_counts.extend(
                subset_label[0][subset_label[0] == l].shape[0]
                for l in range(num_unique_label)
            )

            split_vmap_in_dims = (vmap_in_dims[0], vmap_in_dims[0])

        split_indices = vmap(
            _random_split,
            in_dims=split_vmap_in_dims,
            randomness=vmap_randomness,
        )(
            subset_idx,
            subset_label,
            sizes=split_sizes,
            num_unique_label=num_unique_label,
            label_counts=subset_label_counts,
            generator=generator,
        )

        return [vmap(input.take_along_dim)(idx, dim=dim) for idx in split_indices]

    subset_idx = _undersample(
        torch.arange(input.shape[dim]),
        size=undersample_size,
        label=label,
        num_unique_label=num_unique_label,
        label_counts=label_counts,
        generator=generator,
    )

    subset_label = label[subset_idx] if label is not None else None

    # One bincount pass instead of `num_unique_label` masked-equality scans.
    subset_label_counts = (
        torch.bincount(subset_label, minlength=num_unique_label).tolist()
        if subset_label is not None
        else []
    )

    split_indices = _random_split(
        subset_idx,
        sizes=split_sizes,
        label=subset_label,
        num_unique_label=num_unique_label,
        label_counts=subset_label_counts,
        generator=generator,
    )

    return [input.take_along_dim(index, dim=dim) for index in split_indices]


def _random_split(
    input,
    label: Optional[Tensor] = None,
    *,
    sizes: Sequence[float | int] = (0.5, 0.5),
    dim: int = 0,
    num_unique_label: int = 1,
    label_counts: Sequence[int] = [],
    generator: Optional[Generator] = None,
):

    if label is not None:
        return _stratified_split(
            input,
            label,
            sizes=sizes,
            dim=dim,
            num_unique_label=num_unique_label,
            label_counts=label_counts,
            generator=generator,
        )
    else:
        return _split(
            input,
            sizes=sizes,
            dim=dim,
            generator=generator,
        )


def _split(
    input: Tensor,
    sizes: Sequence[float | int],
    dim: int = 0,
    generator: Optional[Generator] = None,
):
    # g = generator or torch.default_generator
    input_size = input.shape[dim]
    index = torch.randperm(input_size, generator=generator)

    sum_of_sizes = sum(sizes)
    split_sizes = [math.floor((size / sum_of_sizes) * input_size) for size in sizes]
    remainder = input_size - sum(split_sizes)
    for irem in range(remainder):
        isplit = irem % len(split_sizes)
        split_sizes[isplit] += 1

    split_indices = [
        index[offset - length : offset]
        for offset, length in zip(itertools.accumulate(split_sizes), split_sizes)
    ]

    return [input.take_along_dim(idx, dim=dim) for idx in split_indices]


def _stratified_split(
    input: Tensor,
    label: Tensor,
    *,
    sizes: Sequence[float | int],
    dim: int = 0,
    num_unique_label: int = 1,
    label_counts: Sequence[int] = [],
    generator: Optional[Generator] = None,
):
    if num_unique_label == 1:
        return _split(input, sizes, dim=dim, generator=generator)

    assert label.shape == (input.shape[dim],)
    assert len(label_counts) == num_unique_label

    one_hot_label = one_hot(
        label,
        num_classes=num_unique_label,
    )

    sorted_one_hot_indices = torch.argsort(
        one_hot_label,
        dim=0,
        descending=True,
    )

    label_indices = [
        sorted_one_hot_indices[:, l][:lcount] for l, lcount in enumerate(label_counts)
    ]

    label_splits = [
        _split(
            input[lindex],
            sizes,
            dim=dim,
            generator=generator,
        )
        for lindex in label_indices
    ]

    splits = [torch.cat(lsplit) for lsplit in zip(*label_splits)]

    return [
        split[torch.randperm(split.shape[0], generator=generator)] for split in splits
    ]


def undersample(
    *inputs: Tensor,
    dim: int = 0,
    stratify: Optional[Tensor] = None,
    num_replicas: int = 1,
    vmap_in_dims: int | tuple = 0,
    vmap_randomness: str = "different",
    **kwargs,
) -> list[Tensor]:

    lengths = [input.shape[dim] for input in inputs]
    assert all(l == lengths[0] for l in lengths)

    label = None
    num_unique_label = 1
    label_counts = []
    if stratify is not None:
        unique_label, label, label_count_tensor = torch.unique(
            stratify,
            dim=0,
            return_inverse=True,
            return_counts=True,
        )
        num_unique_label = unique_label.shape[0]
        label_counts.extend(label_count_tensor.tolist())

    if num_replicas > 1:
        subsample_idx = vmap(
            _undersample,
            in_dims=vmap_in_dims,
            randomness=vmap_randomness,
        )(
            torch.arange(lengths[0]).expand(num_replicas, lengths[0]),
            label=label,
            num_unique_label=num_unique_label,
            label_counts=label_counts,
            **kwargs,
        )
        return [
            torch.vmap(input.take_along_dim)(subsample_idx, dim=dim) for input in inputs
        ]
        # return [input[subsample_idx.flatten()].reshape_as(subsample_idx) for input in inputs]

    subsample_idx = _undersample(
        torch.arange(lengths[0]),
        label=label,
        num_unique_label=num_unique_label,
        label_counts=label_counts,
        **kwargs,
    )

    return [input.take_along_dim(subsample_idx, dim=dim) for input in inputs]


def _undersample(
    input: Tensor,
    label: Optional[Tensor] = None,
    *,
    size: float | int = 0.5,
    dim: int = 0,
    num_unique_label: int = 1,
    label_counts: Sequence[int] = (),
    generator: Optional[Generator] = None,
) -> Tensor:

    input_size: int = input.shape[dim]
    sub_size: int = 0
    if isinstance(size, float):
        if size > 1.0:
            raise ValueError("Size argument must be <= 1.0 if float")
        sub_size = math.floor(size * input_size)
    elif isinstance(size, int):
        if size > input_size:
            raise ValueError("Integer size argument must be < input.shape[dim]!")
        sub_size = size

    if sub_size == 0:
        raise ValueError("Can't make subsample of size 0!!")

    if label is None or num_unique_label == 1:
        return input[torch.randperm(input_size, generator=generator)[:sub_size]]

    assert label.shape == (input_size,)

    one_hot_label = one_hot(label, num_classes=num_unique_label)

    sorted_one_hot_indices = torch.argsort(one_hot_label, dim=0, descending=True)
    label_indices = [
        sorted_one_hot_indices[:, l][:lcount] for l, lcount in enumerate(label_counts)
    ]

    label_sub_counts = [
        math.floor(float(sub_size / input_size) * lcount) for lcount in label_counts
    ]

    remainder = sub_size - sum(label_sub_counts)

    for i in range(remainder):
        label_sub_counts[i % num_unique_label] += 1

    sub_index = torch.cat(
        [
            lindex[torch.randperm(lindex.shape[0], generator=generator)[:lcount]]
            for lcount, lindex in zip(label_sub_counts, label_indices)
        ]
    )[torch.randperm(sub_size, generator=generator)]

    return input.take_along_dim(sub_index, dim=dim)


class TensorBatchSampler(Sampler[LongTensor]):
    """Yield contiguous batches of a pre-permuted ``source`` index tensor.

    Batches are sequential slices along ``batch_dim`` (the caller shuffles
    upstream, e.g. via ``random_split``/``undersample_and_random_split``), so
    each batch is a cheap ``narrow`` **view** — no per-batch Python index list or
    advanced-indexing gather.
    """

    def __init__(
        self,
        source: Tensor,
        batch_size: int = 32,
        batch_dim: int = 0,
        drop_last: bool = False,
    ) -> None:
        self.source = source
        self.batch_dim = batch_dim
        self.batch_size = batch_size
        self.drop_last = drop_last
        if self.source.shape[self.batch_dim] < batch_size:
            raise ValueError("Subsample size should be greater than batch size!")

    def __iter__(self) -> Iterator[LongTensor]:
        n = self.source.shape[self.batch_dim]
        bs = self.batch_size
        for start in range(0, n, bs):
            size = min(bs, n - start)
            if self.drop_last and size < bs:
                break
            yield self.source.narrow(self.batch_dim, start, size)

    def __len__(self) -> int:
        n = self.source.shape[self.batch_dim]
        bs = self.batch_size
        return n // bs if self.drop_last else -(-n // bs)


class MultiSubsetBatchSampler(TensorBatchSampler):
    def __init__(
        self,
        dataset: Sized,
        *,
        num_replicas: int = 2,
        subsample_size: float | int = 1.0,
        batch_size: int = 32,
        stratify: Optional[Tensor] = None,
        generator: Optional[Generator] = None,
        vmap_randomness: str = "different",
        drop_last: bool = False,
    ) -> None:

        self.length = len(dataset)
        self.subsample_size = (
            subsample_size
            if isinstance(subsample_size, int)
            else math.floor(self.length * subsample_size)
        )

        stacked_indices = vmap(
            undersample,
            randomness=vmap_randomness,
        )(
            torch.arange(self.length).expand(num_replicas, -1),
            size=self.subsample_size,
            stratify=stratify,
            generator=generator,
        )

        super().__init__(
            stacked_indices, batch_size=batch_size, batch_dim=1, drop_last=drop_last
        )


def train_test_multi_subset_samplers(
    indices: Tensor,
    *,
    train_size: float | int,
    num_replicas: int = 2,
    batch_size: int = 32,
    subsample_size: float | int = 1.0,
    generator: Optional[Generator] = None,
    stratify: Optional[Tensor] = None,
    vmap_randomness: str = "different",
    vmap_in_dims: int | tuple = 0,
    drop_last: bool = False,
):
    length = indices.shape[0]
    subsample_size = (
        subsample_size
        if isinstance(subsample_size, int)
        else math.floor(length * subsample_size)
    )

    train_size = (
        train_size
        if isinstance(train_size, int)
        else math.floor(subsample_size * train_size)
    )

    if train_size > subsample_size:
        raise ValueError("training set size must be smaller than subset size.")
    valid_size = subsample_size - train_size

    train_subset_stacked, valid_subset_stacked = undersample_and_random_split(
        indices,
        num_replicas=num_replicas,
        undersample_size=subsample_size,
        split_sizes=(train_size, valid_size),
        stratify=stratify,
        vmap_in_dims=vmap_in_dims,
        vmap_randomness=vmap_randomness,
        generator=generator,
    )

    return (
        TensorBatchSampler(
            train_subset_stacked,
            batch_size=batch_size,
            batch_dim=1,
            drop_last=drop_last,
        ),
        TensorBatchSampler(
            valid_subset_stacked,
            batch_size=batch_size,
            batch_dim=1,
            drop_last=drop_last,
        ),
    )


if __name__ == "__main__":
    # idx = torch.stack([torch.arange(1000) for _ in range(10)])

    idx = torch.arange(1000)
    do_stratify = True

    if do_stratify:
        stratify = torch.cat(
            [
                torch.zeros(700),
                torch.ones(300),
            ]
        )
    else:
        stratify = None

    g = torch.Generator().manual_seed(0)

    num_replicas = 10

    (idx_sub,) = undersample(
        idx,
        num_replicas=num_replicas,
        stratify=stratify,
        generator=g,
    )

    print(idx_sub, idx_sub.shape)

    if stratify is not None:
        stratify_sub = (
            vmap(stratify.__getitem__)(idx_sub)
            if num_replicas > 1
            else stratify[idx_sub]
        )
        print(torch.sum(stratify_sub, dim=-1))

    idx0, idx1 = undersample_and_random_split(
        idx,
        num_replicas=num_replicas,
        stratify=stratify,
        generator=g,
    )

    print(idx0, idx0.shape)
    print(idx1, idx1.shape)

    if stratify is not None:
        stratify0 = (
            vmap(stratify.__getitem__)(idx0) if num_replicas > 1 else stratify[idx0]
        )
        stratify1 = (
            vmap(stratify.__getitem__)(idx1) if num_replicas > 1 else stratify[idx1]
        )

        sum0, sum1 = torch.sum(stratify0, dim=-1), torch.sum(stratify1, dim=-1)
        print(sum0, sum1)
        print(sum0.shape, sum1.shape)
