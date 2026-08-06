import json
from collections import defaultdict

from beartype.typing import Any

import torch
from torch import Tensor

from torchstrap.utils import open_file_like


def list_list() -> list[list[Any]]:
    """Seed for a fresh batch-metric key: a list-of-lists-by-epoch with one
    (empty) epoch slot."""
    return [[]]


def _untensor(obj):
    """Recursively convert a nested dict/list structure into JSON-serializable
    Python, turning any ``Tensor`` leaf into a (nested) list. Replaces the old
    optree ``tree_map(untensor, ...)`` so ``history.py`` carries no pytree deps."""
    if isinstance(obj, Tensor):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _untensor(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_untensor(v) for v in obj]
    return obj


class History(defaultdict):
    """Host-side training-metrics log.

    Epoch-level metrics are flat lists keyed by name; batch-level metrics live
    under a nested ``"batches"`` dict shaped as list-of-lists-by-epoch. The
    stored record is **always host-resident**: device metric tensors handed to
    ``append_batch`` during an epoch are held in a transient device queue and
    drained to host in one bulk transfer by ``flush_epoch`` — call it once per
    epoch (the manual training loop drives it) so any host-side reader sees host
    tensors. Net: one device->host sync per epoch, never per batch.
    """

    def __init__(self, *args, **kwargs):
        # default_factory=None; key creation is handled by __missing__ below.
        super().__init__(None, *args, **kwargs)
        # Transient, per-epoch staging for (possibly on-device) batch metrics.
        # An instance attribute, NOT a dict key, so it is never serialized.
        self._batch_queue: dict[str, list[Tensor]] = {}

    @property
    def num_epochs(self) -> int:
        return len(self.get("iepoch", []))

    def new_epoch(self):
        self["iepoch"].append(self.num_epochs + 1)
        for key in self["batches"].keys():
            self["batches"][key].append([])

    def append(self, key: str, val: Any):
        self[key].append(val)

    def new_batch(self):
        ibatch = self["batches"]["ibatch"][-1]
        ibatch.append(len(ibatch) + 1)

    def append_batch(self, key: str, val: Any):
        # Stage the (possibly on-device) value; flush_epoch drains it to host.
        self._batch_queue.setdefault(key, []).append(val)

    def flush_epoch(self):
        """Drain this epoch's queued batch metrics to host and append them to the
        record. Each key's per-batch values are stacked and copied device->host
        non-blocking (into pinned memory), then a single stream sync materializes
        the whole epoch at once — one device->host sync per epoch, overlapping
        copies, and a record that is provably host-only afterwards."""
        if not self._batch_queue:
            return

        iepoch = self.num_epochs
        batches = self["batches"]

        issued_async = False
        pending: list[tuple[str, Tensor]] = []
        for key, vals in self._batch_queue.items():
            stacked = torch.stack(vals)
            if stacked.is_cuda:
                host = torch.empty(
                    stacked.shape, dtype=stacked.dtype,
                    device="cpu", pin_memory=True,
                )
                host.copy_(stacked, non_blocking=True)
                issued_async = True
            else:
                host = stacked
            pending.append((key, host))

        if issued_async:
            torch.cuda.current_stream().synchronize()

        for key, host in pending:
            slots = batches[key]
            # Align to the current epoch (pad earlier epochs for a late-appearing
            # key), then fill this epoch's slot with the per-batch host tensors.
            while len(slots) < iepoch:
                slots.append([])
            slots[iepoch - 1] = list(host.unbind(0))

        self._batch_queue = {}

    def to_dict(self):
        return _untensor(dict(self))

    def to_file(self, f):
        with open_file_like(f, "w") as fp:
            json.dump(self.to_dict(), fp, indent=4)

    @classmethod
    def from_file(cls, f):
        with open_file_like(f, "r") as fp:
            return cls(json.load(fp))

    def __missing__(self, key):
        self[key] = defaultdict(list_list) if key == "batches" else []
        return self[key]
