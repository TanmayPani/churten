import sys
import time
from itertools import cycle

import torch
from tabulate import tabulate

from torchstrap.utils import Ansi


__all__ = ["EpochTimer", "PrintLog"]


def filter_log_keys(keys, keys_ignored=None):
    """Filter out keys that are generally to be ignored."""
    keys_ignored = keys_ignored or ()
    for key in keys:
        if not (
                key == 'epoch' or
                (key in keys_ignored) or
                key.endswith('_best') or
                key.endswith('_batch_count') or
                key.startswith('event_')
        ):
            yield key


class EpochTimer:
    """Plain start/stop wall-clock timer.

    ``tic()`` marks the start; ``toc()`` returns the elapsed seconds since the
    last ``tic`` (and remembers it as ``.last``). Drop it around an epoch — or
    any span — in the manual loop.
    """

    def __init__(self):
        self._start = None
        self.last = None

    def tic(self) -> None:
        self._start = time.perf_counter()

    def toc(self) -> float:
        self.last = time.perf_counter() - (self._start or time.perf_counter())
        return self.last


class PrintLog:
    """Print one tabulated row of metrics per call.

    A plain callable: build a row from explicit values, e.g.
    ``PrintLog()(epoch=e, valid_loss=score, dur=timer.toc())``. ``(N,)`` tensor
    values are reduced to mean (and ``±std`` shown when it exceeds ``1e-5``);
    pass ``<key>_best`` as a bool to color the corresponding cell. The header is
    printed once, on the first call.
    """

    def __init__(
            self,
            keys_ignored=None,
            sink=print,
            tablefmt='simple',
            floatfmt='.4f',
            stralign='right',
    ):
        self.sink = sink
        self.tablefmt = tablefmt
        self.floatfmt = floatfmt
        self.stralign = stralign
        self.first_iteration_ = True

        if isinstance(keys_ignored, str):
            keys_ignored = [keys_ignored]
        self.keys_ignored_ = set(keys_ignored or [])

    def format_row(self, row, key, color):
        """Format a single cell (floats + best-coloring)."""
        value = row[key]

        if isinstance(value, bool) or value is None:
            return '+' if value else ''

        # Values arrive already reduced to python scalars by `_reduce`.
        if not isinstance(value, (int, float)):
            return value

        is_integer = float(value).is_integer()
        template = '{}' if is_integer else '{:' + self.floatfmt + '}'

        key_best = key + '_best'
        if (key_best in row) and row[key_best]:
            template = color + template + Ansi.ENDC.value

        d_key = "d_" + key
        if d_key in row:
            dvalue = row[d_key]
            return f"{template.format(value)}+/-{template.format(dvalue)}"
        return template.format(value)

    def _sorted_keys(self, keys):
        """'epoch' first, 'dur' last, 'event_*' just before 'dur', rest sorted."""
        sorted_keys = []

        if ('epoch' in keys) and ('epoch' not in self.keys_ignored_):
            sorted_keys.append('epoch')

        for key in filter_log_keys(sorted(keys), keys_ignored=self.keys_ignored_):
            if key != 'dur':
                sorted_keys.append(key)

        for key in sorted(keys):
            if key.startswith('event_') and (key not in self.keys_ignored_):
                sorted_keys.append(key)

        if ('dur' in keys) and ('dur' not in self.keys_ignored_):
            sorted_keys.append('dur')

        return sorted_keys

    def _reduce(self, metrics):
        """Collapse a raw metrics dict to a display row: reduce ``(N,)`` tensors
        to mean (+ ``d_<key>`` std when notable), keep scalars as-is."""
        row = {}
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor) and value.numel() > 1:
                std, mean = torch.std_mean(value.float())
                row[key] = mean.item()
                if std.item() > 1e-5:
                    row[f"d_{key}"] = std.item()
            elif isinstance(value, torch.Tensor):
                row[key] = value.item()
            else:
                row[key] = value
        return row

    def table(self, metrics):
        row = self._reduce(metrics)
        sorted_keys = self._sorted_keys(row.keys())
        colors = cycle([c.value for c in Ansi if c != Ansi.ENDC])

        headers, formatted = [], []
        for key, color in zip(sorted_keys, colors):
            cell = self.format_row(row, key, color=color)
            header = key[6:] if key.startswith('event_') else key
            headers.append(header)
            formatted.append(cell)

        return tabulate(
            [formatted],
            headers=headers,
            tablefmt=self.tablefmt,
            floatfmt=self.floatfmt,
            stralign=self.stralign,
        )

    def _sink(self, text, verbose):
        if (self.sink is not print) or verbose:
            self.sink(text)

    def __call__(self, metrics=None, *, verbose=True, **kw):
        metrics = {**(metrics or {}), **kw}
        tabulated = self.table(metrics)

        if self.first_iteration_:
            header, lines = tabulated.split('\n', 2)[:2]
            self._sink(header, verbose)
            self._sink(lines, verbose)
            self.first_iteration_ = False

        self._sink(tabulated.rsplit('\n', 1)[-1], verbose)
        if self.sink is print:
            sys.stdout.flush()
