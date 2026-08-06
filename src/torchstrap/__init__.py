from beartype.claw import beartype_all, beartype_this_package
from beartype import BeartypeConf

# Skip the modules that define `tensordict.TensorClass` subclasses (`State` /
# `OptimState` / `AdamState`). The claw otherwise decorates their TensorClass-
# generated methods and enforces tensordict's own (too-narrow) hints — e.g. it wraps
# `__getitem__`/`__setitem__` with an `item: NestedKey` hint that rejects the Tensor
# (fancy) indexing those classes rely on for masked per-replica row transfers. These
# classes get their type-safety from typed `TensorClass` fields + `frozen` instead.
beartype_this_package(
    conf=BeartypeConf(
        claw_skip_package_names=(
            "torchstrap.state",
            "torchstrap.optimizer.adam",
        ),
    )
)
#beartype_all(conf=BeartypeConf(violation_type=UserWarning))

from . import optimizer
from . import utils
from . import stateless
from . import history
from . import callbacks


__all__ = [
    "optimizer",
    "utils",
    "stateless",
    "history",
    "callbacks",
]


def hello() -> str:
    return "Hello from torchstrap!"
