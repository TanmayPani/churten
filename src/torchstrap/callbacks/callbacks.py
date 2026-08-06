"""Callback protocol.

torchstrap no longer has a training driver that fires ``on_*`` hooks — training
is a manual ``vmap`` loop (see ``examples/spirals/spirals_parallel.py``). A
callback is therefore just a **plain callable** the user drops anywhere in their
own loop; it takes what it needs explicitly (``state``, a ``(N,)`` score tensor,
…), mutates ``state`` in place and/or returns a value. This ``Protocol`` exists
only for typing/exports — concrete callbacks need not inherit it.
"""

from beartype.typing import Protocol, runtime_checkable


__all__ = ["Callback"]


@runtime_checkable
class Callback(Protocol):
    """Structural type for a torchstrap callback: anything callable."""

    def __call__(self, *args, **kwargs): ...
