from typing import TypeAlias, TypeVar

from ._generated import public_types as _public
from ._generated.public_types import *  # noqa: F403

__all__ = [
    "NotGivenOr",
]
__all__ += _public.__all__


_T = TypeVar("_T")


NotGivenOr: TypeAlias = _T | NotGiven
