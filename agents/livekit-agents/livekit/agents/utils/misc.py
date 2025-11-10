from __future__ import annotations

import platform
import time
import uuid
from typing import TypeVar

from typing_extensions import TypeGuard

from ..types import NotGiven, NotGivenOr

_T = TypeVar("_T")


def time_ms() -> int:
    return int(time.time() * 1000 + 0.5)


def shortuuid(prefix: str = "") -> str:
    return prefix + str(uuid.uuid4().hex)[:12]


def is_given(obj: NotGivenOr[_T]) -> TypeGuard[_T]:
    return not isinstance(obj, NotGiven)


def nodename() -> str:
    return platform.node()
