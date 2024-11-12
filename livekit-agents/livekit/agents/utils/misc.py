from __future__ import annotations

import asyncio
import time
import uuid

from ..types import NotGivenOr, NotGiven, APIConnectionError
from ..log import logger

from typing import TypeVar, TypeGuard, Callable, Awaitable, Any

_T = TypeVar("_T")


def time_ms() -> int:
    return int(time.time() * 1000 + 0.5)


def shortuuid(prefix: str = "") -> str:
    return prefix + str(uuid.uuid4().hex)[:12]


def is_given(obj: NotGivenOr[_T]) -> TypeGuard[_T]:
    return not isinstance(obj, NotGiven)
