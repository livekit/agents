from typing import assert_type

from livekit.agents.utils import log_exceptions


@log_exceptions()
def format_value(value: int, *, prefix: str = "") -> str:
    return prefix + str(value)


@log_exceptions()
async def fetch_value(key: str) -> int:
    return len(key)


async def decorated_results() -> None:
    assert_type(format_value(2, prefix="item-"), str)
    assert_type(await fetch_value("key"), int)
