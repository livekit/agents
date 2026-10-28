from types import TracebackType
from typing import Protocol, cast

from livekit.agents import APIConnectOptions, APIError
from mistralai.client import Mistral


def request_timeout_ms(options: APIConnectOptions) -> int:
    # the SDK has no per-request way to disable a limit or to give phases different limits
    timeout = options.connect_timeout
    if (
        timeout is None
        or options.pool_timeout != timeout
        or options.read_timeout != timeout
        or options.write_timeout != timeout
        or not float(timeout * 1000).is_integer()
    ):
        raise APIError(
            "Mistral SDK applies one finite limit, in whole milliseconds, to every phase: "
            "pool/connect/write/read timeouts must be equal",
            retryable=False,
        )
    return int(timeout * 1000)


class _ClientLifecycle(Protocol):
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...


async def close_client(client: Mistral) -> None:
    # The SDK implements these public methods without annotating their parameters.
    lifecycle = cast(_ClientLifecycle, client)
    try:
        await lifecycle.__aexit__(None, None, None)
    finally:
        lifecycle.__exit__(None, None, None)
