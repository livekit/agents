from typing import Optional, Callable

from slack_sdk.socket_mode.request import SocketModeRequest


class AsyncWebSocketMessageListener(Callable):
    async def __call__(  # type: ignore
        client: "AsyncBaseSocketModeClient",  # noqa: F821
        message: dict,
        raw_message: Optional[str] = None,
    ):  # noqa: F821
        raise NotImplementedError()


class AsyncSocketModeRequestListener(Callable):
    async def __call__(  # type: ignore
        client: "AsyncBaseSocketModeClient",  # noqa: F821
        request: SocketModeRequest,
    ):  # noqa: F821
        raise NotImplementedError()
