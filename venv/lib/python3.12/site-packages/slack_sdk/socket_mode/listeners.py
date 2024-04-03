from typing import Optional

from slack_sdk.socket_mode.request import SocketModeRequest


class WebSocketMessageListener:
    def __call__(  # type: ignore
        client: "BaseSocketModeClient",  # noqa: F821
        message: dict,
        raw_message: Optional[str] = None,
    ):  # noqa: F821
        raise NotImplementedError()


class SocketModeRequestListener:
    def __call__(client: "BaseSocketModeClient", request: SocketModeRequest):  # type: ignore  # noqa: F821  # noqa: F821
        raise NotImplementedError()
