from __future__ import annotations

import contextlib
from typing import Protocol
from ..utils import aio, shortuuid
from . import proto
from ..log import logger
from . import channel
import asyncio


class InferenceExecutor(Protocol):
    async def do_inference(self, method: str, data: bytes) -> bytes | None: ...


class _InferenceRunnerClient(InferenceExecutor):
    def __init__(self, *, cch: aio.duplex_unix._AsyncDuplex) -> None:
        self._cch = cch
        self._active_requests: dict[str, asyncio.Future[proto.InferenceResponse]] = {}

    async def do_inference(self, method: str, data: bytes) -> bytes | None:
        request_id = shortuuid("INF_")
        await channel.asend_message(
            self._cch,
            proto.InferenceRequest(request_id=request_id, method=method, data=data),
        )

        fut = asyncio.Future[proto.InferenceResponse]()
        self._active_requests[request_id] = fut
        return (await fut).data

    def _on_inference_response(self, resp: proto.InferenceResponse) -> None:
        fut = self._active_requests.pop(resp.request_id, None)
        if fut is None:
            logger.warning(
                "received unexpected inference response", extra={"resp": resp}
            )
            return

        print("got response", resp)
        with contextlib.suppress(asyncio.CancelledError):
            fut.set_result(resp)
