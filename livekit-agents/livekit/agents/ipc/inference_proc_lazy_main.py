from multiprocessing import current_process

if current_process().name == "inference_proc":
    import signal
    import sys

    # ignore signals in the inference process (the parent process will handle them)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)

    def _no_traceback_excepthook(exc_type, exc_val, traceback):
        if isinstance(exc_val, KeyboardInterrupt):
            return
        sys.__excepthook__(exc_type, exc_val, traceback)

    sys.excepthook = _no_traceback_excepthook


import asyncio
import socket
from dataclasses import dataclass

from ..inference_runner import _RunnersDict
from ..log import logger
from ..utils import aio, log_exceptions
from . import proto
from .channel import Message
from .proc_client import _ProcClient


@dataclass
class ProcStartArgs:
    log_cch: socket.socket
    mp_cch: socket.socket
    runners: _RunnersDict


def proc_main(args: ProcStartArgs) -> None:
    from .proc_client import _ProcClient

    inf_proc = _InferenceProc(args.runners)

    client = _ProcClient(
        args.mp_cch,
        args.log_cch,
        inf_proc.initialize,
        inf_proc.entrypoint,
    )

    client.initialize_logger()

    pid = current_process().pid
    logger.info("initializing inference process", extra={"pid": pid})
    client.initialize()
    logger.info("inference process initialized", extra={"pid": pid})

    client.run()


class _InferenceProc:
    def __init__(self, runners: _RunnersDict) -> None:
        # create an instance of each runner (the ctor must not requires any argument)
        self._runners = {name: runner() for name, runner in runners.items()}

    def initialize(
        self, init_req: proto.InitializeRequest, client: _ProcClient
    ) -> None:
        self._client = client

        for runner in self._runners.values():
            logger.debug(
                "initializing inference runner",
                extra={"runner": runner.__class__.INFERENCE_METHOD},
            )
            runner.initialize()

    @log_exceptions(logger=logger)
    async def entrypoint(self, cch: aio.ChanReceiver[Message]) -> None:
        async for msg in cch:
            if isinstance(msg, proto.InferenceRequest):
                await self._handle_inference_request(msg)

            if isinstance(msg, proto.ShutdownRequest):
                await self._client.send(proto.Exiting(reason=msg.reason))
                break

    async def _handle_inference_request(self, msg: proto.InferenceRequest) -> None:
        loop = asyncio.get_running_loop()

        if msg.method not in self._runners:
            logger.warning("unknown inference method", extra={"method": msg.method})

        try:
            data = await loop.run_in_executor(
                None, self._runners[msg.method].run, msg.data
            )
            await self._client.send(
                proto.InferenceResponse(
                    request_id=msg.request_id,
                    data=data,
                )
            )

        except Exception as e:
            logger.exception("error running inference")
            await self._client.send(
                proto.InferenceResponse(request_id=msg.request_id, error=str(e))
            )
