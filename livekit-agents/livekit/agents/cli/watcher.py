from __future__ import annotations

import asyncio
import contextlib
import struct

from livekit.protocol.agent_pb import agent_dev

from .. import utils
from ..log import logger
from ..worker import AgentServer
from . import proto


async def _send_proto(writer: asyncio.StreamWriter, msg: bytes) -> None:
    writer.write(struct.pack("!I", len(msg)))
    writer.write(msg)
    await writer.drain()


async def _recv_proto(reader: asyncio.StreamReader) -> bytes:
    len_bytes = await reader.readexactly(4)
    length = struct.unpack("!I", len_bytes)[0]
    return await reader.readexactly(length)


class WatchClient:
    """Connects to Go CLI's reload server over TCP using DevMessage protobuf."""

    def __init__(
        self,
        worker: AgentServer,
        reload_addr: str,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        self._loop = loop or asyncio.get_event_loop()
        self._worker = worker
        self._reload_addr = reload_addr
        self._main_task: asyncio.Task | None = None

    def start(self) -> None:
        self._main_task = self._loop.create_task(self._run())

    @utils.log_exceptions(logger=logger)
    async def _run(self) -> None:
        host, port_str = self._reload_addr.rsplit(":", 1)
        reader, writer = await asyncio.open_connection(host, int(port_str))

        try:
            # On startup: send GetRunningJobsRequest to Go, recv response, reload jobs
            req = agent_dev.AgentDevMessage(
                get_running_jobs_request=agent_dev.GetRunningAgentJobsRequest()
            )
            await _send_proto(writer, req.SerializeToString())

            data = await _recv_proto(reader)
            resp = agent_dev.AgentDevMessage()
            resp.ParseFromString(data)

            if resp.HasField("get_running_jobs_response"):
                jobs_resp = resp.get_running_jobs_response
                if jobs_resp.jobs:
                    running_jobs = [proto.running_job_from_proto(j) for j in jobs_resp.jobs]
                    logger.info(f"reloading {len(running_jobs)} job(s)")
                    await self._worker._reload_jobs(running_jobs)

            # Listen for GetRunningJobsRequest from Go (capture before restart)
            while True:
                try:
                    data = await _recv_proto(reader)
                except (asyncio.IncompleteReadError, ConnectionError, OSError):
                    break

                msg = agent_dev.AgentDevMessage()
                msg.ParseFromString(data)

                if msg.HasField("get_running_jobs_request"):
                    jobs = self._worker.active_jobs
                    job_protos = [proto.running_job_to_proto(j) for j in jobs]
                    resp = agent_dev.AgentDevMessage(
                        get_running_jobs_response=agent_dev.GetRunningAgentJobsResponse(
                            jobs=job_protos
                        )
                    )
                    await _send_proto(writer, resp.SerializeToString())
        except (asyncio.IncompleteReadError, ConnectionError, OSError):
            pass
        finally:
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()

    async def aclose(self) -> None:
        if not self._main_task:
            return

        self._main_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._main_task
