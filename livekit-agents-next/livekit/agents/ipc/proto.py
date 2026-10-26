from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

from livekit.protocol import agent

from ..job import JobAcceptArguments, RunningJobInfo


@dataclass
class InitializeRequest:
    """sent by the main process to the subprocess to initialize it. this is going to call initialize_process_fnc"""  # noqa: E501

    MSG_ID: ClassVar[int] = 0

    asyncio_debug: bool = False
    ping_interval: float = 0
    ping_timeout: float = 0  # if no response, process is considered dead
    # if ping is higher than this, process is considered unresponsive
    high_ping_threshold: float = 0
    http_proxy: str = ""  # empty = None


@dataclass
class InitializeResponse:
    """mark the process as initialized"""

    MSG_ID: ClassVar[int] = 1
    error: str = ""


@dataclass
class PingRequest:
    """sent by the main process to the subprocess to check if it is still alive"""

    MSG_ID: ClassVar[int] = 2
    timestamp: int = 0


@dataclass
class PongResponse:
    """response to a PingRequest"""

    MSG_ID: ClassVar[int] = 3
    last_timestamp: int = 0
    timestamp: int = 0


@dataclass
class StartJobRequest:
    """sent by the main process to the subprocess to start a job, the subprocess will only
    receive this message if the process is fully initialized (after sending a InitializeResponse)."""  # noqa: E501

    MSG_ID: ClassVar[int] = 4
    running_job: RunningJobInfo = field(init=False)

    def _to_wire(self) -> dict[str, Any]:
        info = self.running_job
        return {
            "job": info.job.SerializeToString(),
            "name": info.accept_arguments.name,
            "identity": info.accept_arguments.identity,
            "metadata": info.accept_arguments.metadata,
            "url": info.url,
            "token": info.token,
            "worker_id": info.worker_id,
            "fake_job": info.fake_job,
            # unix timestamps: a 32-bit float only resolves ~128 s at this magnitude
            "received_at": info.received_at,
            "accepted_at": info.accepted_at,
            "assigned_at": info.assigned_at,
            "launched_at": info.launched_at,
        }

    @staticmethod
    def _from_wire(fields: dict[str, Any]) -> StartJobRequest:
        job = agent.Job()
        job.ParseFromString(fields.pop("job"))
        accept_arguments = JobAcceptArguments(
            name=fields.pop("name"),
            identity=fields.pop("identity"),
            metadata=fields.pop("metadata"),
        )
        req = StartJobRequest()
        req.running_job = RunningJobInfo(accept_arguments=accept_arguments, job=job, **fields)
        return req


@dataclass
class ShutdownRequest:
    """sent by the main process to the subprocess to indicate that it should shut down
    gracefully. the subprocess will follow with a ExitInfo message"""

    MSG_ID: ClassVar[int] = 5
    reason: str = ""


@dataclass
class Exiting:
    """sent by the subprocess to the main process to indicate that it is exiting"""

    MSG_ID: ClassVar[int] = 6
    reason: str = ""


@dataclass
class InferenceRequest:
    """sent by a subprocess to the main process to request inference"""

    MSG_ID: ClassVar[int] = 7
    method: str = ""
    request_id: str = ""
    data: bytes = b""


@dataclass
class InferenceResponse:
    """response to an InferenceRequest"""

    MSG_ID: ClassVar[int] = 8
    request_id: str = ""
    data: bytes | None = None
    error: str = ""


@dataclass
class DumpStackTraceRequest:
    """sent by the main process to request a stack trace dump before killing"""

    MSG_ID: ClassVar[int] = 9


@dataclass
class ShutdownRequestAck:
    MSG_ID: ClassVar[int] = 10


@dataclass
class ShuttingDown:
    MSG_ID: ClassVar[int] = 11


IPC_MESSAGES = {
    InitializeRequest.MSG_ID: InitializeRequest,
    InitializeResponse.MSG_ID: InitializeResponse,
    PingRequest.MSG_ID: PingRequest,
    PongResponse.MSG_ID: PongResponse,
    StartJobRequest.MSG_ID: StartJobRequest,
    ShutdownRequest.MSG_ID: ShutdownRequest,
    Exiting.MSG_ID: Exiting,
    InferenceRequest.MSG_ID: InferenceRequest,
    InferenceResponse.MSG_ID: InferenceResponse,
    DumpStackTraceRequest.MSG_ID: DumpStackTraceRequest,
    ShutdownRequestAck.MSG_ID: ShutdownRequestAck,
    ShuttingDown.MSG_ID: ShuttingDown,
}
