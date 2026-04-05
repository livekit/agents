from __future__ import annotations

from dataclasses import dataclass

from livekit.protocol import agent
from livekit.protocol.agent_pb import agent_dev

from ..job import JobAcceptArguments, RunningJobInfo


@dataclass
class CliArgs:
    log_level: str
    url: str | None = None
    api_key: str | None = None
    api_secret: str | None = None
    reload_addr: str | None = None


def running_job_to_proto(info: RunningJobInfo) -> agent_dev.RunningAgentJobInfo:
    return agent_dev.RunningAgentJobInfo(
        job=info.job.SerializeToString(),
        accept_name=info.accept_arguments.name,
        accept_identity=info.accept_arguments.identity,
        accept_metadata=info.accept_arguments.metadata,
        url=info.url,
        token=info.token,
        worker_id=info.worker_id,
        mock_job=info.fake_job,
    )


def running_job_from_proto(pb: agent_dev.RunningAgentJobInfo) -> RunningJobInfo:
    return RunningJobInfo(
        accept_arguments=JobAcceptArguments(
            name=pb.accept_name,
            identity=pb.accept_identity,
            metadata=pb.accept_metadata,
        ),
        job=agent.Job.FromString(pb.job),
        url=pb.url,
        token=pb.token,
        worker_id=pb.worker_id,
        fake_job=pb.mock_job,
    )
