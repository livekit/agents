"""Hot reload under ``lk agent dev``.

The CLI keeps a call's job while the agent restarts. It gives the kept job back over the dev
channel, and the restarted worker resumes it. These tests run a real worker, a fake dev channel,
and a fake LiveKit server on loopback.
"""

from __future__ import annotations

import asyncio
import json
import os
import struct
import sys
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from livekit import api
from livekit.protocol import agent, models
from livekit.protocol.agent_pb import agent_dev
from tests.e2e.resources import STARTUP_SECONDS, child_process
from tests.fakes import FakeLiveKit, until

KEY, SECRET = "server-key", "server-secret-server-secret-server"
WORKER_TOKEN = "the-worker-token-lk-agent-dev-holds"
REFRESHED_TOKEN = "the-worker-token-lk-agent-dev-refreshed"
KEPT = agent.Job(
    id="AJ_kept",
    type=agent.JobType.JT_ROOM,
    room=models.Room(sid="RM_kept", name="kept-room"),
    agent_name="token-agent",
)
REFUSED = agent.Job(
    id="AJ_refused",
    type=agent.JobType.JT_ROOM,
    room=models.Room(sid="RM_kept", name="kept-room"),
    agent_name="token-agent",
)


@dataclass
class DevChannel:
    """The CLI's end of the dev channel.

    It answers the agent's running-jobs request with the kept jobs. ``refresh`` sends a new worker
    token and returns after the agent read it.
    """

    kept: agent_dev.RunningAgentJobInfo
    others: list[agent_dev.RunningAgentJobInfo] = field(default_factory=list)
    seen: list[agent_dev.AgentDevMessage] = field(default_factory=list)
    addr: str = ""
    _writer: asyncio.StreamWriter | None = None
    _answers: asyncio.Queue[None] = field(default_factory=asyncio.Queue)

    async def refresh(self, token: str) -> None:
        assert self._writer is not None, "the agent never opened the channel"
        self._send(
            self._writer, agent_dev.AgentDevMessage(credentials=agent_dev.Credentials(token=token))
        )
        # The agent reads the channel in order, so the answer comes after it took the token.
        await self.capture()

    async def capture(self) -> list[str]:
        """Asks the agent for its running jobs, as the CLI does before a reload. Returns the job
        ids."""
        await until(
            lambda: self._writer is not None, "the agent opened the channel", STARTUP_SECONDS
        )
        assert self._writer is not None
        request = agent_dev.GetRunningAgentJobsRequest()
        self._send(self._writer, agent_dev.AgentDevMessage(get_running_jobs_request=request))
        await self._writer.drain()
        await asyncio.wait_for(self._answers.get(), STARTUP_SECONDS)
        [*_, answer] = [m for m in self.seen if m.HasField("get_running_jobs_response")]
        jobs = answer.get_running_jobs_response.jobs
        return [agent.Job.FromString(job.job).id for job in jobs]

    async def serve(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self._writer = writer
        try:
            await self._read(reader, writer)
        finally:
            writer.close()

    async def _read(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        while True:
            try:
                (length,) = struct.unpack(">I", await reader.readexactly(4))
                message = agent_dev.AgentDevMessage.FromString(await reader.readexactly(length))
            except (asyncio.IncompleteReadError, ConnectionError):
                return
            self.seen.append(message)
            if message.HasField("get_running_jobs_request"):
                response = agent_dev.GetRunningAgentJobsResponse(jobs=[*self.others, self.kept])
                self._send(writer, agent_dev.AgentDevMessage(get_running_jobs_response=response))
                await writer.drain()
            elif message.HasField("get_running_jobs_response"):
                self._answers.put_nowait(None)

    @staticmethod
    def _send(writer: asyncio.StreamWriter, message: agent_dev.AgentDevMessage) -> None:
        data = message.SerializeToString()
        writer.write(struct.pack(">I", len(data)) + data)


def kept_job(job: agent.Job = KEPT) -> agent_dev.RunningAgentJobInfo:
    """A kept job as the CLI stores it, with the join token that the server gave the agent."""
    token = (
        api.AccessToken(KEY, SECRET)
        .with_identity("agent-7")
        .with_kind("agent")
        .with_grants(api.VideoGrants(room_join=True, room="kept-room", agent=True))
        .to_jwt()
    )
    return agent_dev.RunningAgentJobInfo(
        job=job.SerializeToString(),
        accept_identity="agent-7",
        url="ws://127.0.0.1:9",
        token=token,
        worker_id="AW_before_reload",
    )


@pytest.fixture
async def reloaded(
    tmp_path: Path, request: pytest.FixtureRequest
) -> AsyncIterator[tuple[FakeLiveKit, DevChannel, Path]]:
    """A worker restarted under ``lk agent dev`` with a kept job.

    ``request.param`` selects the setup:

    - ``False``: key mode.
    - ``True``: token mode, with a worker token and no key.
    - ``"refused"``: token mode with two kept jobs. The server refuses the token of the first
      job, ``AJ_refused``.
    """
    token_mode = bool(request.param)
    async with FakeLiveKit(KEY, SECRET) as livekit:
        channel = DevChannel(kept_job())
        if request.param == "refused":
            channel.others.append(kept_job(REFUSED))
            livekit.refused_job_tokens.add(REFUSED.id)
        server = await asyncio.start_server(channel.serve, "127.0.0.1", 0)
        channel.addr = f"127.0.0.1:{server.sockets[0].getsockname()[1]}"
        credentials = (
            {"LIVEKIT_TOKEN": WORKER_TOKEN}
            if token_mode
            else {"LIVEKIT_API_KEY": KEY, "LIVEKIT_API_SECRET": SECRET}
        )
        env = {
            "PATH": "/usr/bin:/bin",
            "HOME": str(tmp_path),
            **({"LD_LIBRARY_PATH": lib} if (lib := os.environ.get("LD_LIBRARY_PATH")) else {}),
            "LIVEKIT_URL": livekit.url,
            **credentials,
            "LIVEKIT_AGENTS_PRELOAD_LOCAL_INFERENCE": "0",
            "PYTHONUNBUFFERED": "1",
            "AGENT_DIR": str(tmp_path),
            "AGENT_HOLD": "1",
        }
        app = str(Path(__file__).with_name("token_mode_agent.py"))
        args = ("start", "--dev", "--cli-addr", channel.addr, "--drain-timeout=0.1")
        async with server, child_process(sys.executable, app, *args, log=tmp_path / "log", env=env):
            yield livekit, channel, tmp_path


@pytest.mark.parametrize("reloaded", [False], indirect=True)
async def test_a_kept_job_is_reported_and_held_after_a_reload(
    reloaded: tuple[FakeLiveKit, DevChannel, Path],
) -> None:
    """Key mode reload, then the server ends the kept job.

    Catches a kept job that resumes without a `MigrateJobRequest`, or that the worker does not
    hold. Then the worker ignores the server's termination and refuses the job's status.
    """
    livekit, _, agent_dir = reloaded
    await until((agent_dir / "job.json").exists, "the kept job resumed", STARTUP_SECONDS)
    await until(lambda: livekit.migrations, "the kept job reported", STARTUP_SECONDS)
    [job] = livekit.migrations[0].jobs
    assert (job.id, job.state.participant_identity) == (KEPT.id, "agent-7")

    await livekit.terminate(KEPT.id)
    ended = {agent.JobStatus.JS_SUCCESS, agent.JobStatus.JS_FAILED}
    await until(
        lambda: ended & set(livekit.job_statuses.get(KEPT.id, [])),
        "the kept job ended and its status reached the server",
        STARTUP_SECONDS,
    )


@pytest.mark.parametrize("reloaded", [True], indirect=True)
async def test_a_token_mode_reload_asks_the_server_for_the_jobs_new_token(
    reloaded: tuple[FakeLiveKit, DevChannel, Path],
) -> None:
    """Token mode reload, then the CLI sends a new worker token and the connection drops.

    Catches these bugs:

    - the worker signs a token itself, or reuses the old join token;
    - the token request asks for the agent grant;
    - the token request goes out before the `MigrateJobRequest`;
    - the next registration does not use the new worker token.
    """
    livekit, channel, agent_dir = reloaded
    await until((agent_dir / "job.json").exists, "the kept job resumed", STARTUP_SECONDS)
    [info] = [m.server_info for m in channel.seen if m.HasField("server_info")]
    assert info.supports_token_mode
    # Once before its token request, and again when the worker adopts the launched job.
    assert {m.jobs[0].id for m in livekit.migrations} == {KEPT.id}
    reload = [r for r in livekit.job_token_requests if "video" in json.loads(r.claims)]
    [request] = reload
    claims = json.loads(request.claims)
    assert request.job_id == KEPT.id
    assert (claims["identity"], claims["kind"]) == ("agent-7", "agent")
    assert (claims["video"]["roomJoin"], claims["video"]["room"]) == (True, "kept-room")
    assert "agent" not in claims["video"]
    assert "sub" not in claims and "sha256" not in claims

    await channel.refresh(REFRESHED_TOKEN)
    await livekit.disconnect()
    await until(lambda: len(livekit.connections) == 2, "the worker registered again", 10)
    assert [authorization for _, authorization, _ in livekit.connections] == [
        f"Bearer {WORKER_TOKEN}",
        f"Bearer {REFRESHED_TOKEN}",
    ]


@pytest.mark.parametrize("reloaded", ["refused"], indirect=True)
async def test_a_job_the_server_refuses_ends_on_reload_and_the_others_resume(
    reloaded: tuple[FakeLiveKit, DevChannel, Path],
) -> None:
    """Two kept jobs, and the server refuses the first job's token, as a self-hosted server does.

    Catches a refused job that launches anyway or is reported to the next reload, a refusal that
    stops the other job, and a refusal without the reload warning.
    """
    livekit, channel, agent_dir = reloaded
    await until((agent_dir / "job.json").exists, "the other kept job resumed", STARTUP_SECONDS)
    log = (agent_dir / "log").read_text()
    assert "job ended on reload. reload is only supported when LiveKit API key and secret" in log
    assert await channel.capture() == [KEPT.id]
