"""Token mode: a worker with ``LIVEKIT_TOKEN`` and no API key or secret, on a ``FakeLiveKit``.

The worker registers with the worker token. Every other token comes from the server through a
``CreateJobTokenRequest``:

- the job's inference token and telemetry token;
- one token for each server API call;
- a token for another participant.

The tests are hermetic: a real worker and job process, and a fake LiveKit server on loopback.
"""

from __future__ import annotations

import json
import os
import signal
import sys
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from livekit.protocol import agent, models
from tests.e2e.resources import STARTUP_SECONDS, Child, child_process
from tests.fakes import FakeLiveKit, until

WORKER_TOKEN = "the-worker-token-lk-agent-dev-holds"
JOB = agent.Job(
    id="AJ_token_mode",
    type=agent.JobType.JT_ROOM,
    room=models.Room(sid="RM_token_mode", name="token-room"),
    agent_name="token-agent",
)


@pytest.fixture
async def token_mode(
    tmp_path: Path, request: pytest.FixtureRequest
) -> AsyncIterator[tuple[FakeLiveKit, Path, Child]]:
    """Runs the worker on a ``FakeLiveKit`` with a worker token and no key or secret.

    With indirect parametrization, ``request.param`` gives extra ``start`` flags and extra
    environment variables.
    """
    flags, extra_env = getattr(request, "param", ((), {}))
    async with FakeLiveKit("server-key", "server-secret-server-secret-server") as livekit:
        env = {
            "PATH": "/usr/bin:/bin",
            "HOME": str(tmp_path),
            **({"LD_LIBRARY_PATH": lib} if (lib := os.environ.get("LD_LIBRARY_PATH")) else {}),
            "LIVEKIT_URL": livekit.url,
            "LIVEKIT_TOKEN": WORKER_TOKEN,
            "LIVEKIT_AGENTS_PRELOAD_LOCAL_INFERENCE": "0",
            "PYTHONUNBUFFERED": "1",
            "AGENT_DIR": str(tmp_path),
            **extra_env,
        }
        app = str(Path(__file__).with_name("token_mode_agent.py"))
        log = tmp_path / "agent.log"
        async with child_process(sys.executable, app, "start", *flags, log=log, env=env) as child:
            yield livekit, tmp_path, child


async def test_a_token_mode_worker_gets_every_job_token_from_the_server(
    token_mode: tuple[FakeLiveKit, Path, Child],
) -> None:
    """A token mode job uses `ctx.api` and asks for a participant token.

    Catches these bugs:

    - the worker token is signed, dropped, or sent anywhere other than the registration;
    - the job has no inference token or telemetry token;
    - `ctx.api` or the participant token is signed locally instead of created by the server;
    - a job token request asks for the agent grant or a sha256.
    """
    livekit, agent_dir, _ = token_mode
    await livekit.until_registered()
    [(_, authorization, _)] = livekit.connections
    assert authorization == f"Bearer {WORKER_TOKEN}"

    assert (await livekit.dispatch(JOB)).available
    job = agent_dir / "job.json"
    await until(job.exists, "the job ran", STARTUP_SECONDS)
    seen = json.loads(job.read_text())

    asked = {livekit.issued[r.request_id]: json.loads(r.claims) for r in livekit.job_token_requests}
    assert {r.job_id for r in livekit.job_token_requests} == {JOB.id}
    for claims in asked.values():
        assert not claims.get("video", {}).get("agent") and not claims.get("sha256"), claims
    claims = list(asked.values())
    assert {"identity": "agent", "inference": {"perform": True}} in claims
    assert {"observability": {"write": True}} in claims

    # The grant that the server API needs for DeleteRoom.
    [api_bearer] = livekit.api_bearers
    assert asked[api_bearer]["video"]["roomCreate"] is True

    participant = asked[seen["participant_token"]]
    assert (participant["identity"], participant["kind"]) == ("avatar", "agent")
    assert (participant["video"]["roomJoin"], participant["video"]["room"]) == (True, "token-room")
    assert WORKER_TOKEN not in [api_bearer, seen["participant_token"]]


@pytest.mark.parametrize(
    "token_mode",
    [(("--drain-timeout=0.1", "--shutdown-process-timeout=5"), {"AGENT_HOLD": "1"})],
    indirect=True,
)
async def test_a_jobs_shutdown_reaches_the_server_api_while_the_server_closes(
    token_mode: tuple[FakeLiveKit, Path, Child],
) -> None:
    """A token mode job calls the server API in its shutdown callback while the server stops.

    Catches a worker connection that closes before the job processes. In token mode the job's
    tokens come through that connection, so the shutdown callback would get no token.
    """
    livekit, agent_dir, child = token_mode
    await livekit.until_registered()
    assert (await livekit.dispatch(JOB)).available
    await until((agent_dir / "job.json").exists, "the job ran", STARTUP_SECONDS)

    child.process.send_signal(signal.SIGTERM)
    await until((agent_dir / "shutdown").exists, "the shutdown callback's call", STARTUP_SECONDS)
    [api_bearer] = livekit.api_bearers
    asked = {livekit.issued[r.request_id]: json.loads(r.claims) for r in livekit.job_token_requests}
    assert asked[api_bearer]["video"]["roomCreate"] is True
