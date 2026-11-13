"""LiveKit Cloud hosting: an agent run as a Cloud pod, observed as Cloud observes it.

The pod is ``cloud_pod.py start`` with the variables ``cloud-agents`` injects
(``scheduling_provider/kube.go``, the admission webhook and the secret injector), registered on a
``FakeLiveKit``. Kubernetes probes ``/`` on the health port; the ``cloud-agents`` watcher reads
``/worker`` with Go's ``encoding/json`` into ``livekit.WorkerInfo``, whose JSON tags are the proto
field names; a terminating pod receives SIGTERM.

Hermetic: real OS processes and signals, LiveKit faked on loopback. Not a real pod: the image,
``run.sh`` and the sandbox are ``cloud-agents``' to test.
"""

from __future__ import annotations

import asyncio
import json
import signal
import socket
import sys
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import httpx
import pytest

from livekit import api
from livekit.agents import __version__
from livekit.protocol import agent, models
from tests.e2e.resources import STARTUP_SECONDS, Child, child_process, gone
from tests.fakes import FakeLiveKit, until

API_KEY, API_SECRET = "pod-key", "pod-secret-pod-secret-pod-secret-pod"
REGION = "us-east"
WORKER_TOKEN = "wt+pod/token="  # characters a query string must escape
IN_CLUSTER_INFERENCE = "http://inference.livekit.svc.cluster.local/v1"
DRAIN_TIMEOUT = 0.5  # the drain a test cuts short, instead of the production hour
PROCESS_EXIT_TIMEOUT = 0.5  # what a job process is given to exit, instead of the production 10 s
SHUTDOWN_SLACK = 12.0  # what the pod may take past its drain: job and pool cleanup
# `livekit.WorkerInfo`'s JSON tags: the only keys the watcher decodes
WORKER_INFO_KEYS = {
    "worker_type",
    "agent_name",
    "active_jobs",
    "sdk_version",
    "worker_load",
    "protocol_version",
}
JOB = agent.Job(
    id="AJ_pod",
    type=agent.JobType.JT_ROOM,
    room=models.Room(sid="RM_pod", name="pod-room"),
    agent_name="cloud-agent",
)
LATE_JOB = agent.Job(
    id="AJ_late",
    type=agent.JobType.JT_ROOM,
    room=models.Room(sid="RM_late", name="late-room"),
    agent_name="cloud-agent",
)


class JobView(TypedDict):
    """What ``cloud_pod.py``'s job writes once it runs."""

    pid: int
    env: dict[str, str]
    inference_headers: dict[str, str]
    inference_url: str
    hosted: bool


@dataclass
class Pod:
    child: Child
    livekit: FakeLiveKit
    health: httpx.AsyncClient
    dir: Path

    async def registered(self) -> None:
        self.livekit.answer_registration()
        await self.livekit.until_registered()

    async def in_a_call(self) -> JobView:
        """Dispatches ``JOB``; what the job saw once it runs."""
        assert (await self.livekit.dispatch(JOB)).available
        job = self.dir / "job.json"
        await until(job.exists, "the job started", STARTUP_SECONDS)
        seen: JobView = json.loads(job.read_text())
        return seen

    async def until_health(self, path: str, status: int, timeout: float) -> httpx.Response:
        async with asyncio.timeout(timeout):
            while (response := await self.health.get(path)).status_code != status:
                await asyncio.sleep(0.05)
        return response

    def hang_up(self) -> None:
        (self.dir / "hangup").touch()

    async def exit_code(self, within: float = STARTUP_SECONDS) -> int:
        process = self.child.process
        await until(lambda: process.returncode is not None, "the pod exited", within)
        assert process.returncode is not None
        return process.returncode

    async def draining(self) -> None:
        """SIGTERM, as kubelet sends a terminating pod; returns once LiveKit sees it full."""
        self.child.process.send_signal(signal.SIGTERM)
        await until(lambda: not self.livekit.available, "the pod reported itself full", 10)


@pytest.fixture
async def pod(tmp_path: Path, request: pytest.FixtureRequest) -> AsyncIterator[Pod]:
    """The pod, ``start`` given the flags in ``request.param`` when parametrized indirectly."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        health_port = probe.getsockname()[1]
    async with FakeLiveKit(API_KEY, API_SECRET, region=REGION, hold_registration=True) as livekit:
        env = {
            "PATH": "/usr/bin:/bin",
            "HOME": str(tmp_path),
            # scheduling_provider/kube.go: the pod template
            "LK_HOSTED_AGENTS": "1",
            "LIVEKIT_REGION_NAME": REGION,
            "LIVEKIT_DEPLOYED_AGENT_VERSION": "v20261110",
            "LIVEKIT_REMOTE_EOT_URL": "http://cloud-infserv.livekit.svc.cluster.local:7858",
            "LIVEKIT_INFERENCE_URL": IN_CLUSTER_INFERENCE,
            "LIVEKIT_AGENT_DEPLOYMENT": "staging",
            "LIVEKIT_AGENT_ID": "CA_pod",
            # the admission webhook
            "LIVEKIT_WORKER_ID": "CAW_pod",
            "LIVEKIT_WORKER_TOKEN": WORKER_TOKEN,
            # the secret injector
            "LIVEKIT_URL": livekit.url,
            "LIVEKIT_API_KEY": API_KEY,
            "LIVEKIT_API_SECRET": API_SECRET,
            # the test's
            "LIVEKIT_AGENTS_PRELOAD_LOCAL_INFERENCE": "0",
            "PYTHONUNBUFFERED": "1",
            "POD_DIR": str(tmp_path),
            "POD_HEALTH_PORT": str(health_port),
        }
        app = str(Path(__file__).with_name("cloud_pod.py"))
        async with (
            child_process(
                sys.executable,
                app,
                "start",
                *getattr(request, "param", ()),
                log=tmp_path / "pod.log",
                env=env,
            ) as child,
            httpx.AsyncClient(
                base_url=f"http://127.0.0.1:{health_port}", trust_env=False
            ) as health,
        ):
            pod = Pod(child, livekit, health, tmp_path)
            try:
                yield pod
            finally:
                pod.hang_up()


async def test_the_pod_registers_with_its_worker_token_and_deployment(pod: Pod) -> None:
    """Catches: the worker token missing from `/agent` (Cloud refuses the worker), the deployment
    lost from the registration, or credentials not read from the pod's environment."""
    await pod.registered()

    [(query, authorization, register)] = pod.livekit.connections
    assert query == {"worker_token": WORKER_TOKEN}
    scheme, _, jwt = authorization.partition(" ")
    assert scheme == "Bearer"
    grants = api.TokenVerifier(API_KEY, API_SECRET).verify(jwt)
    assert grants.video is not None and grants.video.agent
    assert (register.type, register.agent_name, register.deployment, register.version) == (
        agent.JobType.JT_ROOM,
        "cloud-agent",
        "staging",
        __version__,
    )


async def test_the_pod_is_live_only_once_registered(pod: Pod) -> None:
    """Catches: kubelet told a pod is live before LiveKit accepted it, or never told."""
    await pod.child.ready(f"{pod.health.base_url}/worker")
    assert (await pod.health.get("/")).status_code == 503

    await pod.registered()
    assert (await pod.until_health("/", 200, STARTUP_SECONDS)).text == "OK"


async def test_a_fresh_pod_takes_its_first_call_at_once(pod: Pod) -> None:
    """Catches: a pod that has not measured its load yet reporting itself full, so the first call
    of a pod scaled up from zero is refused or waits for the next status."""
    await pod.registered()
    await until(lambda: pod.livekit.statuses, "the first status", STARTUP_SECONDS)

    assert pod.livekit.statuses[0].status == agent.WorkerStatus.WS_AVAILABLE
    assert (await pod.livekit.dispatch(JOB)).available


async def test_a_disconnected_pod_stays_live_and_takes_no_traffic_until_it_reconnects(
    pod: Pod,
) -> None:
    """Catches: a dropped connection never reaching the probes (a pod that cannot reach LiveKit
    reported ready, or live forever), a short reconnect failing liveness, or a pod back on
    LiveKit still failing readiness. The grace past which `/` fails is `health_tests.rs`'s, on a
    paused clock."""
    await pod.registered()
    await pod.until_health("/ready", 200, STARTUP_SECONDS)

    pod.livekit.hold_registration()
    await pod.livekit.disconnect()
    reconnecting = await pod.until_health("/ready", 503, STARTUP_SECONDS)
    assert reconnecting.text == "reconnecting"
    assert (await pod.health.get("/")).status_code == 200, "a short reconnect stays live"

    pod.livekit.answer_registration()
    await pod.livekit.until_registered()
    assert (await pod.until_health("/ready", 200, STARTUP_SECONDS)).text == "OK"
    assert (await pod.health.get("/")).text == "OK"


async def test_a_pod_that_reconnects_mid_call_reports_the_call_again(pod: Pod) -> None:
    """Catches: a call lost to LiveKit after the pod's connection drops (the server re-dispatches
    it or ends it), or reported as a bare id instead of the job it was assigned, under the
    identity the pod accepted it with."""
    await pod.registered()
    await pod.in_a_call()
    identity = pod.livekit.answers[JOB.id].participant_identity

    await pod.livekit.disconnect()
    await until(lambda: pod.livekit.migrations, "the call reported again", STARTUP_SECONDS)

    [migration] = pod.livekit.migrations
    [job] = migration.jobs
    assert (list(migration.job_ids), job.id, job.room.name) == ([JOB.id], JOB.id, JOB.room.name)
    assert job.state.participant_identity == identity
    assert (await pod.health.get("/worker")).json()["active_jobs"] == 1, "the call kept running"


async def test_the_watcher_reads_the_worker_and_its_calls(pod: Pod) -> None:
    """Catches: `/worker` keys `encoding/json` cannot decode. The watcher then reads no agent, no
    deployment support and zero active jobs, and may scale down a pod in a call."""
    await pod.registered()
    idle = (await pod.health.get("/worker")).json()
    assert set(idle) <= WORKER_INFO_KEYS, f"keys the watcher drops: {set(idle) - WORKER_INFO_KEYS}"
    assert idle["worker_type"] == "JT_ROOM"
    assert idle["agent_name"] == "cloud-agent"
    assert idle["sdk_version"] == __version__
    assert idle["protocol_version"] >= 1  # SupportsAgentDeployment
    assert idle.get("active_jobs", 0) == 0

    await pod.in_a_call()
    assert (await pod.health.get("/worker")).json()["active_jobs"] == 1


async def test_a_call_runs_with_the_pod_identity(pod: Pod) -> None:
    """Catches: the worker token, deployment, agent id or region not reaching the job process,
    inference requests without `X-LiveKit-Worker-Token` (the gateway authorizes the pod by it),
    the in-cluster gateway bypassed, or the job not knowing it is hosted."""
    await pod.registered()
    seen = await pod.in_a_call()

    env = seen["env"]
    identity = ("LIVEKIT_WORKER_TOKEN", "LIVEKIT_AGENT_DEPLOYMENT", "LIVEKIT_AGENT_ID")
    assert {name: env.get(name) for name in (*identity, "LIVEKIT_REGION_NAME")} == {
        "LIVEKIT_WORKER_TOKEN": WORKER_TOKEN,
        "LIVEKIT_AGENT_DEPLOYMENT": "staging",
        "LIVEKIT_AGENT_ID": "CA_pod",
        "LIVEKIT_REGION_NAME": REGION,
    }
    headers = seen["inference_headers"]
    assert headers["X-LiveKit-Worker-Token"] == WORKER_TOKEN
    assert (headers["X-LiveKit-Room-ID"], headers["X-LiveKit-Job-ID"]) == ("RM_pod", "AJ_pod")
    assert seen["inference_url"] == IN_CLUSTER_INFERENCE
    assert seen["hosted"] is True


async def test_the_pod_ignores_its_own_load_policy(pod: Pod) -> None:
    """Catches: a custom load function or threshold honoured on Cloud, where the platform sizes
    the fleet: this pod's own policy (always 1.0, full at 0.0) would keep it full forever."""
    await pod.registered()
    await until(lambda: pod.livekit.available, "the pod reported available", STARTUP_SECONDS)

    assert [status.load for status in pod.livekit.statuses if status.load == 1.0] == []


async def test_sigterm_drains_the_call_then_exits(pod: Pod) -> None:
    """Catches: a terminating pod still offered jobs, liveness failing mid-drain (kubelet then
    kills the call), the call cut before it ends, or the pod lingering after it."""
    await pod.registered()
    await pod.in_a_call()

    await pod.draining()
    assert pod.livekit.statuses[-1].job_count == 1
    assert not (await pod.livekit.offer(LATE_JOB)).available, "a draining pod takes no call"
    assert pod.child.process.returncode is None, "the call holds the pod"
    assert (await pod.health.get("/")).status_code == 200, "live while draining"
    assert pod.livekit.job_statuses == {JOB.id: [agent.JobStatus.JS_RUNNING]}

    pod.hang_up()
    assert await pod.exit_code() == 0
    assert pod.livekit.job_statuses[JOB.id][-1] == agent.JobStatus.JS_SUCCESS


@pytest.mark.parametrize(
    "pod",
    [
        (
            f"--drain-timeout={DRAIN_TIMEOUT}",
            f"--shutdown-process-timeout={PROCESS_EXIT_TIMEOUT}",
        )
    ],
    indirect=True,
)
async def test_a_call_that_outlives_the_drain_is_ended_and_the_pod_exits(pod: Pod) -> None:
    """Catches: a shutdown that waits forever on a call that never ends (the pod sits in
    Terminating until SIGKILL), or its job process left running. As in 1.x, the connection closes
    before the job ends: LiveKit fails the job of a worker that left."""
    await pod.registered()
    seen = await pod.in_a_call()

    await pod.draining()

    assert await pod.exit_code(within=DRAIN_TIMEOUT + PROCESS_EXIT_TIMEOUT + SHUTDOWN_SLACK) == 0
    assert await asyncio.to_thread(gone, seen["pid"], 5), "the call's job process outlived the pod"


async def test_a_second_signal_ends_the_drain_at_once(pod: Pod) -> None:
    """Catches: a second SIGTERM or Ctrl+C waiting out the drain (1.x and the `lk agent` launcher
    exit at once on it), or the forced exit orphaning the call's job process."""
    await pod.registered()
    seen = await pod.in_a_call()
    await pod.draining()

    pod.child.process.send_signal(signal.SIGINT)

    assert await pod.exit_code(within=5) == 1
    assert await asyncio.to_thread(gone, seen["pid"], 5), "the call's job process outlived the pod"
    assert "exiting forcefully" in pod.child.log.read_text()


async def test_sigterm_while_disconnected_exits(pod: Pod) -> None:
    """Catches: a shutdown that waits on a LiveKit connection that is down, so the pod sits in
    Terminating until SIGKILL."""
    await pod.registered()
    pod.livekit.hold_registration()
    await pod.livekit.disconnect()
    await until(lambda: len(pod.livekit.connections) == 2, "a reconnection", STARTUP_SECONDS)

    pod.child.process.send_signal(signal.SIGTERM)

    assert await pod.exit_code(within=SHUTDOWN_SLACK) == 0


async def test_every_log_line_carries_the_region(pod: Pod) -> None:
    """Catches: `LIVEKIT_REGION_NAME` missing from the JSON lines Cloud collects, from the worker
    or from the job process."""
    await pod.registered()
    await pod.in_a_call()
    pod.hang_up()
    pod.child.process.send_signal(signal.SIGTERM)
    assert await pod.exit_code() == 0

    lines = [json.loads(line) for line in pod.child.log.read_text().splitlines()]
    assert lines
    assert [line for line in lines if line.get("region") != REGION] == []
