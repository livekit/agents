"""A job's Python callbacks driven by the native job runner: shutdown order, the job's
credentials and headers, failures it must report, and the sessions it ends."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import sys
import types
from collections.abc import AsyncIterator
from pathlib import Path

import jwt
import pytest
from aiohttp import web

from livekit import api
from livekit.agents import Agent, JobContext, Plugin, VoiceSession, _core
from livekit.agents.beta import EndCallTool, ToolCalledEvent
from livekit.agents.http import client as http
from livekit.agents.inference import get_inference_headers
from livekit.agents.ipc import load_preload_modules
from livekit.agents.job import get_job_context
from livekit.agents.voice import CloseEvent
from livekit.protocol import agent, models
from tests_new.fakes import FakeLLM, tool_call

from .conftest import RunJob

KEY, SECRET = "server-key", "server-secret-at-least-32-bytes-long"


@pytest.fixture
async def livekit_api() -> AsyncIterator[tuple[str, list[tuple[str, str]]]]:
    """A LiveKit server API answering `DeleteRoom` with the server's key; yields its URL and
    each (signing key, room) it was asked to delete."""
    deleted: list[tuple[str, str]] = []

    async def delete(request: web.Request) -> web.Response:
        token = request.headers["Authorization"].removeprefix("Bearer ")
        claims = jwt.decode(token, SECRET, algorithms=["HS256"], issuer=KEY)
        deleted.append((claims["iss"], api.DeleteRoomRequest.FromString(await request.read()).room))
        return web.Response(
            body=api.DeleteRoomResponse().SerializeToString(), content_type="application/protobuf"
        )

    async def headers(request: web.Request) -> web.Response:
        return web.json_response(dict(request.headers))

    app = web.Application()
    app.router.add_post("/twirp/livekit.RoomService/DeleteRoom", delete)
    app.router.add_get("/headers", headers)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", 0).start()
    yield f"http://127.0.0.1:{runner.addresses[0][1]}", deleted
    await runner.cleanup()


@pytest.mark.parametrize("entrypoint_raises", [False, True])
async def test_shutdown_runs_session_end_then_the_shutdown_callbacks(
    run_job: RunJob, entrypoint_raises: bool
) -> None:
    """`on_session_end` runs first, then every shutdown callback (with or without the
    reason), even when the entrypoint raised; the job's room has its assigned name though it
    never connected.

    https://github.com/livekit/agents/issues/1512 https://github.com/livekit/agents/pull/5741
    https://github.com/livekit/agents/pull/716
    """
    seen: list[str] = []

    async def close() -> None:
        seen.append("close")

    async def close_with_reason(reason: str) -> None:
        seen.append("close with a reason" if reason else "close with an empty reason")

    async def entrypoint(ctx: JobContext) -> None:
        ctx.add_shutdown_callback(close)
        ctx.add_shutdown_callback(close_with_reason)
        if entrypoint_raises:
            raise RuntimeError("entrypoint failed")
        ctx.shutdown("finished")

    async def session_end(ctx: JobContext) -> None:
        seen.append(f"session end in {ctx.room.name}")

    job = agent.Job(id="AJ_shutdown", room=models.Room(name="assigned-room"))
    await run_job(entrypoint, job, session_end=session_end, fake_job=False)

    assert seen[0] == "session end in assigned-room"
    assert sorted(seen[1:]) == ["close", "close with a reason"]


async def test_the_job_api_signs_with_the_servers_credentials_not_the_environment(
    run_job: RunJob,
    livekit_api: tuple[str, list[tuple[str, str]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`ctx.api` calls the server's URL signed with its key, over different LIVEKIT_*
    variables in the environment. https://github.com/livekit/agents/issues/2651"""
    url, deleted = livekit_api
    monkeypatch.setenv("LIVEKIT_URL", "http://127.0.0.1:1")
    monkeypatch.setenv("LIVEKIT_API_KEY", "environment-key")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "environment-secret")

    async def entrypoint(ctx: JobContext) -> None:
        await ctx.api.room.delete_room(api.DeleteRoomRequest(room="assigned"))
        ctx.shutdown("done")

    server_env = {"LIVEKIT_URL": url, "LIVEKIT_API_KEY": KEY, "LIVEKIT_API_SECRET": SECRET}
    await run_job(entrypoint, agent.Job(id="AJ_credentials"), env=server_env, fake_job=False)

    assert deleted == [(KEY, "assigned")]


async def test_a_console_job_connects_and_finds_its_user_without_a_server(run_job: RunJob) -> None:
    """A console job's concurrent `connect()` calls both complete and `wait_for_participant`
    returns the console's user, with no LiveKit server behind the job."""
    users: list[str] = []

    async def entrypoint(ctx: JobContext) -> None:
        await asyncio.gather(ctx.connect(), ctx.connect())
        users.append((await ctx.wait_for_participant()).identity)
        ctx.shutdown("done")

    await run_job(entrypoint, agent.Job(id="AJ_console", room=models.Room(name="console")))

    assert len(users) == 1 and users[0]


async def test_deleting_a_console_jobs_room_calls_no_server(
    run_job: RunJob, livekit_api: tuple[str, list[tuple[str, str]]]
) -> None:
    """`ctx.delete_room()` on a console job answers without calling the LiveKit server."""
    url, deleted = livekit_api
    answers: list[object] = []

    async def entrypoint(ctx: JobContext) -> None:
        answers.append(await ctx.delete_room())
        ctx.shutdown("done")

    server_env = {"LIVEKIT_URL": url, "LIVEKIT_API_KEY": KEY, "LIVEKIT_API_SECRET": SECRET}
    await run_job(
        entrypoint, agent.Job(id="AJ_console", room=models.Room(name="console")), env=server_env
    )

    assert [type(answer) for answer in answers] == [api.DeleteRoomResponse]
    assert deleted == []


@pytest.mark.parametrize("fake_job", [False, True])
async def test_inference_headers_name_the_room_and_job_but_no_unconnected_participant(
    run_job: RunJob, livekit_api: tuple[str, list[tuple[str, str]]], fake_job: bool
) -> None:
    """A job's inference headers carry its room and job ids, and no agent participant id
    before it joined a room. https://github.com/livekit/agents/pull/5943"""
    url, _ = livekit_api
    received: list[object] = []

    async def entrypoint(ctx: JobContext) -> None:
        async with http.http_session().get(f"{url}/headers", headers=get_inference_headers()) as r:
            received.append(await r.json())
        ctx.shutdown("sent")

    job = agent.Job(id="AJ_headers", room=models.Room(sid="RM_headers"))
    await run_job(entrypoint, job, fake_job=fake_job)

    [headers] = received
    assert isinstance(headers, dict)
    assert (headers["X-LiveKit-Room-ID"], headers["X-LiveKit-Job-ID"]) == (
        "RM_headers",
        "AJ_headers",
    )
    assert "X-LiveKit-Agent-ID" not in headers


async def test_an_entrypoint_failure_is_logged_with_its_exception(
    run_job: RunJob, caplog: pytest.LogCaptureFixture
) -> None:
    """An exception escaping the entrypoint is logged with its message.
    https://github.com/livekit/agents/pull/244"""

    async def entrypoint(ctx: JobContext) -> None:
        raise RuntimeError("unhandled-entrypoint-regression")

    await run_job(entrypoint, agent.Job(id="AJ_unhandled"), fake_job=False)

    assert "unhandled exception while running the job task" in caplog.text
    assert "unhandled-entrypoint-regression" in caplog.text


async def test_an_unawaited_job_api_failure_is_logged(
    run_job: RunJob, caplog: pytest.LogCaptureFixture, no_livekit_credentials: None
) -> None:
    """A `delete_room` the entrypoint never awaits still reports its failure, before garbage
    collection would. https://github.com/livekit/agents/pull/5961"""
    pending: list[asyncio.Future[None]] = []

    async def entrypoint(ctx: JobContext) -> None:
        pending.append(ctx.delete_room("assigned-room"))
        while not pending[0].done():
            await asyncio.sleep(0)
        ctx.shutdown("done")

    await run_job(entrypoint, agent.Job(id="AJ_detached"), fake_job=False)

    assert "unknown error while deleting room" in caplog.text
    assert "url must be set" in caplog.text
    pending[0].exception()


async def test_session_events_shutdown_callbacks_and_session_end_see_the_job(
    run_job: RunJob, no_livekit_credentials: None
) -> None:
    """A session event handler, a shutdown callback and `on_session_end` each run in the job's
    context, so `get_job_context()` (and `delete_room_on_close`, which needs it) reach the job."""
    seen: list[tuple[str, str | None]] = []

    def job_id() -> str | None:
        ctx = get_job_context(required=False)
        return ctx.job.id if ctx is not None else None

    async def entrypoint(ctx: JobContext) -> None:
        async def shutdown(reason: str) -> None:
            seen.append(("shutdown", job_id()))

        ctx.add_shutdown_callback(shutdown)
        session = VoiceSession(llm=FakeLLM())
        session.on("close", lambda _: seen.append(("session event", job_id())))
        await session.start(Agent(instructions="Help."))
        await session.aclose()
        ctx.shutdown("done")

    async def session_end(ctx: JobContext) -> None:
        seen.append(("session end", job_id()))

    await run_job(entrypoint, agent.Job(id="AJ_context"), session_end=session_end)

    assert sorted(seen) == [
        ("session end", "AJ_context"),
        ("session event", "AJ_context"),
        ("shutdown", "AJ_context"),
    ]


async def test_an_end_call_tool_closes_the_session_and_shuts_the_job_down(
    run_job: RunJob, caplog: pytest.LogCaptureFixture, no_livekit_credentials: None
) -> None:
    """`EndCallTool` called inside the job's own call ends the reply, the session and the job,
    in that order, without deadlocking; the tool sees the job's context.

    https://github.com/livekit/agents/issues/3391 https://github.com/livekit/agents/issues/3969
    """
    seen: list[tuple[str, str | None]] = []

    async def called(event: ToolCalledEvent) -> None:
        seen.append(("tool", get_job_context().job.id))

    async def entrypoint(ctx: JobContext) -> None:
        async def shutdown(reason: str) -> None:
            seen.append(("shutdown", reason))

        ctx.add_shutdown_callback(shutdown)
        closed = asyncio.Event()
        session = VoiceSession(llm=FakeLLM([tool_call("end_call"), "goodbye"]))

        @session.on("close")
        def on_close(event: CloseEvent) -> None:
            seen.append(("close", event.reason.value))
            closed.set()

        await session.start(Agent(instructions="Help.", tools=[EndCallTool(on_tool_called=called)]))
        await session.generate_reply(user_input="bye")
        seen.append(("reply", session.history.messages()[-1].text_content))
        await closed.wait()

    await run_job(entrypoint, agent.Job(id="AJ_end_call"))

    assert seen == [
        ("tool", "AJ_end_call"),
        ("reply", "goodbye"),
        ("close", "user_initiated"),
        ("shutdown", "user_initiated"),
    ]
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]


@pytest.mark.xfail(
    strict=True,
    reason="X8039 reproduces: the job's shutdown closes its session as user_initiated",
)
async def test_job_shutdown_closes_the_jobs_session_with_the_job_shutdown_reason(
    run_job: RunJob, no_livekit_credentials: None
) -> None:
    """A session the job's shutdown ends reports `job_shutdown` (docs/voice-session.md, A job's
    sessions)."""
    reasons: list[str] = []

    async def entrypoint(ctx: JobContext) -> None:
        session = VoiceSession(llm=FakeLLM())
        session.on("close", lambda ev: reasons.append(ev.reason.value))
        await session.start(Agent(instructions="Help."))
        ctx.shutdown("done")

    await run_job(entrypoint, agent.Job(id="AJ_job_shutdown"))

    assert reasons == ["job_shutdown"]


@pytest.mark.xfail(
    strict=True,
    reason="https://github.com/livekit/agents/issues/7486 reproduces: add/remove of lk outcome tags break the outcome",
)
def test_reserved_outcome_tags_cannot_disagree_with_the_outcome() -> None:
    """The reported outcome always matches its tags and reason.
    https://github.com/livekit/agents/issues/7486"""
    tagger = _core.Tagger()  # what `JobContext.tagger` is
    tagger.fail(reason="booking failed")
    with contextlib.suppress(ValueError):
        tagger.add("lk.success")

    outcome_tags = tagger.tags & {"lk.success", "lk.fail"}
    assert len(outcome_tags) == 1
    assert {"lk.success": "success", "lk.fail": "fail"}[next(iter(outcome_tags))] == tagger.outcome


def test_a_plugin_listing_a_name_it_never_defines_still_lets_jobs_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A job process preloads every registered plugin's exports before it is ready. A published
    plugin whose `__all__` names something it never defines (livekit-plugins-ai-coustics 0.3.5's
    `LiveKitCloud`) must not stop every job process from starting; its real exports still load."""
    package = tmp_path / "stale_plugin"
    package.mkdir()
    (package / "__init__.py").write_text(
        "import importlib\n"
        "__all__ = ['Enhancer', 'LiveKitCloud']\n"
        "def __getattr__(name):\n"
        "    if name != 'Enhancer':\n"
        "        raise AttributeError(name)\n"
        "    return importlib.import_module('.enhancer', __name__).Enhancer\n"
    )
    (package / "enhancer.py").write_text("class Enhancer: ...\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(
        Plugin, "registered_plugins", [types.SimpleNamespace(package="stale_plugin")]
    )
    try:
        load_preload_modules([])
        assert "stale_plugin.enhancer" in sys.modules
    finally:
        for name in [m for m in sys.modules if m.split(".")[0] == "stale_plugin"]:
            del sys.modules[name]
