import asyncio
from base64 import b64decode
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock
from xml.etree import ElementTree

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from livekit import rtc
from livekit.agents.beta.workflows import TwilioConnectorWarmTransferTask, warm_transfer
from livekit.agents.llm import ToolError

pytestmark = [pytest.mark.unit, pytest.mark.no_concurrent]

CALLER_NUMBER = "+15555550101"
TWILIO_NUMBER = "+15555550102"
HUMAN_NUMBER = "+15555550103"
CALL_TOKEN = "opaque-call-token+with/encoding=="
CONNECT_URL = "wss://connector.example.test/stream?one=1&two=2"
CALLS_PATH = "/2010-04-01/Accounts/AC_test_account/Calls"


@pytest.fixture(autouse=True)
def mock_background_audio(monkeypatch: pytest.MonkeyPatch) -> None:
    # these tests exercise call origination only; no room or audio mixer is started
    monkeypatch.setattr(warm_transfer, "BackgroundAudioPlayer", Mock())


@pytest.fixture
async def twilio_http(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[AsyncMock]:
    handler = AsyncMock(
        side_effect=lambda **kwargs: web.json_response({"sid": "CA_test_transfer"}, status=201)
    )

    async def handle(request: web.Request) -> web.Response:
        return await handler(
            path=request.path,
            data=dict(await request.post()),
            auth=request.headers.get("Authorization"),
            content_type=request.content_type,
        )

    app = web.Application()
    app.router.add_post("/{path:.*}", handle)
    async with TestServer(app) as server, aiohttp.ClientSession() as session:
        monkeypatch.setattr(warm_transfer, "_TWILIO_API_BASE", str(server.make_url("/2010-04-01")))
        monkeypatch.setattr(warm_transfer.utils.http_context, "http_session", lambda: session)
        yield handler


@pytest.fixture
def connector(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    connect = AsyncMock(return_value=SimpleNamespace(connect_url=CONNECT_URL))
    ctx = SimpleNamespace(
        api=SimpleNamespace(connector=SimpleNamespace(connect_twilio_call=connect))
    )
    monkeypatch.setattr(warm_transfer, "get_job_context", lambda: ctx)
    return connect


def build_task(**options: str) -> TwilioConnectorWarmTransferTask:
    return TwilioConnectorWarmTransferTask(
        HUMAN_NUMBER,
        twilio_from_number=TWILIO_NUMBER,
        twilio_account_sid="AC_test_account",
        twilio_auth_token="test_auth_token",
        **options,
    )


def twilio_error(status: int, code: int) -> web.Response:
    return web.json_response({"code": code}, status=status)


async def dial(task: TwilioConnectorWarmTransferTask, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(task, "_wait_for_human_agent", AsyncMock())
    try:
        await task._originate_human_agent(
            room_name="consult-room", identity="human", room=Mock(spec=rtc.Room)
        )
    finally:
        await asyncio.gather(*task._twilio_tasks.tasks, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("options", "expected_from", "expect_token"),
    [
        (
            {"twilio_call_token": CALL_TOKEN, "original_caller_number": CALLER_NUMBER},
            CALLER_NUMBER,
            True,
        ),
        ({}, TWILIO_NUMBER, False),
        ({"original_caller_number": CALLER_NUMBER}, TWILIO_NUMBER, False),
        ({"twilio_call_token": "", "original_caller_number": CALLER_NUMBER}, TWILIO_NUMBER, False),
    ],
    ids=["token-and-original", "no-token", "original-without-token", "empty-token"],
)
async def test_dial_uses_expected_caller_id(
    monkeypatch: pytest.MonkeyPatch,
    twilio_http: AsyncMock,
    connector: AsyncMock,
    options: dict[str, str],
    expected_from: str,
    expect_token: bool,
) -> None:
    await dial(build_task(**options), monkeypatch)

    connector.assert_awaited_once()
    twilio_http.assert_awaited_once()
    request = twilio_http.call_args.kwargs
    assert request["path"] == f"{CALLS_PATH}.json"
    assert request["content_type"] == "application/x-www-form-urlencoded"
    scheme, auth = request["auth"].split(" ")
    assert scheme == "Basic"
    assert b64decode(auth).decode() == "AC_test_account:test_auth_token"
    data = request["data"]
    assert data["To"] == HUMAN_NUMBER
    assert data["From"] == expected_from
    assert ("CallToken" in data) is expect_token
    if expect_token:
        assert data["CallToken"] == CALL_TOKEN
    stream = ElementTree.fromstring(data["Twiml"]).find("./Connect/Stream")
    assert stream is not None and stream.attrib == {"url": CONNECT_URL}
    assert CALL_TOKEN not in data["Twiml"]
    assert CALL_TOKEN not in str(connector.call_args)


@pytest.mark.asyncio
@pytest.mark.parametrize("code", [21210, 21212], ids=["unverified-from", "invalid-from"])
async def test_rejected_caller_id_retries_once_without_token(
    monkeypatch: pytest.MonkeyPatch, twilio_http: AsyncMock, connector: AsyncMock, code: int
) -> None:
    twilio_http.side_effect = [
        twilio_error(400, code),
        web.json_response({"sid": "CA_retry"}, status=201),
    ]
    await dial(
        build_task(twilio_call_token=CALL_TOKEN, original_caller_number=CALLER_NUMBER), monkeypatch
    )

    assert twilio_http.await_count == 2
    first, second = (call.kwargs["data"] for call in twilio_http.call_args_list)
    assert first["From"] == CALLER_NUMBER
    assert first["CallToken"] == CALL_TOKEN
    assert second["From"] == TWILIO_NUMBER
    assert "CallToken" not in second


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "code"),
    [(400, 21211), (403, 21210), (500, 21210), (429, 20429), (403, 21212), (500, 21212)],
)
async def test_other_twilio_errors_propagate_without_retry(
    monkeypatch: pytest.MonkeyPatch,
    twilio_http: AsyncMock,
    connector: AsyncMock,
    status: int,
    code: int,
) -> None:
    twilio_http.side_effect = None
    twilio_http.return_value = twilio_error(status, code)
    with pytest.raises(RuntimeError, match=rf"HTTP {status}, code {code}"):
        await dial(
            build_task(twilio_call_token=CALL_TOKEN, original_caller_number=CALLER_NUMBER),
            monkeypatch,
        )
    twilio_http.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("body", ['{"code": "21210"}', "[]", "null", "invalid JSON"])
async def test_unrecognized_error_response_does_not_trigger_fallback(
    monkeypatch: pytest.MonkeyPatch, twilio_http: AsyncMock, connector: AsyncMock, body: str
) -> None:
    twilio_http.side_effect = None
    twilio_http.return_value = web.Response(status=400, text=body)
    with pytest.raises(RuntimeError, match="HTTP 400, code unknown"):
        await dial(
            build_task(twilio_call_token=CALL_TOKEN, original_caller_number=CALLER_NUMBER),
            monkeypatch,
        )
    twilio_http.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("body", ["{}", '{"sid": null}', "invalid JSON"])
async def test_creation_without_sid_is_not_retried(
    monkeypatch: pytest.MonkeyPatch, twilio_http: AsyncMock, connector: AsyncMock, body: str
) -> None:
    twilio_http.side_effect = None
    twilio_http.return_value = web.Response(status=201, text=body)
    with pytest.raises(RuntimeError, match="returned no call SID"):
        await dial(build_task(), monkeypatch)
    twilio_http.assert_awaited_once()


def test_call_token_requires_original_caller_number() -> None:
    with pytest.raises(ValueError, match="original_caller_number"):
        build_task(twilio_call_token=CALL_TOKEN)


@pytest.mark.asyncio
@pytest.mark.parametrize("code", [21210, 21212])
async def test_failed_fallback_is_not_retried(
    monkeypatch: pytest.MonkeyPatch, twilio_http: AsyncMock, connector: AsyncMock, code: int
) -> None:
    twilio_http.side_effect = [twilio_error(400, code), twilio_error(400, code)]
    with pytest.raises(RuntimeError, match=rf"HTTP 400, code {code}"):
        await dial(
            build_task(twilio_call_token=CALL_TOKEN, original_caller_number=CALLER_NUMBER),
            monkeypatch,
        )
    assert twilio_http.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("token", [None, ""])
@pytest.mark.parametrize("code", [21210, 21212])
async def test_business_number_rejection_is_not_retried(
    monkeypatch: pytest.MonkeyPatch,
    twilio_http: AsyncMock,
    connector: AsyncMock,
    token: str | None,
    code: int,
) -> None:
    twilio_http.side_effect = None
    twilio_http.return_value = twilio_error(400, code)
    with pytest.raises(RuntimeError, match=rf"HTTP 400, code {code}"):
        await dial(
            build_task(**({"twilio_call_token": token} if token is not None else {})), monkeypatch
        )
    twilio_http.assert_awaited_once()


@pytest.mark.asyncio
async def test_ambiguous_transport_failure_is_not_retried(
    monkeypatch: pytest.MonkeyPatch, twilio_http: AsyncMock, connector: AsyncMock
) -> None:
    task = build_task(twilio_call_token=CALL_TOKEN, original_caller_number=CALLER_NUMBER)
    request = AsyncMock(side_effect=TimeoutError("request timed out"))
    monkeypatch.setattr(task, "_twilio_request", request)
    with pytest.raises(TimeoutError):
        await dial(task, monkeypatch)
    request.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("fallback", [False, True])
async def test_unanswered_call_cancels_the_created_sid(
    monkeypatch: pytest.MonkeyPatch, twilio_http: AsyncMock, connector: AsyncMock, fallback: bool
) -> None:
    twilio_http.side_effect = ([twilio_error(400, 21210)] if fallback else []) + [
        web.json_response({"sid": "CA_created"}, status=201),
        web.json_response({}),
    ]
    task = build_task(twilio_call_token=CALL_TOKEN, original_caller_number=CALLER_NUMBER)
    monkeypatch.setattr(
        task, "_wait_for_human_agent", AsyncMock(side_effect=ToolError("no answer"))
    )
    with pytest.raises(ToolError, match="no answer"):
        await task._originate_human_agent(room_name="consult", identity="human", room=Mock())
    await asyncio.gather(*task._twilio_tasks.tasks)
    request = twilio_http.call_args.kwargs
    assert request["path"] == f"{CALLS_PATH}/CA_created.json"
    assert request["data"] == {"Status": "canceled"}
    assert twilio_http.await_count == (3 if fallback else 2)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["tokenless", "forwarded", "fallback"])
@pytest.mark.parametrize("rejected", [False, True])
async def test_cancellation_returns_before_late_creation_and_retains_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    twilio_http: AsyncMock,
    connector: AsyncMock,
    mode: str,
    rejected: bool,
) -> None:
    task = build_task(
        **(
            {"twilio_call_token": CALL_TOKEN, "original_caller_number": CALLER_NUMBER}
            if mode != "tokenless"
            else {}
        )
    )
    wait = AsyncMock()
    monkeypatch.setattr(task, "_wait_for_human_agent", wait)
    started = asyncio.Event()
    release = asyncio.Event()
    attempts = 0

    async def respond(*, path: str, **kwargs: Any) -> web.Response:
        nonlocal attempts
        if path != f"{CALLS_PATH}.json":
            assert path == f"{CALLS_PATH}/CA_late.json"
            assert kwargs["data"] == {"Status": "canceled"}
            return web.json_response({})
        attempts += 1
        if mode == "fallback" and attempts == 1:
            return twilio_error(400, 21210)
        started.set()
        await release.wait()
        if rejected:
            return twilio_error(400, 21210)
        return web.json_response({"sid": "CA_late"}, status=201)

    twilio_http.side_effect = respond
    pending = asyncio.create_task(
        task._originate_human_agent(room_name="consult", identity="human", room=Mock())
    )
    try:
        await asyncio.wait_for(started.wait(), timeout=5)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(asyncio.shield(pending), timeout=0.5)
        assert not release.is_set()
        assert task._twilio_tasks.tasks
    finally:
        release.set()
        await asyncio.gather(pending, return_exceptions=True)
        await asyncio.gather(*task._twilio_tasks.tasks, return_exceptions=True)
    assert not task._twilio_tasks.tasks
    assert attempts == (2 if mode == "fallback" else 1)
    assert twilio_http.await_count == attempts + (0 if rejected else 1)
    wait.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_while_waiting", [False, True])
async def test_timeout_or_cancellation_does_not_wait_for_pending_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    twilio_http: AsyncMock,
    connector: AsyncMock,
    cancel_while_waiting: bool,
) -> None:
    task = build_task(twilio_call_token=CALL_TOKEN, original_caller_number=CALLER_NUMBER)
    error = asyncio.CancelledError() if cancel_while_waiting else ToolError("no answer")
    monkeypatch.setattr(task, "_wait_for_human_agent", AsyncMock(side_effect=error))
    started = asyncio.Event()
    release = asyncio.Event()

    async def respond(*, path: str, **kwargs: Any) -> web.Response:
        if path == f"{CALLS_PATH}.json":
            return web.json_response({"sid": "CA_created"}, status=201)
        started.set()
        await release.wait()
        return web.json_response({})

    twilio_http.side_effect = respond
    pending = asyncio.create_task(
        task._originate_human_agent(room_name="consult", identity="human", room=Mock())
    )
    try:
        await asyncio.wait_for(started.wait(), timeout=5)
        with pytest.raises(asyncio.CancelledError if cancel_while_waiting else ToolError):
            await asyncio.wait_for(asyncio.shield(pending), timeout=0.5)
        assert not release.is_set()
        assert task._twilio_tasks.tasks
    finally:
        release.set()
        await asyncio.gather(pending, return_exceptions=True)
        await asyncio.gather(*task._twilio_tasks.tasks)
    assert not task._twilio_tasks.tasks


@pytest.mark.asyncio
@pytest.mark.parametrize("call_state", ["queued", "ringing", "in-progress"])
async def test_ends_call_on_timeout(
    monkeypatch: pytest.MonkeyPatch, twilio_http: AsyncMock, connector: AsyncMock, call_state: str
) -> None:
    initial_state = call_state

    async def respond(*, path: str, data: dict[str, str], **kwargs: Any) -> web.Response:
        nonlocal call_state
        if path != f"{CALLS_PATH}.json":
            assert path == f"{CALLS_PATH}/CA_test_transfer.json"
            status = data["Status"]
            if status == "canceled" and call_state == "in-progress":
                return twilio_error(400, 21220)
            assert status == ("completed" if call_state == "in-progress" else "canceled")
            call_state = status
        return web.json_response({"sid": "CA_test_transfer", "status": call_state})

    twilio_http.side_effect = respond
    task = build_task(twilio_call_token=CALL_TOKEN, original_caller_number=CALLER_NUMBER)
    monkeypatch.setattr(
        task, "_wait_for_human_agent", AsyncMock(side_effect=ToolError("no answer"))
    )
    with pytest.raises(ToolError, match="no answer"):
        await task._originate_human_agent(room_name="consult", identity="human", room=Mock())
    await asyncio.gather(*task._twilio_tasks.tasks)

    assert call_state == ("completed" if initial_state == "in-progress" else "canceled")
    assert twilio_http.await_count == (3 if initial_state == "in-progress" else 2)


@pytest.mark.asyncio
@pytest.mark.parametrize(("status", "code"), [(403, 21220), (400, 21211), (500, 21220)])
async def test_cleanup_does_not_retry_unrelated_errors(
    monkeypatch: pytest.MonkeyPatch,
    twilio_http: AsyncMock,
    connector: AsyncMock,
    status: int,
    code: int,
) -> None:
    twilio_http.side_effect = [
        web.json_response({"sid": "CA_test_transfer"}, status=201),
        twilio_error(status, code),
    ]
    task = build_task()
    monkeypatch.setattr(
        task, "_wait_for_human_agent", AsyncMock(side_effect=ToolError("no answer"))
    )
    with pytest.raises(ToolError, match="no answer"):
        await task._originate_human_agent(room_name="consult", identity="human", room=Mock())
    await asyncio.gather(*task._twilio_tasks.tasks)
    assert twilio_http.await_count == 2
    assert twilio_http.call_args.kwargs["data"] == {"Status": "canceled"}


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["connect", "start", "originate", None])
@pytest.mark.parametrize("cancelled", [False, True])
async def test_consultation_setup_releases_resources_on_failure(
    monkeypatch: pytest.MonkeyPatch, stage: str | None, cancelled: bool
) -> None:
    task = build_task()
    task._caller_room = Mock(name="caller-room", local_participant=Mock(identity="agent"))
    task._caller_room.name = "caller-room"
    ctx = SimpleNamespace(_info=SimpleNamespace(url="wss://example.test"))
    monkeypatch.setattr(warm_transfer, "get_job_context", lambda: ctx)
    monkeypatch.setattr(warm_transfer.api, "AccessToken", Mock())
    room = Mock(connect=AsyncMock(), disconnect=AsyncMock())
    monkeypatch.setattr(warm_transfer.rtc, "Room", Mock(return_value=room))
    parent = SimpleNamespace(vad=None, llm=None, stt=None, tts=None, turn_detection=None)
    monkeypatch.setattr(warm_transfer.AgentTask, "session", property(lambda _: parent))
    session = Mock(start=AsyncMock())
    monkeypatch.setattr(warm_transfer, "AgentSession", Mock(return_value=session))
    monkeypatch.setattr(warm_transfer, "Agent", Mock())
    originate = AsyncMock()
    monkeypatch.setattr(task, "_originate_human_agent", originate)

    if stage is None:
        assert await task._dial_human_agent() is session
        session.shutdown.assert_not_called()
        room.disconnect.assert_not_awaited()
        return

    error = asyncio.CancelledError() if cancelled else RuntimeError("setup failed")
    {"connect": room.connect, "start": session.start, "originate": originate}[
        stage
    ].side_effect = error
    with pytest.raises(type(error)) as caught:
        await task._dial_human_agent()

    assert caught.value is error
    session.shutdown.assert_called_once_with(drain=False)
    room.off.assert_called_once_with("disconnected", task._on_human_agent_room_close)
    room.disconnect.assert_awaited_once()
