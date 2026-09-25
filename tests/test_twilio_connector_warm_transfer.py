import asyncio
import json
import logging
import sys
import threading
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock, create_autospec
from urllib.parse import parse_qs
from xml.etree import ElementTree

import pytest
import requests

from livekit import rtc
from livekit.agents.beta.workflows import TwilioConnectorWarmTransferTask, warm_transfer
from livekit.agents.llm import ToolError

pytestmark = [pytest.mark.unit, pytest.mark.no_concurrent]

CALLER_NUMBER = "+15555550101"
TWILIO_NUMBER = "+15555550102"
HUMAN_NUMBER = "+15555550103"
CALL_TOKEN = "opaque-call-token+with/encoding=="
CONNECT_URL = "wss://connector.example.test/stream?one=1&two=2"


class FakeTwilioRestException(Exception):
    def __init__(self, status: int, code: int) -> None:
        super().__init__("Twilio rejected the call")
        self.status = status
        self.code = code


def token_create(*, to: str, from_: str, twiml: str, call_token: str = "") -> None: ...


def legacy_create(*, to: str, from_: str, twiml: str) -> None: ...


class TokenCallList:
    create = staticmethod(token_create)


class LegacyCallList:
    create = staticmethod(legacy_create)


@pytest.fixture(autouse=True)
def mock_background_audio(monkeypatch: pytest.MonkeyPatch) -> None:
    # these tests exercise call origination only; no room or audio mixer is started
    monkeypatch.setattr(warm_transfer, "BackgroundAudioPlayer", Mock())


@pytest.fixture
def twilio_sdk(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Install a stub `twilio` package and return the mocked REST client."""

    def install(call_list: type = TokenCallList) -> Mock:
        client = Mock()
        client.calls.create = create_autospec(
            call_list.create, return_value=SimpleNamespace(sid="CA_test_transfer")
        )
        rest = SimpleNamespace(Client=Mock(return_value=client))
        modules = {
            "twilio": SimpleNamespace(rest=rest),
            "twilio.rest": rest,
            "twilio.base.exceptions": SimpleNamespace(TwilioRestException=FakeTwilioRestException),
            "twilio.http.http_client": SimpleNamespace(TwilioHttpClient=Mock()),
            "twilio.rest.api.v2010.account.call": SimpleNamespace(CallList=call_list),
        }
        for name, module in modules.items():
            monkeypatch.setitem(sys.modules, name, module)
        return client

    return install


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


async def dial(task: TwilioConnectorWarmTransferTask, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(task, "_wait_for_human_agent", AsyncMock())
    await task._originate_human_agent(
        room_name="consult-room", identity="human", room=Mock(spec=rtc.Room)
    )


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
    twilio_sdk: Any,
    connector: AsyncMock,
    options: dict[str, str],
    expected_from: str,
    expect_token: bool,
) -> None:
    client = twilio_sdk()

    await dial(build_task(**options), monkeypatch)

    connector.assert_awaited_once()
    client.calls.create.assert_called_once()
    kwargs = client.calls.create.call_args.kwargs
    assert kwargs["to"] == HUMAN_NUMBER
    assert kwargs["from_"] == expected_from
    assert ("call_token" in kwargs) is expect_token
    if expect_token:
        assert kwargs["call_token"] == CALL_TOKEN
    stream = ElementTree.fromstring(kwargs["twiml"]).find("./Connect/Stream")
    assert stream is not None and stream.attrib == {"url": CONNECT_URL}
    assert CALL_TOKEN not in kwargs["twiml"]
    assert CALL_TOKEN not in str(connector.call_args)


@pytest.mark.asyncio
@pytest.mark.parametrize("code", [21210, 21212], ids=["unverified-from", "invalid-from"])
async def test_rejected_caller_id_retries_once_without_token(
    monkeypatch: pytest.MonkeyPatch, twilio_sdk: Any, connector: AsyncMock, code: int
) -> None:
    client = twilio_sdk()
    client.calls.create.side_effect = [
        FakeTwilioRestException(400, code),
        SimpleNamespace(sid="CA_retry"),
    ]

    await dial(
        build_task(twilio_call_token=CALL_TOKEN, original_caller_number=CALLER_NUMBER), monkeypatch
    )

    assert client.calls.create.call_count == 2
    first, second = (call.kwargs for call in client.calls.create.call_args_list)
    assert first["from_"] == CALLER_NUMBER
    assert first["call_token"] == CALL_TOKEN
    assert second["from_"] == TWILIO_NUMBER
    assert "call_token" not in second


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "code"),
    [(400, 21211), (403, 21210), (500, 21210), (429, 20429), (403, 21212), (500, 21212)],
)
async def test_other_twilio_errors_propagate_without_retry(
    monkeypatch: pytest.MonkeyPatch, twilio_sdk: Any, connector: AsyncMock, status: int, code: int
) -> None:
    client = twilio_sdk()
    client.calls.create.side_effect = FakeTwilioRestException(status, code)

    with pytest.raises(RuntimeError, match=rf"HTTP {status}, code {code}"):
        await dial(
            build_task(twilio_call_token=CALL_TOKEN, original_caller_number=CALLER_NUMBER),
            monkeypatch,
        )

    client.calls.create.assert_called_once()


def test_call_token_requires_original_caller_number(twilio_sdk: Any) -> None:
    twilio_sdk()

    with pytest.raises(ValueError, match="original_caller_number"):
        build_task(twilio_call_token=CALL_TOKEN)


def test_call_token_requires_recent_twilio_sdk(twilio_sdk: Any) -> None:
    twilio_sdk(LegacyCallList)

    with pytest.raises(RuntimeError, match=r"twilio>=6\.55\.0"):
        build_task(twilio_call_token=CALL_TOKEN, original_caller_number=CALLER_NUMBER)

    # the token-free path stays usable on older SDKs
    build_task(original_caller_number=CALLER_NUMBER)


def test_missing_twilio_fails_at_construction_only_with_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "twilio.rest.api.v2010.account.call", None)
    with pytest.raises(ImportError, match=r"twilio>=6\.55\.0"):
        build_task(twilio_call_token=CALL_TOKEN, original_caller_number=CALLER_NUMBER)
    build_task()


@pytest.mark.asyncio
async def test_legacy_sdk_can_still_dial_without_token(
    monkeypatch: pytest.MonkeyPatch,
    twilio_sdk: Any,
    connector: AsyncMock,
) -> None:
    client = twilio_sdk(LegacyCallList)
    await dial(build_task(original_caller_number=CALLER_NUMBER), monkeypatch)
    client.calls.create.assert_called_once()
    assert "call_token" not in client.calls.create.call_args.kwargs


@pytest.mark.asyncio
@pytest.mark.parametrize("code", [21210, 21212])
async def test_failed_fallback_is_not_retried(
    monkeypatch: pytest.MonkeyPatch, twilio_sdk: Any, connector: AsyncMock, code: int
) -> None:
    client = twilio_sdk()
    client.calls.create.side_effect = FakeTwilioRestException(400, code)
    with pytest.raises(RuntimeError, match=rf"HTTP 400, code {code}"):
        await dial(
            build_task(twilio_call_token=CALL_TOKEN, original_caller_number=CALLER_NUMBER),
            monkeypatch,
        )
    assert client.calls.create.call_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("token", [None, ""])
@pytest.mark.parametrize("code", [21210, 21212])
async def test_business_number_rejection_is_not_retried(
    monkeypatch: pytest.MonkeyPatch,
    twilio_sdk: Any,
    connector: AsyncMock,
    token: str | None,
    code: int,
) -> None:
    client = twilio_sdk()
    client.calls.create.side_effect = FakeTwilioRestException(400, code)
    with pytest.raises(RuntimeError, match=rf"HTTP 400, code {code}"):
        await dial(
            build_task(**({"twilio_call_token": token} if token is not None else {})), monkeypatch
        )
    client.calls.create.assert_called_once()


@pytest.mark.asyncio
async def test_ambiguous_transport_failure_is_not_retried(
    monkeypatch: pytest.MonkeyPatch,
    twilio_sdk: Any,
    connector: AsyncMock,
) -> None:
    client = twilio_sdk()
    client.calls.create.side_effect = TimeoutError("request timed out")
    with pytest.raises(TimeoutError):
        await dial(
            build_task(twilio_call_token=CALL_TOKEN, original_caller_number=CALLER_NUMBER),
            monkeypatch,
        )
    client.calls.create.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("fallback", [False, True])
async def test_unanswered_call_cancels_the_created_sid(
    monkeypatch: pytest.MonkeyPatch,
    twilio_sdk: Any,
    connector: AsyncMock,
    fallback: bool,
) -> None:
    client = twilio_sdk()
    client.calls.create.side_effect = (
        [FakeTwilioRestException(400, 21210)] if fallback else []
    ) + [SimpleNamespace(sid="CA_created")]
    task = build_task(twilio_call_token=CALL_TOKEN, original_caller_number=CALLER_NUMBER)
    monkeypatch.setattr(
        task, "_wait_for_human_agent", AsyncMock(side_effect=ToolError("no answer"))
    )
    with pytest.raises(ToolError, match="no answer"):
        await task._originate_human_agent(room_name="consult", identity="human", room=Mock())
    await asyncio.gather(*task._twilio_tasks.tasks)
    client.calls.assert_called_once_with("CA_created")
    client.calls.return_value.update.assert_called_once_with(status="completed")


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["legacy", "forwarded", "fallback"])
@pytest.mark.parametrize("rejected", [False, True])
async def test_cancellation_returns_before_late_creation_and_retains_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    twilio_sdk: Any,
    connector: AsyncMock,
    mode: str,
    rejected: bool,
) -> None:
    client = twilio_sdk()
    task = build_task(
        **(
            {"twilio_call_token": CALL_TOKEN, "original_caller_number": CALLER_NUMBER}
            if mode != "legacy"
            else {}
        )
    )
    wait = AsyncMock()
    monkeypatch.setattr(task, "_wait_for_human_agent", wait)
    started = asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()
    attempts = 0

    def create(**kwargs: str) -> SimpleNamespace:
        nonlocal attempts
        attempts += 1
        if mode == "fallback" and attempts == 1:
            raise FakeTwilioRestException(400, 21210)
        loop.call_soon_threadsafe(started.set)
        if not release.wait(timeout=5):
            raise RuntimeError("test did not release worker")
        if rejected:
            raise FakeTwilioRestException(400, 21210)
        return SimpleNamespace(sid="CA_late")

    client.calls.create.side_effect = create
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
        outcomes = await asyncio.gather(*task._twilio_tasks.tasks, return_exceptions=True)
        assert all(
            result is None or isinstance(result, (str, FakeTwilioRestException))
            for result in outcomes
        )
    assert not task._twilio_tasks.tasks
    assert attempts == (2 if mode == "fallback" else 1)
    wait.assert_not_awaited()
    if rejected:
        client.calls.assert_not_called()
    else:
        client.calls.assert_called_once_with("CA_late")
        client.calls.return_value.update.assert_called_once_with(status="completed")


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_while_waiting", [False, True])
async def test_timeout_or_cancellation_does_not_wait_for_pending_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    twilio_sdk: Any,
    connector: AsyncMock,
    cancel_while_waiting: bool,
) -> None:
    client = twilio_sdk()
    task = build_task(twilio_call_token=CALL_TOKEN, original_caller_number=CALLER_NUMBER)
    error = asyncio.CancelledError() if cancel_while_waiting else ToolError("no answer")
    monkeypatch.setattr(task, "_wait_for_human_agent", AsyncMock(side_effect=error))
    started = asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()

    def cancel(**kwargs: str) -> None:
        loop.call_soon_threadsafe(started.set)
        if not release.wait(timeout=5):
            raise RuntimeError("test did not release cleanup")

    client.calls.return_value.update.side_effect = cancel
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


@pytest.fixture
def twilio_http(monkeypatch: pytest.MonkeyPatch) -> Mock:
    def response(request: requests.PreparedRequest, **kwargs: Any) -> requests.Response:
        result = requests.Response()
        result.status_code = 201 if request.url.endswith("/Calls.json") else 200
        result._content = json.dumps({"sid": "CA_test_transfer", "status": "in-progress"}).encode()
        return result

    send = Mock(side_effect=response)
    monkeypatch.setattr(requests.Session, "send", send)
    return send


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "code",
    [None, 21211, 21212, CALL_TOKEN],
    ids=["success", "rejected", "failed-fallback", "nonnumeric-code"],
)
async def test_real_twilio_sdk_does_not_log_call_token(
    monkeypatch: pytest.MonkeyPatch,
    connector: AsyncMock,
    twilio_http: Mock,
    caplog: pytest.LogCaptureFixture,
    code: int | str | None,
) -> None:
    caplog.set_level(logging.DEBUG)
    if code is not None:
        response = requests.Response()
        response.status_code = 400
        response._content = json.dumps({"code": code, "message": CALL_TOKEN}).encode()
        twilio_http.side_effect = None
        twilio_http.return_value = response

    task = build_task(twilio_call_token=CALL_TOKEN, original_caller_number=CALLER_NUMBER)
    try:
        if code is None:
            await dial(task, monkeypatch)
        else:
            expected_code = "unknown" if code == CALL_TOKEN else code
            with pytest.raises(RuntimeError, match=rf"HTTP 400, code {expected_code}"):
                try:
                    await dial(task, monkeypatch)
                except Exception:
                    warm_transfer.logger.exception("could not dial human agent")
                    raise
    finally:
        await asyncio.gather(*task._twilio_tasks.tasks)

    assert twilio_http.call_count == (2 if code == 21212 else 1)
    assert parse_qs(twilio_http.call_args_list[0].args[0].body)["CallToken"] == [CALL_TOKEN]
    assert CALL_TOKEN not in caplog.text
    logging.getLogger("twilio.http_client").info("unrelated Twilio client")
    assert "unrelated Twilio client" in caplog.text


@pytest.mark.asyncio
async def test_real_twilio_sdk_ends_an_answered_call_on_timeout(
    monkeypatch: pytest.MonkeyPatch,
    connector: AsyncMock,
    twilio_http: Mock,
) -> None:
    task = build_task(twilio_call_token=CALL_TOKEN, original_caller_number=CALLER_NUMBER)
    monkeypatch.setattr(
        task, "_wait_for_human_agent", AsyncMock(side_effect=ToolError("no answer"))
    )

    with pytest.raises(ToolError, match="no answer"):
        await task._originate_human_agent(room_name="consult", identity="human", room=Mock())
    await asyncio.gather(*task._twilio_tasks.tasks)

    assert twilio_http.call_count == 2
    request = twilio_http.call_args.args[0]
    assert request.url.endswith("/Calls/CA_test_transfer.json")
    assert parse_qs(request.body) == {"Status": ["completed"]}


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
