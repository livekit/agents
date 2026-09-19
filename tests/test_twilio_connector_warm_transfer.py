import asyncio
import sys
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, create_autospec
from xml.etree import ElementTree

import pytest

from livekit import api, rtc
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
        super().__init__("Twilio rejected call")
        self.status = status
        self.code = code


def legacy_create(*, to: str, from_: str, twiml: str) -> None:
    pass


def token_create(*, to: str, from_: str, twiml: str, call_token: str = "") -> None:
    pass


@pytest.fixture(autouse=True)
def mock_background_audio(monkeypatch: pytest.MonkeyPatch) -> None:
    # These tests exercise call origination without starting a room or an audio mixer.
    monkeypatch.setattr(warm_transfer, "BackgroundAudioPlayer", Mock())


@pytest.fixture
def twilio_client(monkeypatch: pytest.MonkeyPatch) -> Mock:
    client = Mock()
    client.calls.create = create_autospec(token_create)
    client.calls.create.return_value = SimpleNamespace(sid="CA_test_transfer")
    rest = SimpleNamespace(Client=Mock(return_value=client))
    monkeypatch.setitem(sys.modules, "twilio", SimpleNamespace(rest=rest))
    monkeypatch.setitem(sys.modules, "twilio.rest", rest)
    monkeypatch.setitem(
        sys.modules,
        "twilio.base.exceptions",
        SimpleNamespace(TwilioRestException=FakeTwilioRestException),
    )
    monkeypatch.setitem(
        sys.modules, "twilio.http.http_client", SimpleNamespace(TwilioHttpClient=Mock())
    )
    return client


@pytest.fixture
def connector(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    connect = AsyncMock(return_value=SimpleNamespace(connect_url=CONNECT_URL))
    ctx = SimpleNamespace(
        api=SimpleNamespace(connector=SimpleNamespace(connect_twilio_call=connect))
    )
    monkeypatch.setattr(warm_transfer, "get_job_context", lambda: ctx)
    return connect


@pytest.mark.asyncio
@pytest.mark.parametrize("use_call_token", [False, True])
async def test_legacy_sdk_compatibility(
    monkeypatch: pytest.MonkeyPatch, twilio_client: Mock, connector: AsyncMock, use_call_token: bool
) -> None:
    twilio_client.calls.create = create_autospec(
        legacy_create, return_value=SimpleNamespace(sid="CA_test_transfer")
    )
    options = {"twilio_call_token": CALL_TOKEN} if use_call_token else {}
    task = TwilioConnectorWarmTransferTask(
        HUMAN_NUMBER,
        twilio_from_number=TWILIO_NUMBER,
        original_caller_number=CALLER_NUMBER,
        twilio_account_sid="AC_test_account",
        twilio_auth_token="test_auth_token",
        **options,
    )
    wait_for_answer = AsyncMock()
    monkeypatch.setattr(task, "_wait_for_human_agent", wait_for_answer)
    room = Mock(spec=rtc.Room)

    if use_call_token:
        with pytest.raises(RuntimeError, match=r"pip install 'twilio>=6\.55\.0'"):
            await task._originate_human_agent(room_name="consult-room", identity="human", room=room)
        connector.assert_not_awaited()
        twilio_client.calls.create.assert_not_called()
        wait_for_answer.assert_not_awaited()
    else:
        await task._originate_human_agent(room_name="consult-room", identity="human", room=room)
        twilio_client.calls.create.assert_called_once()
        assert "call_token" not in twilio_client.calls.create.call_args.kwargs
        wait_for_answer.assert_awaited_once_with(room=room, identity="human")


@pytest.mark.asyncio
@pytest.mark.parametrize("use_call_token", [False, True])
async def test_transfer_dial_preserves_caller_id(
    monkeypatch: pytest.MonkeyPatch, twilio_client: Mock, connector: AsyncMock, use_call_token: bool
) -> None:
    options = {"twilio_call_token": CALL_TOKEN} if use_call_token else {}
    from_number = CALLER_NUMBER if use_call_token else TWILIO_NUMBER
    task = TwilioConnectorWarmTransferTask(
        HUMAN_NUMBER,
        twilio_from_number=TWILIO_NUMBER,
        original_caller_number=CALLER_NUMBER,
        twilio_account_sid="AC_test_account",
        twilio_auth_token="test_auth_token",
        **options,
    )
    wait_for_answer = AsyncMock()
    monkeypatch.setattr(task, "_wait_for_human_agent", wait_for_answer)
    room = Mock(spec=rtc.Room)

    await task._originate_human_agent(room_name="consult-room", identity="human", room=room)

    connector.assert_awaited_once_with(
        api.ConnectTwilioCallRequest(
            twilio_call_direction=api.ConnectTwilioCallRequest.TwilioCallDirection.TWILIO_CALL_DIRECTION_OUTBOUND,
            room_name="consult-room",
            participant_identity="human",
        )
    )
    twilio_client.calls.create.assert_called_once()
    dial = twilio_client.calls.create.call_args.kwargs
    assert dial["from_"] == from_number
    assert dial["to"] == HUMAN_NUMBER
    if use_call_token:
        assert dial["call_token"] == CALL_TOKEN
    else:
        assert "call_token" not in dial
    stream = ElementTree.fromstring(dial["twiml"]).find("./Connect/Stream")
    assert stream is not None and stream.attrib == {"url": CONNECT_URL}
    assert CALL_TOKEN not in dial["twiml"]
    wait_for_answer.assert_awaited_once_with(room=room, identity="human")


@pytest.mark.asyncio
async def test_rejected_call_token_does_not_retry_with_another_caller_id(
    monkeypatch: pytest.MonkeyPatch, twilio_client: Mock, connector: AsyncMock
) -> None:
    task = TwilioConnectorWarmTransferTask(
        HUMAN_NUMBER,
        twilio_from_number=TWILIO_NUMBER,
        original_caller_number=CALLER_NUMBER,
        twilio_call_token=CALL_TOKEN,
        twilio_account_sid="AC_test_account",
        twilio_auth_token="test_auth_token",
    )
    wait_for_answer = AsyncMock()
    monkeypatch.setattr(task, "_wait_for_human_agent", wait_for_answer)
    twilio_client.calls.create.side_effect = RuntimeError("Twilio rejected the forwarded call")

    with pytest.raises(RuntimeError, match="Twilio rejected"):
        await task._originate_human_agent(
            room_name="consult-room", identity="human", room=Mock(spec=rtc.Room)
        )

    twilio_client.calls.create.assert_called_once()
    wait_for_answer.assert_not_awaited()


@pytest.mark.asyncio
async def test_call_token_transfer_cancels_unanswered_call(
    monkeypatch: pytest.MonkeyPatch, twilio_client: Mock, connector: AsyncMock
) -> None:
    task = TwilioConnectorWarmTransferTask(
        HUMAN_NUMBER,
        twilio_from_number=TWILIO_NUMBER,
        original_caller_number=CALLER_NUMBER,
        twilio_call_token=CALL_TOKEN,
        twilio_account_sid="AC_test_account",
        twilio_auth_token="test_auth_token",
    )
    monkeypatch.setattr(
        task, "_wait_for_human_agent", AsyncMock(side_effect=ToolError("supervisor did not answer"))
    )

    with pytest.raises(ToolError, match="supervisor did not answer"):
        await task._originate_human_agent(
            room_name="consult-room", identity="human", room=Mock(spec=rtc.Room)
        )

    twilio_client.calls.assert_called_once_with("CA_test_transfer")
    twilio_client.calls.return_value.update.assert_called_once_with(status="canceled")


@pytest.mark.asyncio
@pytest.mark.parametrize("second_failure", [False, True])
async def test_caller_id_rejection_retries_once(
    monkeypatch: pytest.MonkeyPatch,
    twilio_client: Mock,
    connector: AsyncMock,
    second_failure: bool,
) -> None:
    task = TwilioConnectorWarmTransferTask(
        HUMAN_NUMBER,
        twilio_from_number=TWILIO_NUMBER,
        original_caller_number=CALLER_NUMBER,
        twilio_call_token=CALL_TOKEN,
        twilio_account_sid="AC_test_account",
        twilio_auth_token="test_auth_token",
    )
    wait = AsyncMock()
    monkeypatch.setattr(task, "_wait_for_human_agent", wait)
    twilio_client.calls.create.side_effect = [
        FakeTwilioRestException(400, 21210),
        FakeTwilioRestException(400, 21210)
        if second_failure
        else SimpleNamespace(sid="CA_fallback"),
    ]
    if second_failure:
        with pytest.raises(FakeTwilioRestException):
            await task._originate_human_agent(room_name="consult", identity="human", room=Mock())
        wait.assert_not_awaited()
    else:
        await task._originate_human_agent(room_name="consult", identity="human", room=Mock())
        wait.assert_awaited_once()
    assert twilio_client.calls.create.call_count == 2
    first, second = [call.kwargs for call in twilio_client.calls.create.call_args_list]
    assert first["from_"] == CALLER_NUMBER
    assert first["call_token"] == CALL_TOKEN
    assert second == {"to": HUMAN_NUMBER, "from_": TWILIO_NUMBER, "twiml": first["twiml"]}
    connector.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("status,code", [(400, 21211), (403, 21210), (429, 20429), (500, 21210)])
async def test_unrelated_rejections_do_not_retry(
    monkeypatch: pytest.MonkeyPatch,
    twilio_client: Mock,
    connector: AsyncMock,
    status: int,
    code: int,
) -> None:
    task = TwilioConnectorWarmTransferTask(
        HUMAN_NUMBER,
        twilio_from_number=TWILIO_NUMBER,
        original_caller_number=CALLER_NUMBER,
        twilio_call_token=CALL_TOKEN,
        twilio_account_sid="AC_test_account",
        twilio_auth_token="test_auth_token",
    )
    twilio_client.calls.create.side_effect = FakeTwilioRestException(status, code)
    with pytest.raises(FakeTwilioRestException):
        await task._originate_human_agent(room_name="consult", identity="human", room=Mock())
    twilio_client.calls.create.assert_called_once()


def test_token_requires_original_caller() -> None:
    with pytest.raises(ValueError, match="requires original_caller_number"):
        TwilioConnectorWarmTransferTask(
            HUMAN_NUMBER,
            twilio_from_number=TWILIO_NUMBER,
            twilio_call_token=CALL_TOKEN,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("token", [None, ""])
async def test_business_caller_rejection_does_not_retry(
    twilio_client: Mock,
    connector: AsyncMock,
    token: str | None,
) -> None:
    options = {"twilio_call_token": token} if token is not None else {}
    task = TwilioConnectorWarmTransferTask(
        HUMAN_NUMBER,
        twilio_from_number=TWILIO_NUMBER,
        original_caller_number=CALLER_NUMBER,
        twilio_account_sid="AC_test",
        twilio_auth_token="test",
        **options,
    )
    twilio_client.calls.create.side_effect = FakeTwilioRestException(400, 21210)
    with pytest.raises(FakeTwilioRestException):
        await task._originate_human_agent(room_name="consult", identity="human", room=Mock())
    twilio_client.calls.create.assert_called_once()
    assert twilio_client.calls.create.call_args.kwargs["from_"] == TWILIO_NUMBER
    assert "call_token" not in twilio_client.calls.create.call_args.kwargs


@pytest.mark.asyncio
async def test_unanswered_fallback_cancels_fallback_call(
    monkeypatch: pytest.MonkeyPatch,
    twilio_client: Mock,
    connector: AsyncMock,
) -> None:
    task = TwilioConnectorWarmTransferTask(
        HUMAN_NUMBER,
        twilio_from_number=TWILIO_NUMBER,
        original_caller_number=CALLER_NUMBER,
        twilio_call_token=CALL_TOKEN,
        twilio_account_sid="AC_test",
        twilio_auth_token="test",
    )
    twilio_client.calls.create.side_effect = [
        FakeTwilioRestException(400, 21210),
        SimpleNamespace(sid="CA_fallback"),
    ]
    monkeypatch.setattr(
        task, "_wait_for_human_agent", AsyncMock(side_effect=ToolError("no answer"))
    )
    with pytest.raises(ToolError):
        await task._originate_human_agent(room_name="consult", identity="human", room=Mock())
    assert twilio_client.calls.create.call_count == 2
    twilio_client.calls.assert_called_once_with("CA_fallback")
    twilio_client.calls.return_value.update.assert_called_once_with(status="canceled")


@pytest.mark.asyncio
@pytest.mark.parametrize("fallback", [False, True])
@pytest.mark.parametrize("rejected", [False, True])
async def test_cancellation_during_call_creation_retains_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    twilio_client: Mock,
    connector: AsyncMock,
    fallback: bool,
    rejected: bool,
) -> None:
    task = TwilioConnectorWarmTransferTask(
        HUMAN_NUMBER,
        twilio_from_number=TWILIO_NUMBER,
        original_caller_number=CALLER_NUMBER,
        twilio_call_token=CALL_TOKEN,
        twilio_account_sid="AC_test",
        twilio_auth_token="test",
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
        if fallback and attempts == 1:
            raise FakeTwilioRestException(400, 21210)
        loop.call_soon_threadsafe(started.set)
        if not release.wait(timeout=5):
            raise RuntimeError("test did not release worker")
        if rejected:
            raise FakeTwilioRestException(400, 21210)
        return SimpleNamespace(sid="CA_late_call")

    twilio_client.calls.create.side_effect = create
    dial = asyncio.create_task(
        task._originate_human_agent(
            room_name="consult",
            identity="human",
            room=Mock(),
        )
    )
    try:
        await asyncio.wait_for(started.wait(), timeout=5)
        dial.cancel()
        await asyncio.sleep(0)
        dial.cancel()  # teardown can request cancellation more than once
        await asyncio.sleep(0)
        assert not dial.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(dial, timeout=5)
        assert attempts == (2 if fallback else 1)
        wait.assert_not_awaited()
        if rejected:
            twilio_client.calls.assert_not_called()
        else:
            twilio_client.calls.assert_called_once_with("CA_late_call")
            twilio_client.calls.return_value.update.assert_called_once_with(status="canceled")
    finally:
        release.set()
        if not dial.done():
            dial.cancel()
            await asyncio.gather(dial, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("stall", ["create", "unanswered_cancel"])
async def test_stalled_cleanup_has_deadline_and_retains_late_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    twilio_client: Mock,
    connector: AsyncMock,
    stall: str,
) -> None:
    monkeypatch.setattr(warm_transfer, "_TWILIO_CLEANUP_TIMEOUT", 0.03)
    task = TwilioConnectorWarmTransferTask(
        HUMAN_NUMBER,
        twilio_from_number=TWILIO_NUMBER,
        original_caller_number=CALLER_NUMBER,
        twilio_call_token=CALL_TOKEN,
        twilio_account_sid="AC_test",
        twilio_auth_token="test",
    )
    started = asyncio.Event()
    release = threading.Event()
    cleaned = asyncio.Event()
    loop = asyncio.get_running_loop()
    attempts = 0

    def block() -> None:
        loop.call_soon_threadsafe(started.set)
        if not release.wait(timeout=3):
            raise RuntimeError("worker not released")

    def create(**kwargs: str) -> SimpleNamespace:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise FakeTwilioRestException(400, 21210)
        if stall == "create":
            block()
        return SimpleNamespace(sid="CA_late")

    def cancel(**kwargs: str) -> None:
        if stall != "create":
            block()
        loop.call_soon_threadsafe(cleaned.set)

    twilio_client.calls.create.side_effect = create
    twilio_client.calls.return_value.update.side_effect = cancel
    wait = AsyncMock(side_effect=ToolError("no answer"))
    monkeypatch.setattr(task, "_wait_for_human_agent", wait)
    dial = asyncio.create_task(
        task._originate_human_agent(
            room_name="consult",
            identity="human",
            room=Mock(),
        )
    )
    try:
        await asyncio.wait_for(started.wait(), timeout=1)
        if stall == "create":
            dial.cancel()
        # For cancel stalls, the answer failure already initiated cleanup.
        for _ in range(3):
            await asyncio.sleep(0)
            if not dial.done():
                dial.cancel()
        done, _ = await asyncio.wait({dial}, timeout=0.5)
        assert dial in done, "teardown exceeded its cleanup deadline"
        with pytest.raises(asyncio.CancelledError):
            dial.result()
        assert not release.is_set()
        assert warm_transfer._twilio_cleanup_tasks
        release.set()
        await asyncio.wait_for(cleaned.wait(), timeout=1)
        await asyncio.gather(*warm_transfer._twilio_cleanup_tasks)
        await asyncio.sleep(0)
        assert not warm_transfer._twilio_cleanup_tasks
        twilio_client.calls.assert_called_once_with("CA_late")
        assert attempts == 2
    finally:
        release.set()
        await asyncio.gather(dial, *warm_transfer._twilio_cleanup_tasks, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("cancelled", [False, True])
async def test_failure_cleanup_preserves_shutdown_cancellation(
    monkeypatch: pytest.MonkeyPatch,
    twilio_client: Mock,
    connector: AsyncMock,
    cancelled: bool,
) -> None:
    task = TwilioConnectorWarmTransferTask(
        HUMAN_NUMBER,
        twilio_from_number=TWILIO_NUMBER,
        twilio_account_sid="AC_test",
        twilio_auth_token="test",
    )
    monkeypatch.setattr(
        task, "_wait_for_human_agent", AsyncMock(side_effect=ToolError("no answer"))
    )
    started = asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()

    def cancel(**kwargs: str) -> None:
        loop.call_soon_threadsafe(started.set)
        assert release.wait(timeout=3)

    twilio_client.calls.return_value.update.side_effect = cancel
    dial = asyncio.create_task(
        task._originate_human_agent(
            room_name="consult",
            identity="human",
            room=Mock(),
        )
    )
    try:
        await asyncio.wait_for(started.wait(), timeout=1)
        if cancelled:
            dial.cancel("shutdown")
            await asyncio.sleep(0)
            dial.cancel("shutdown again")
            await asyncio.sleep(0)
        release.set()
        with pytest.raises(asyncio.CancelledError if cancelled else ToolError):
            await asyncio.wait_for(dial, timeout=1)
        assert dial.cancelled() is cancelled
        twilio_client.calls.return_value.update.assert_called_once_with(status="canceled")
    finally:
        release.set()
        await asyncio.gather(dial, *warm_transfer._twilio_cleanup_tasks, return_exceptions=True)
