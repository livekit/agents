import sys
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
        twilio_from_number=CALLER_NUMBER if use_call_token else TWILIO_NUMBER,
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
        twilio_from_number=from_number,
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
        twilio_from_number=CALLER_NUMBER,
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
        twilio_from_number=CALLER_NUMBER,
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
