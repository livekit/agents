import pytest

from livekit.plugins.phonic.realtime import RealtimeModel

pytestmark = pytest.mark.unit


async def test_external_id_is_sent_in_config() -> None:
    model = RealtimeModel(api_key="test", external_id="call-123")
    session = model.session()
    try:
        options = session._build_config_options(system_prompt="", tools_payload=[])
        assert options["external_id"] == "call-123"
    finally:
        await session.aclose()


async def test_external_id_is_omitted_when_not_given() -> None:
    model = RealtimeModel(api_key="test")
    session = model.session()
    try:
        options = session._build_config_options(system_prompt="", tools_payload=[])
        assert "external_id" not in options
    finally:
        await session.aclose()


async def test_session_reconnects_on_abnormal_disconnect() -> None:
    model = RealtimeModel(api_key="test")
    session = model.session()
    try:
        assert session._client._client_wrapper._reconnect_conversation_on_abnormal_disconnect
    finally:
        await session.aclose()
