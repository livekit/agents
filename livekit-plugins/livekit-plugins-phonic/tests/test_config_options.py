from unittest.mock import AsyncMock

import pytest
from phonic.types import GenerateReplyPayload

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


async def test_text_modality_is_sent_in_additional_params() -> None:
    model = RealtimeModel(api_key="test", modalities=["text"], additional_params={"foo": 1})
    session = model.session()
    try:
        options = session._build_config_options(system_prompt="", tools_payload=[])
        assert options["additional_params"] == {
            "foo": 1,
            "input_modality": "text",
            "output_modality": "text",
        }
        assert not model.capabilities.audio_output
    finally:
        await session.aclose()


async def test_audio_modality_sends_no_modality_params() -> None:
    model = RealtimeModel(api_key="test")
    session = model.session()
    try:
        options = session._build_config_options(system_prompt="", tools_payload=[])
        assert "additional_params" not in options
        assert model.capabilities.audio_output
    finally:
        await session.aclose()


async def test_text_modality_sends_typed_turns_as_user_messages() -> None:
    model = RealtimeModel(api_key="test", modalities=["text"])
    session = model.session()
    try:
        session._socket = AsyncMock()
        session._ready_to_start.set()
        session._pending_user_text = "What are your hours?"
        await session._send_generate_reply(GenerateReplyPayload())

        payload = session._socket.send_generate_reply.call_args.args[0]
        assert payload.user_message == "What are your hours?"
        assert payload.system_message is None
    finally:
        session._socket = None
        await session.aclose()


async def test_text_modality_generations_carry_no_audio() -> None:
    model = RealtimeModel(api_key="test", modalities=["text"])
    session = model.session()
    try:
        generation = session._start_new_assistant_turn()
        message = await generation.message_stream.__anext__()
        assert await message.modalities == ["text"]
        assert message.audio_stream.closed
    finally:
        await session.aclose()
