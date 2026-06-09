import pytest

from livekit.agents import llm, utils
from livekit.agents.inference.sts import STS, STSSession, _ResponseGeneration

pytestmark = pytest.mark.unit


def _make_session() -> STSSession:
    model = STS(
        model="openai/gpt-realtime",
        api_key="test-key",
        api_secret="test-secret",
        base_url="https://example.livekit.cloud",
    )
    return model.session()


def _new_generation() -> _ResponseGeneration:
    return _ResponseGeneration(
        message_ch=utils.aio.Chan(),
        function_ch=utils.aio.Chan(),
        messages={},
        response_id="resp_1",
    )


@pytest.mark.asyncio
async def test_function_call_emitted_only_when_arguments_complete():
    """Function-call arguments only arrive by output_item.done, so the FunctionCall
    must be emitted there (not at output_item.added, where arguments are empty)."""
    session = _make_session()
    session._current_generation = _new_generation()

    session._handle_response_output_item_added(
        {
            "item": {
                "id": "fc_1",
                "type": "function_call",
                "call_id": "call_1",
                "name": "get_weather",
                "arguments": "",
            }
        }
    )
    # nothing should be emitted while arguments are still empty
    assert session._current_generation.function_ch.qsize() == 0

    session._handle_response_output_item_done(
        {
            "item": {
                "id": "fc_1",
                "type": "function_call",
                "call_id": "call_1",
                "name": "get_weather",
                "arguments": '{"city": "SF"}',
            }
        }
    )

    fc = session._current_generation.function_ch.recv_nowait()
    assert isinstance(fc, llm.FunctionCall)
    assert fc.id == "fc_1"
    assert fc.call_id == "call_1"
    assert fc.name == "get_weather"
    assert fc.arguments == '{"city": "SF"}'
    # exactly one function call emitted
    assert session._current_generation.function_ch.qsize() == 0


@pytest.mark.asyncio
async def test_incomplete_function_call_on_done_is_skipped():
    """A function_call item missing call_id/name should not emit a partial FunctionCall."""
    session = _make_session()
    session._current_generation = _new_generation()

    session._handle_response_output_item_done(
        {"item": {"id": "fc_1", "type": "function_call", "call_id": "", "name": "", "arguments": ""}}
    )
    assert session._current_generation.function_ch.qsize() == 0
