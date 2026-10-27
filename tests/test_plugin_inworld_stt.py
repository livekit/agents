import pytest

from .stt_contract_cases import CASES_BY_NAME, assert_control_write_failure

pytestmark = pytest.mark.plugin("inworld")


@pytest.mark.virtual_time
@pytest.mark.no_concurrent
@pytest.mark.parametrize(
    ("message", "action"),
    [("audioChunk", "audio"), ("endTurn", "flush"), ("closeStream", "end")],
)
async def test_send_failure_is_not_treated_as_a_successful_stop(message, action, monkeypatch):
    await assert_control_write_failure(CASES_BY_NAME["inworld"], monkeypatch, message, action)
