"""Workflow briefing configuration through its generated native boundary."""

import pytest

from livekit.agents import _core
from livekit.agents.beta.workflows import WorkflowInstructions

pytestmark = pytest.mark.unit


def test_warm_transfer_keeps_an_omitted_persona_and_honors_an_explicit_empty_one() -> None:
    default = _core.WarmTransfer(sip_call_to="+15550100", has_sip_connection=True)
    omitted = _core.WarmTransfer(
        sip_call_to="+15550100",
        has_sip_connection=True,
        instructions=WorkflowInstructions(extra="Mention the booking number."),
    )
    empty = _core.WarmTransfer(
        sip_call_to="+15550100",
        has_sip_connection=True,
        instructions=WorkflowInstructions(persona=""),
    )
    assert "# Identity" in default.instructions
    assert "# Identity" in omitted.instructions
    assert "Mention the booking number." in omitted.instructions
    assert "# Identity" not in empty.instructions
    assert "# Context" in empty.instructions

    with pytest.raises(TypeError, match="persona"):
        _core.WarmTransfer(
            sip_call_to="+15550100",
            has_sip_connection=True,
            instructions=WorkflowInstructions(persona=None),  # type: ignore[arg-type]
        )
