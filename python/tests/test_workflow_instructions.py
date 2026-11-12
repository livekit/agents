"""Workflow briefing configuration through its generated native boundary."""

import pytest

from livekit import api
from livekit.agents import _core
from livekit.agents.beta.workflows import WorkflowInstructions

pytestmark = pytest.mark.unit

_TRUNK = api.SIPOutboundConfig(hostname="sip.example.com").SerializeToString()


def test_warm_transfer_keeps_an_omitted_persona_and_honors_an_explicit_empty_one() -> None:
    default = _core.WarmTransfer(sip_call_to="+15550100", sip_connection=_TRUNK)
    omitted = _core.WarmTransfer(
        sip_call_to="+15550100",
        sip_connection=_TRUNK,
        instructions=WorkflowInstructions(extra="Mention the booking number."),
    )
    empty = _core.WarmTransfer(
        sip_call_to="+15550100",
        sip_connection=_TRUNK,
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
            sip_connection=_TRUNK,
            instructions=WorkflowInstructions(persona=None),  # type: ignore[arg-type]
        )


def test_a_warm_transfer_dials_through_its_sip_connection_without_a_trunk_id() -> None:
    with pytest.raises(ValueError, match="sip_connection"):
        _core.WarmTransfer(sip_call_to="+15550100")
    with pytest.raises(ValueError, match="invalid sip_connection"):
        _core.WarmTransfer(sip_call_to="+15550100", sip_connection=b"\xff")
