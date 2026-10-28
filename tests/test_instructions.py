"""Static agent and workflow instruction authoring."""

import pytest

from livekit.agents import Agent
from livekit.agents.beta.workflows import WarmTransfer, WorkflowInstructions

pytestmark = pytest.mark.unit


def test_workflow_sections_produce_static_agent_instructions() -> None:
    transfer = WarmTransfer(
        "+15550100",
        sip_trunk_id="trunk",
        instructions=WorkflowInstructions(
            persona="You are the desk.", extra="Confirm the booking."
        ),
    )
    instructions = transfer._agent.instructions
    assert isinstance(instructions, str)
    assert "You are the desk." in instructions
    assert "Confirm the booking." in instructions
    assert instructions.count("You are the desk.") == 1
    for placeholder in ("{persona}", "{extra}", "{_conversation_history}"):
        assert placeholder not in instructions


def test_agent_instructions_refuse_non_text_authoring() -> None:
    class ModalInstructions:
        common = "common"
        audio = "voice"
        text = "text"

    with pytest.raises(TypeError, match="instructions must be a string"):
        Agent(instructions=ModalInstructions())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("instructions", "expected", "absent"),
    [
        ("Fixed text.", "Fixed text.", "# Context"),
        (None, "# Identity", "{persona}"),
        (WorkflowInstructions(persona=""), "# Context", "# Identity"),
    ],
)
def test_workflow_static_instructions_keep_fixed_default_and_removed_sections(
    instructions: str | WorkflowInstructions | None, expected: str, absent: str
) -> None:
    options = {} if instructions is None else {"instructions": instructions}
    transfer = WarmTransfer("+15550100", sip_trunk_id="trunk", **options)
    assert expected in transfer._agent.instructions
    assert absent not in transfer._agent.instructions
