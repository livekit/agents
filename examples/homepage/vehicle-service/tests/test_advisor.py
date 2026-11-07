"""Nick's calls without a model: hanging up."""

from datetime import date
from types import SimpleNamespace

import pytest
from livekit.agents import llm
from livekit.agents.llm import ToolError

from advisor import ServiceAdvisor
from service_desk import ServiceDesk

MONDAY = date(2026, 10, 19)


class FakeContext:
    def __init__(self, call_id: str, items: list[llm.ChatItem]) -> None:
        self.function_call = llm.FunctionCall(
            call_id=call_id, name="end_call", arguments="{}"
        )
        self.session = SimpleNamespace(history=SimpleNamespace(items=items))
        self.speech_handle = SimpleNamespace(add_done_callback=lambda cb: None)

    async def wait_for_playout(self) -> None:
        pass


def said(role: str, text: str, *, interrupted: bool = False) -> llm.ChatMessage:
    return llm.ChatMessage(role=role, content=[text], interrupted=interrupted)


def tool_output(
    call_id: str, name: str, *, is_error: bool = False
) -> llm.FunctionCallOutput:
    return llm.FunctionCallOutput(
        call_id=call_id, name=name, output="{}", is_error=is_error
    )


async def test_a_cut_off_confirmation_is_replayed_before_hanging_up():
    agent = ServiceAdvisor(ServiceDesk(MONDAY))
    agent._must_tell(FakeContext("call_done", []), "their booking")
    items: list[llm.ChatItem] = [
        tool_output("call_done", "book_visit"),
        said("assistant", "You're booked for Wednesday at", interrupted=True),
        said("user", "Sorry, what was that?"),
        said("assistant", "Wednesday at seven thirty with Dana, shuttle from home."),
        said("user", "Great, bye."),
        said("assistant", "Bye!"),
    ]
    with pytest.raises(ToolError, match="first tell them their booking"):
        await agent.end_call(FakeContext("call_end", items))
    items += [
        tool_output("call_end", "end_call", is_error=True),
        said("assistant", "Wednesday at seven thirty, shuttle from home. Bye!"),
    ]
    assert await agent.end_call(FakeContext("call_end_again", items)) is None
