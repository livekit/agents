"""Type checks for ``Agent[Context, Output]``, run by ``scripts/check_types.py``.

The constructor infers both parameters, as pydantic-ai's: ``context_type`` the context,
``output_type`` the output, ``str`` when omitted.

Each ``type: ignore[code]`` marks an error the type checker must report; strict mode reports an
ignore that is no longer needed, so the file fails both when a check is lost and when a new
error appears.
"""

from collections.abc import AsyncIterable
from dataclasses import dataclass
from typing import Any

from typing_extensions import assert_type

from livekit.agents import Agent, ModelSettings, RunContext, RunResult, StopResponse, llm


@dataclass
class Calendar:
    url: str = ""


@dataclass
class Patient:
    name: str = ""


@dataclass
class Claim:
    number: str = ""


@dataclass
class AppContext:
    calendar: Calendar
    patient: Patient


backend = Agent(instructions="", context_type=AppContext)
fnol = Agent(instructions="", output_type=Claim)
plain = Agent(instructions="")
plain_none = Agent(instructions="", output_type=None)
typed = Agent(instructions="", context_type=AppContext, output_type=Claim)
explicit = Agent[AppContext, Claim](instructions="", output_type=Claim)


async def claim_of(ctx: RunContext[AppContext]) -> Claim:
    return Claim()


resolved = Agent(instructions="", context_type=AppContext, output_type=claim_of)

assert_type(backend, Agent[AppContext, str])
assert_type(fnol, Agent[Any, Claim])
assert_type(plain, Agent[Any, str])
assert_type(plain_none, Agent[Any, str])
assert_type(typed, Agent[AppContext, Claim])
assert_type(explicit, Agent[AppContext, Claim])
assert_type(resolved, Agent[AppContext, Claim])


@backend.instructions
def known(ctx: RunContext[AppContext]) -> str:
    return ctx.context.patient.name


@backend.instructions  # type: ignore[arg-type]
def other_context(ctx: RunContext[str]) -> str:
    return ""


@fnol.output_validator
async def claim_number(ctx: RunContext, output: Claim) -> Claim:
    return output


@fnol.output_validator  # type: ignore[arg-type]
async def other_output(ctx: RunContext, output: str) -> str:
    return output


async def runs(patient: Patient) -> None:
    assert_type(await fnol.run("hi"), RunResult[Claim])
    assert_type(await plain.run("hi"), RunResult[str])
    assert_type(await typed.run("hi", context=AppContext(Calendar(), patient)), RunResult[Claim])
    await typed.run("hi", context="a name")  # type: ignore[arg-type]
    await backend.run("hi", context=AppContext(calendar=Calendar(), patient=patient))
    await backend.run("hi", context="a name")  # type: ignore[arg-type]
    await backend.run("hi", context=42)  # type: ignore[arg-type]

    result = await fnol.run("hi")
    if result.status == "completed":
        _ = result.output.numbr  # type: ignore[attr-defined]


class FrontDesk(Agent):
    """The node overrides of voice-session.md "Customizing the pipeline"."""

    async def on_user_turn_completed(
        self, turn_ctx: llm.ChatContext, new_message: llm.ChatMessage
    ) -> None:
        if (new_message.text_content or "").strip().lower() == "hold on":
            raise StopResponse()

    async def transcription_node(
        self, text: AsyncIterable[str], model_settings: ModelSettings
    ) -> AsyncIterable[str]:
        async for chunk in text:
            yield chunk.replace("Live Kit", "LiveKit")


def any_agent(agent: Agent) -> None: ...


any_agent(backend)
any_agent(fnol)
any_agent(FrontDesk(instructions=""))
