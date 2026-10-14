"""Type checks for ``Agent[Deps, State, Output]``, run by ``scripts/check_types.py``.

Each ``type: ignore[code]`` marks an error the type checker must report; strict mode reports an
ignore that is no longer needed, so the file fails both when a check is lost and when a new
error appears.
"""

from dataclasses import dataclass
from typing import Any

from typing_extensions import assert_type

from livekit.agents import Agent, RunContext, RunResult


@dataclass
class Calendar:
    url: str = ""


@dataclass
class Patient:
    name: str = ""


@dataclass
class Claim:
    number: str = ""


backend = Agent(instructions="", deps_type=Calendar, state_type=Patient)
fnol = Agent(instructions="", output_type=Claim)
plain = Agent(instructions="")

assert_type(backend, Agent[Calendar, Patient, Any])
assert_type(fnol, Agent[Any, Any, Claim])
assert_type(plain, Agent[Any, Any, Any])


@backend.instructions
def known(ctx: RunContext[Calendar, Patient]) -> str:
    return ctx.state.name


@backend.instructions  # type: ignore[arg-type]
def other_context(ctx: RunContext[str, int]) -> str:
    return ""


@fnol.output_validator
async def claim_number(ctx: RunContext, output: Claim) -> Claim:
    return output


@fnol.output_validator  # type: ignore[arg-type]
async def other_output(ctx: RunContext, output: str) -> str:
    return output


async def runs(patient: Patient) -> None:
    assert_type(await fnol.run("hi"), RunResult[Claim])
    await backend.run("hi", state=patient, deps=Calendar())
    await backend.run("hi", state="a name")  # type: ignore[arg-type]
    await backend.run("hi", deps=42)  # type: ignore[arg-type]

    result = await fnol.run("hi")
    if result.output is not None:
        _ = result.output.numbr  # type: ignore[attr-defined]


class FrontDesk(Agent):
    pass


def any_agent(agent: Agent) -> None: ...


any_agent(backend)
any_agent(fnol)
any_agent(FrontDesk(instructions=""))
