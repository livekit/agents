from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from order import OrderedItem, OrderState
from pydantic import TypeAdapter

from livekit.agents import Agent, AgentSession, SimulationContext
from livekit.agents.voice.run_result import mock_tools

# Per-scenario userdata (see scenarios.yaml):
#   expected_order:    the items the order must hold when the run ends; order ids are ignored
#   unavailable_items: menu item ids marked unavailable before the agent starts
#   failing_tools:     tools that raise instead of running

_ordered_items = TypeAdapter(list[OrderedItem])


def setup(sim: SimulationContext, session: AgentSession, agent: type[Agent]) -> None:
    """Apply the scenario's userdata. Must run before the agent is built, since
    the agent's instructions list the menu's availability."""
    userdata = session.userdata
    unavailable = set(sim.userdata().get("unavailable_items", []))
    for items in (
        userdata.drink_items,
        userdata.combo_items,
        userdata.happy_items,
        userdata.regular_items,
        userdata.sauce_items,
    ):
        for item in items:
            if item.id in unavailable:
                item.available = False

    failing = sim.userdata().get("failing_tools", [])
    mock_tools(
        agent,
        {name: lambda: RuntimeError("simulated tool failure") for name in failing},
        session=session,
    )


async def on_simulation_end(ctx: SimulationContext) -> None:
    # grade the run on the final order; a mismatch vetoes the run
    if "expected_order" not in ctx.userdata():
        return  # scenario graded on conversation only

    order: OrderState = ctx.job_context.primary_session.userdata.order
    expected = _canonical(_ordered_items.validate_python(ctx.userdata()["expected_order"]))
    actual = _canonical(order.items.values())
    if actual != expected:
        ctx.fail(reason=f"expected the order to be {expected}, got {actual}")


def _canonical(items: Iterable[OrderedItem]) -> list[dict[str, Any]]:
    return sorted((item.model_dump(exclude={"order_id"}) for item in items), key=repr)
