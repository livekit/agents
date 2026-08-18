# Copyright 2023 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import asyncio
import time
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from floe_guard import hosted_enforcement_available, hosted_remaining_usd, turn_cost

from livekit.agents.metrics import LLMModelUsage

from .log import logger
from .services import FLOE_PROVIDER

if TYPE_CHECKING:
    from livekit.agents import AgentSession, SessionUsageUpdatedEvent

# How long a fetched hosted-budget value is reused before a background refresh.
_BUDGET_TTL = 30.0


def enable_cost_receipts(session: AgentSession[Any], *, show_budget: bool = True) -> None:
    """Log a one-line Floe cost receipt after every Floe-routed turn.

    Zero config: attach it to a session and each turn that ran through this
    plugin's ``floe.LLM`` prints what that turn cost, e.g.::

        floe · gpt-4o · $0.0064 est · left $12.34

    Cost is always shown (priced locally from the bundled cost map — free,
    offline, no account). The budget half (``left $…``) is added when a
    ``FLOE_API_KEY`` is set.

    The hosted budget read is a blocking network call, so it is **never** made
    from this synchronous handler — that would stall the agent's audio/speech
    loop. Instead it runs off-loop (``asyncio.to_thread``), throttled to once per
    ``_BUDGET_TTL`` seconds; each receipt uses the last cached value (a one-turn
    refresh lag is fine). A failed read drops the budget (the cost still prints,
    without ``left $…``). If no event loop is running, the budget is skipped
    entirely and cost is still shown.

    Only ``provider="floe"`` usage is counted, so other providers or plugins in a
    mixed session don't produce phantom receipts. ``session_usage_updated`` is
    cumulative, so each turn's cost is the per-model delta since the last event.

    Args:
        session: The ``AgentSession`` to observe.
        show_budget: When ``True`` (default) and a Floe key is present, append the
            remaining hosted budget to each receipt.
    """
    last_tokens: dict[str, tuple[int, int]] = {}
    remaining_usd: float | None = None
    fetched_at: float = 0.0
    bg_tasks: set[asyncio.Task[None]] = set()

    async def _refresh_budget() -> None:
        nonlocal remaining_usd
        try:
            remaining_usd = await asyncio.to_thread(hosted_remaining_usd)
        except Exception:  # noqa: BLE001 - a budget read must never break a call
            remaining_usd = None  # drop a stale value on failure
            logger.debug("floe: budget read failed; showing cost without budget", exc_info=True)

    def _on_usage_updated(ev: SessionUsageUpdatedEvent) -> None:
        nonlocal fetched_at

        # Refresh the hosted budget off-loop at most once per TTL — never block
        # this handler on the network. The receipt below uses the cached value.
        if show_budget and hosted_enforcement_available():
            now = time.monotonic()
            if now - fetched_at >= _BUDGET_TTL:
                fetched_at = now  # throttle regardless of outcome
                try:
                    task = asyncio.get_running_loop().create_task(_refresh_budget())
                except RuntimeError:
                    pass  # no running loop — skip the budget, never raise
                else:
                    bg_tasks.add(task)
                    task.add_done_callback(bg_tasks.discard)

        for mu in ev.usage.model_usage:
            if not isinstance(mu, LLMModelUsage):
                continue
            if mu.provider != FLOE_PROVIDER:
                continue  # only receipt usage routed through this plugin's floe.LLM

            prev_in, prev_out = last_tokens.get(mu.model, (0, 0))
            delta_in = mu.input_tokens - prev_in
            delta_out = mu.output_tokens - prev_out
            last_tokens[mu.model] = (mu.input_tokens, mu.output_tokens)
            if delta_in <= 0 and delta_out <= 0:
                continue  # nothing new served for this model this turn

            # Price on the full id (a bare name can be unpriceable); show the
            # short model in the receipt.
            cost = turn_cost(mu.model, delta_in, delta_out, remaining_usd=remaining_usd)
            if cost is None:
                continue  # fail closed — never a fabricated $0
            logger.info(replace(cost, model=mu.model.split("/")[-1]).format())

    session.on("session_usage_updated", _on_usage_updated)
