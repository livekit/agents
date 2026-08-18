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

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from floe_guard import price_tokens, resolve_price

from livekit.agents.metrics import LLMModelUsage

from .log import logger

if TYPE_CHECKING:
    from livekit.agents import AgentSession, SessionUsageUpdatedEvent
    from livekit.agents.metrics import AgentSessionUsage


@dataclass(frozen=True)
class FloeModelCost:
    """Floe-priced usage for a single served model.

    Token counts are LiveKit's own accounting; ``estimated_usd`` is the local
    USD estimate for those tokens via Floe's cost map, or ``None`` when the
    model is not priceable.
    """

    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    estimated_usd: float | None
    priced: bool


@dataclass(frozen=True)
class FloeUsageReconciliation:
    """A reconciliation snapshot across every LLM model a session served.

    ``per_model`` is the Floe-priced breakdown, ``total_estimated_usd`` sums the
    priceable entries, and ``unpriced_models`` lists any the cost map could not
    price (their tokens are excluded from the total).
    """

    per_model: list[FloeModelCost]
    total_estimated_usd: float
    unpriced_models: list[str] = field(default_factory=list)


def _display_provider_model(mu: LLMModelUsage) -> tuple[str, str]:
    """Report-friendly ``(provider, model)``.

    Floe ids are ``'<provider>/<model>'``; split them for display. Falls back to
    the runtime ``provider``/``model`` fields for a bare id. Display only —
    pricing always uses the full ``mu.model`` id.
    """
    if "/" in mu.model:
        provider, _, model = mu.model.partition("/")
        return provider, model
    return mu.provider, mu.model


class FloeUsageReconciler:
    """Reconcile LiveKit-reported token usage against Floe pricing.

    This demonstrates usage reconciliation: LiveKit's ``session_usage_updated``
    metrics on one side (tokens the runtime actually served, per model) and the
    Floe cost map on the other (what those tokens are worth). Wire it to an
    ``AgentSession`` with :meth:`attach`; it tracks the latest cumulative usage
    and :meth:`summary` prices every served LLM model on demand.

    The estimate is local and advisory. Floe's billed amount is authoritative;
    a divergence between the two is the signal worth acting on.

    Unlike a single-model meter, this reads the model id off each usage entry,
    so a session that swaps or fans out across models is priced correctly.
    """

    def __init__(self) -> None:
        """Create a reconciler. No model needs to be named up front."""
        self._latest: AgentSessionUsage | None = None

    def attach(self, session: AgentSession[Any]) -> None:
        """Subscribe to a session's ``session_usage_updated`` event.

        The latest cumulative usage payload is stored for :meth:`summary`.

        Args:
            session: The ``AgentSession`` to observe.
        """
        session.on("session_usage_updated", self._on_usage_updated)

    def _on_usage_updated(self, ev: SessionUsageUpdatedEvent) -> None:
        self._latest = ev.usage

    def summary(self) -> FloeUsageReconciliation:
        """Price every served LLM model from the latest usage payload.

        Returns an empty snapshot (zero total, empty lists) if no usage has
        been reported yet. Models the Floe cost map cannot price are listed in
        ``unpriced_models`` and excluded from ``total_estimated_usd``.
        """
        per_model: list[FloeModelCost] = []
        unpriced_models: list[str] = []
        total = 0.0

        usage = self._latest
        if usage is None:
            return FloeUsageReconciliation(per_model=[], total_estimated_usd=0.0)

        for mu in usage.model_usage:
            if not isinstance(mu, LLMModelUsage):
                continue
            provider, model = _display_provider_model(mu)  # display only
            priced = resolve_price(mu.model)  # pricing uses the full id
            if priced is None:
                logger.warning("no Floe price for model %r; excluded from total", mu.model)
                unpriced_models.append(mu.model)
                per_model.append(
                    FloeModelCost(
                        model=model,
                        provider=provider,
                        input_tokens=mu.input_tokens,
                        output_tokens=mu.output_tokens,
                        estimated_usd=None,
                        priced=False,
                    )
                )
                continue
            usd = price_tokens(priced, mu.input_tokens, mu.output_tokens)
            total += usd
            per_model.append(
                FloeModelCost(
                    model=model,
                    provider=provider,
                    input_tokens=mu.input_tokens,
                    output_tokens=mu.output_tokens,
                    estimated_usd=usd,
                    priced=True,
                )
            )

        return FloeUsageReconciliation(
            per_model=per_model,
            total_estimated_usd=total,
            unpriced_models=unpriced_models,
        )
