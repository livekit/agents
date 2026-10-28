"""Airline support payment guards and persisted customer follow-up on scripted models."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import pytest
from pydantic import BaseModel

from livekit.agents import Agent, InputRequired
from livekit.agents.llm import ChatContext, FunctionToolCall, LLMStream, Tool
from livekit.agents.storage.context_store import ContextStore
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions

from .fake_llm import FakeLLM, FakeLLMResponse, FakeLLMStream

pytestmark = pytest.mark.unit

EXAMPLE = Path(__file__).parents[1] / "examples" / "airline-support"
sys.path.insert(0, str(EXAMPLE))

# the example builds its storage at import; a placeholder project, so the storage tests that run
# later in the session still skip when no real one is configured
_PLACEHOLDER = {
    "LIVEKIT_API_KEY": "devkey",
    "LIVEKIT_API_SECRET": "secretsecretsecretsecretsecretsecret",
    "LIVEKIT_URL": "ws://localhost:9000",
}
with mock.patch.dict(os.environ, {k: os.environ.get(k, v) for k, v in _PLACEHOLDER.items()}):
    import backend_agent as sa  # noqa: E402
    from skyline import Skyline  # noqa: E402

APPROVED = sa.Verdict(approved=True)
DELAY = sa.Email(sender="ana.garcia@example.com", subject="SK204 late", body="Booking SKY4F2.")
LONG_HAUL = sa.Email(sender="marcus.lee@example.com", subject="SK880 late", body="SKY7Q9.")
NO_REF = sa.Email(sender="tom.okafor@example.com", subject="Lisbon", body="You cancelled SK517.")
SPAM = sa.Email(sender="deals@cheap-meds.example", subject="You WON", body="Click here.")


class Scripted(FakeLLM):
    """Answers each call with the next output: text as content, a model as its final_result."""

    def __init__(self, *outputs: str | BaseModel) -> None:
        super().__init__()
        self._outputs = list(outputs)

    def chat(
        self,
        *,
        chat_ctx: ChatContext,
        tools: list[Tool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        **kwargs: Any,
    ) -> LLMStream:
        out = self._outputs.pop(0)
        calls = []
        if isinstance(out, BaseModel):
            calls = [
                FunctionToolCall(name="final_result", arguments=out.model_dump_json(), call_id="c")
            ]
        self._fake_response_map = {
            "": FakeLLMResponse(
                input="",
                content=out if isinstance(out, str) else "",
                ttft=0,
                duration=0,
                tool_calls=calls,
            )
        }
        return _AnyInput(self, chat_ctx=chat_ctx, tools=tools or [], conn_options=conn_options)


class _AnyInput(FakeLLMStream):
    def _get_index_text(self) -> str:
        return ""


@pytest.fixture
def skyline(monkeypatch: pytest.MonkeyPatch) -> Skyline:
    seeded = Skyline.seeded()
    monkeypatch.setattr(sa, "skyline", seeded)
    return seeded


def script(
    triage: sa.Classification,
    drafts: tuple[str, ...] = (),
    verdicts: tuple[sa.Verdict, ...] = (),
) -> None:
    sa.classifier = Agent(
        instructions="classify", llm=Scripted(triage), output_type=sa.Classification
    )
    sa.writer = Agent(instructions="draft", llm=Scripted(*drafts), output_type=str)
    sa.reviewer = Agent(instructions="review", llm=Scripted(*verdicts), output_type=sa.Verdict)


async def start(email: sa.Email) -> sa.TriageResult | InputRequired:
    app = sa.SupportContext()
    store = ContextStore()
    await store.load_context(app)
    return await sa.triage(SimpleNamespace(context=app, store=store), email)


async def test_eu261_delay_is_paid_after_an_approved_reply(skyline: Skyline) -> None:
    script(
        sa.Classification(kind="delay_or_cancellation", booking_ref="SKY4F2"),
        ("EUR 250.",),
        (APPROVED,),
    )
    out = await start(DELAY)
    assert (out.action, out.team, out.paid_eur, out.reply_draft) == (
        "resolved",
        "claims",
        250,
        "EUR 250.",
    )
    assert [(p.booking_ref, p.eur) for p in skyline.payments] == [("SKY4F2", 250)]


async def test_an_earlier_payment_is_taken_off(skyline: Skyline) -> None:
    skyline.pay("SKY4F2", 250, "paid before")
    script(
        sa.Classification(kind="delay_or_cancellation", booking_ref="SKY4F2"),
        ("Paid.",),
        (APPROVED,),
    )
    out = await start(DELAY)
    assert (out.action, out.paid_eur) == ("resolved", 0)
    assert len(skyline.payments) == 1


async def test_over_the_limit_goes_to_a_person_unpaid(skyline: Skyline) -> None:
    script(
        sa.Classification(kind="delay_or_cancellation", booking_ref="SKY7Q9"),
        ("EUR 600.",),
        (APPROVED,),
    )
    out = await start(LONG_HAUL)
    assert (out.action, out.priority, out.paid_eur) == ("to_human", "high", 0)
    assert skyline.payments == []


async def test_a_reply_failing_review_twice_goes_to_a_person_unpaid(skyline: Skyline) -> None:
    script(
        sa.Classification(kind="delay_or_cancellation", booking_ref="SKY4F2"),
        ("EUR 600.", "EUR 400."),
        (
            sa.Verdict(approved=False, objection="not 600"),
            sa.Verdict(approved=False, objection="not 400"),
        ),
    )
    out = await start(DELAY)
    assert (out.action, out.paid_eur) == ("to_human", 0)
    assert skyline.payments == []


async def test_spam_closes_without_a_reply(skyline: Skyline) -> None:
    script(sa.Classification(kind="spam"))
    out = await start(SPAM)
    assert (out.action, out.reply_draft, out.paid_eur) == ("close", None, 0)


async def test_a_missing_booking_reference_is_asked_and_the_reply_continues(
    skyline: Skyline,
) -> None:
    script(sa.Classification(kind="delay_or_cancellation"), ("EUR 400.",), (APPROVED,))
    app = sa.SupportContext()
    store = ContextStore()
    await store.load_context(app)
    ctx = SimpleNamespace(context=app, store=store)
    asked = await sa.triage(ctx, NO_REF)
    assert isinstance(asked, InputRequired) and "booking reference" in asked.question

    async def reply(reference: str) -> sa.TriageResult | InputRequired:
        nonlocal ctx
        restored = sa.SupportContext()
        restored_store = ContextStore(scopes=[write.scope for write in ctx.store._scope_writes])
        await restored_store.load_context(restored)
        ctx = SimpleNamespace(context=restored, store=restored_store)
        return await sa.continue_request(ctx, booking_ref=reference)

    someone_elses = await reply("SKY4F2")
    assert (
        isinstance(someone_elses, InputRequired) and "booking reference" in someone_elses.question
    )
    done = await reply("SKY5C1")
    assert isinstance(done, sa.TriageResult) and done.paid_eur == 400
    assert [(p.booking_ref, p.eur) for p in skyline.payments] == [("SKY5C1", 400)]
