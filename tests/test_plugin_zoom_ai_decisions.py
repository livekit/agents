"""Hermetic tests for the Zoom AI (Zoom Decisions) DecisionModel, against a local aiohttp server."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from livekit.agents import APIConnectionError, APIConnectOptions, APIError, APIStatusError
from livekit.agents.llm import ChatContext, decisions
from livekit.plugins import zoom_ai

# the fake server and HTTP session are fixtures bound to a single task
pytestmark = [pytest.mark.unit, pytest.mark.no_concurrent]

API_KEY = "test-key"
NO_RETRY = APIConnectOptions(max_retry=0, timeout=5)
ONE_RETRY = APIConnectOptions(max_retry=1, timeout=5)

QUESTIONS = {
    "human": decisions.Probability("The caller asks for a human."),
    "intent": decisions.Choice(
        "What does the caller want?", options={"status": "Order status", "other": ""}
    ),
    "frustration": decisions.Score("How frustrated is the caller?", levels=["Calm", "Angry"]),
}

Reply = Callable[[web.Request], Awaitable[web.Response]]


def answers() -> list[dict[str, Any]]:
    return [
        {"type": "predicate", "name": "human", "probability": 0.9},
        {
            "type": "choice",
            "name": "intent",
            "choice": "status",
            "probabilities": [
                {"value": "status", "probability": 0.8},
                {"value": "other", "probability": 0.2},
            ],
        },
        {
            "type": "score",
            "name": "frustration",
            "score": 0.7,
            "probabilities": [{"value": 0, "probability": 0.3}, {"value": 1, "probability": 0.7}],
        },
    ]


def ok(body_answers: list[dict[str, Any]] | None = None) -> Reply:
    async def reply(request: web.Request) -> web.Response:
        return web.json_response(
            {
                "request_id": "WEB_1",
                "answers": answers() if body_answers is None else body_answers,
                "model": "zoom-decisions",
                "usage": {"input_units": 42, "output_units": 3, "unit_type": "tokens"},
            }
        )

    return reply


@pytest.fixture(autouse=True)
def _no_retry_delay(monkeypatch: pytest.MonkeyPatch) -> None:
    # the framework's first retry otherwise waits a fixed 0.1s
    monkeypatch.setattr(APIConnectOptions, "_interval_for_retry", lambda self, n: 0.0)


class FakeZoomDecisions:
    """Records each decide request and replies with ``reply``."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.reply: Reply = ok()

    async def decide(self, request: web.Request) -> web.Response:
        self.requests.append(
            {"headers": dict(request.headers), "json": await request.json(), "at": time.monotonic()}
        )
        return await self.reply(request)


@pytest.fixture
async def server() -> AsyncIterator[tuple[FakeZoomDecisions, str]]:
    fake = FakeZoomDecisions()
    app = web.Application()
    app.router.add_post("/decisions/decide", fake.decide)
    srv = TestServer(app)
    await srv.start_server()
    try:
        yield fake, str(srv.make_url("/decisions"))
    finally:
        await srv.close()


@pytest.fixture
async def http() -> AsyncIterator[aiohttp.ClientSession]:
    session = aiohttp.ClientSession()
    try:
        yield session
    finally:
        await session.close()


def make_model(http: aiohttp.ClientSession, base_url: str, **kwargs: Any) -> zoom_ai.DecisionModel:
    return zoom_ai.DecisionModel(api_key=API_KEY, base_url=base_url, http_session=http, **kwargs)


def test_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ZOOM_AI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="ZOOM_AI_API_KEY"):
        zoom_ai.DecisionModel()


def test_guardrails_is_rejected() -> None:
    with pytest.raises(ValueError, match="zoom-guardrails"):
        zoom_ai.DecisionModel(api_key=API_KEY, model="zoom-guardrails")


def test_labels_and_capabilities() -> None:
    model = zoom_ai.DecisionModel(api_key=API_KEY, model="zoom-decisions-pro")
    assert model.model == "zoom-decisions-pro"
    assert model.provider == "Zoom"
    assert set(model.capabilities.decision_kinds) == {"probability", "choice", "score"}


async def test_request_matches_the_docs(server, http) -> None:
    fake, base = server
    await make_model(http, base).evaluate(input="I want a person.", decisions=QUESTIONS)

    body = fake.requests[0]["json"]
    assert body == {
        "model": "zoom-decisions",
        "input": "I want a person.",
        "questions": [
            {"type": "predicate", "name": "human", "instructions": "The caller asks for a human."},
            {
                "type": "choice",
                "name": "intent",
                "instructions": "What does the caller want?",
                # an empty description is left out
                "choices": [{"value": "status", "description": "Order status"}, {"value": "other"}],
            },
            {
                "type": "score",
                "name": "frustration",
                "instructions": "How frustrated is the caller?",
                "levels": [{"label": "Calm"}, {"label": "Angry"}],
            },
        ],
    }
    headers = fake.requests[0]["headers"]
    assert headers["Authorization"] == f"Bearer {API_KEY}"
    assert headers["Content-Type"] == "application/json"
    assert headers["User-Agent"].startswith("livekit-plugins-zoom-ai/")
    assert headers["x-zm-trackingid"].startswith("zoom_ai_community_livekit_")


async def test_extra_headers_go_out_below_the_auth_header(server, http) -> None:
    fake, base = server
    fake.reply = ok([answers()[0]])
    await make_model(http, base).evaluate(
        input="hi",
        decisions={"human": QUESTIONS["human"]},
        extra_headers={"X-Trace": "abc", "Authorization": "Bearer other"},
    )

    headers = fake.requests[-1]["headers"]
    assert headers["X-Trace"] == "abc"
    assert headers["Authorization"] == f"Bearer {API_KEY}"


async def test_chat_context_is_sent_as_a_transcript(server, http) -> None:
    fake, base = server
    ctx = ChatContext()
    ctx.add_message(role="assistant", content="How can I help?")
    ctx.add_message(role="user", content="Get me a human.")
    fake.reply = ok([answers()[0]])
    await make_model(http, base).evaluate(input=ctx, decisions={"human": QUESTIONS["human"]})

    assert fake.requests[0]["json"]["input"] == "assistant: How can I help?\nuser: Get me a human."


async def test_all_kinds_decode_with_distributions_and_usage(server, http) -> None:
    _, base = server
    response = await make_model(http, base).evaluate(input="x", decisions=QUESTIONS)

    assert not response.errors
    human, intent, frustration = (response.results[n] for n in ("human", "intent", "frustration"))
    assert isinstance(human, decisions.ProbabilityResult) and human.value == 0.9
    assert isinstance(intent, decisions.ChoiceResult)
    assert intent.value == "status" and intent.probabilities == {"status": 0.8, "other": 0.2}
    assert isinstance(frustration, decisions.ScoreResult)
    assert frustration.value == 0.7
    assert frustration.probabilities == {0: 0.3, 1: 0.7}
    assert frustration.levels == ["Calm", "Angry"]
    assert response.model == "zoom-decisions"
    assert response.provider == "Zoom"
    assert response.request_id == "WEB_1"
    # input_units are tokens; output_units count decisions, so they aren't output tokens
    assert response.input_tokens == 42
    assert response.output_tokens is None


async def test_missing_probabilities_count_as_zero(server, http) -> None:
    fake, base = server
    reply = answers()
    reply[1]["probabilities"] = [{"value": "status", "probability": 1.0}]
    reply[2]["probabilities"] = [{"value": 1, "probability": 1.0}]
    reply[2]["score"] = 1.0
    fake.reply = ok(reply)
    response = await make_model(http, base).evaluate(input="x", decisions=QUESTIONS)

    assert not response.errors
    assert response.results["intent"].probabilities == {"status": 1.0, "other": 0.0}  # type: ignore[union-attr]
    assert response.results["frustration"].probabilities == {0: 0.0, 1: 1.0}  # type: ignore[union-attr]


async def test_refusal_is_reported_without_dropping_the_batch(server, http) -> None:
    fake, base = server
    reply = answers()
    reply[0] = {"type": "refusal", "name": "human"}
    fake.reply = ok(reply)
    response = await make_model(http, base).evaluate(input="x", decisions=QUESTIONS)

    assert response.errors == {"human": "the model declined this decision"}
    assert set(response.results) == {"intent", "frustration"}


async def test_a_missing_answer_is_reported_per_decision(server, http) -> None:
    fake, base = server
    fake.reply = ok(answers()[:2])
    response = await make_model(http, base).evaluate(input="x", decisions=QUESTIONS)

    assert response.errors == {"frustration": "missing decision answer"}


async def test_an_answer_to_an_unknown_decision_rejects_the_batch(server, http) -> None:
    fake, base = server
    fake.reply = ok([*answers(), {"type": "predicate", "name": "extra", "probability": 0.1}])
    with pytest.raises(APIError, match="unknown"):
        await make_model(http, base).evaluate(input="x", decisions=QUESTIONS, conn_options=NO_RETRY)


async def test_invalid_response_is_not_retried(server, http) -> None:
    fake, base = server

    async def not_json(request: web.Request) -> web.Response:
        return web.Response(text="oops")

    fake.reply = not_json
    with pytest.raises(APIError, match="invalid Zoom Decisions response") as excinfo:
        await make_model(http, base).evaluate(
            input="x", decisions=QUESTIONS, conn_options=ONE_RETRY
        )

    assert not excinfo.value.retryable
    assert len(fake.requests) == 1


async def test_bad_request_reports_zoom_reason_without_retry(server, http) -> None:
    fake, base = server

    async def reject(request: web.Request) -> web.Response:
        return web.json_response(
            {"code": 400, "reason": "INVALID_INPUT", "message": "duplicate question name"},
            status=400,
        )

    fake.reply = reject
    with pytest.raises(APIStatusError) as excinfo:
        await make_model(http, base).evaluate(
            input="x", decisions=QUESTIONS, conn_options=ONE_RETRY
        )

    error = excinfo.value
    assert error.status_code == 400 and not error.retryable
    assert "duplicate question name" in error.message
    assert error.request_id == fake.requests[0]["headers"]["x-zm-trackingid"]
    assert len(fake.requests) == 1


async def test_retry_waits_for_retry_after(server, http) -> None:
    fake, base = server
    good = ok()

    async def rate_limited_once(request: web.Request) -> web.Response:
        if len(fake.requests) == 1:
            return web.json_response(
                {"code": 429, "reason": "RATE_LIMIT_EXCEEDED", "message": "slow down"},
                status=429,
                headers={"Retry-After": "1"},
            )
        return await good(request)

    fake.reply = rate_limited_once
    response = await make_model(http, base).evaluate(
        input="x", decisions=QUESTIONS, conn_options=ONE_RETRY
    )

    assert not response.errors
    assert len(fake.requests) == 2
    assert fake.requests[1]["at"] - fake.requests[0]["at"] >= 0.9


async def test_retry_after_longer_than_the_timeout_stops_the_retries(server, http) -> None:
    fake, base = server

    async def unavailable(request: web.Request) -> web.Response:
        return web.json_response(
            {"code": 503, "reason": "SERVICE_UNAVAILABLE", "metadata": {"retry_after_seconds": 30}},
            status=503,
        )

    fake.reply = unavailable
    started = time.monotonic()
    with pytest.raises(APIConnectionError, match="asked to wait") as excinfo:
        await make_model(http, base).evaluate(
            input="x", decisions=QUESTIONS, conn_options=ONE_RETRY
        )

    assert not excinfo.value.retryable
    assert len(fake.requests) == 1
    assert time.monotonic() - started < 2
