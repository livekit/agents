"""Unit tests for livekit.plugins.hume.DecisionModel.

Covers the multipart request (part order, question encoding, audio conversion, input
as context), answer decoding, the run id, local validation, and error mapping and
retries for Hume's error body.
"""

from __future__ import annotations

import contextlib
import io
import json
import wave
from typing import Any

import aiohttp
import numpy as np
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from livekit import rtc
from livekit.agents import APIStatusError
from livekit.agents.llm import ChatContext, decisions
from livekit.agents.types import APIConnectOptions
from livekit.plugins.hume import DecisionModel

pytestmark = pytest.mark.unit

QUESTIONS = {
    "frustrated": decisions.Probability("Does the caller sound frustrated?"),
    "tone": decisions.Choice(
        "Which best describes the agent's tone?",
        options={"warm": "Warm and friendly", "curt": ""},
    ),
    "naturalness": decisions.Score(
        "How natural does the agent sound?", levels=["Not at all", "Somewhat", "Extremely"]
    ),
}


def response_body() -> dict[str, Any]:
    return {
        "run_id": "019a1f3c-7d2e-7b41-9c58-2e6f4a1d8b03",
        "answers": {
            "frustrated": {"type": "noul", "noul": 0.1274},
            "tone": {
                "type": "choice",
                "choice": "warm",
                "confidence": 0.3,
                "probabilities": {"warm": 0.65, "curt": 0.35},
            },
            "naturalness": {
                "type": "score",
                "score": 1.4,
                "confidence": 0.42,
                "probabilities": {"0": 0.1, "1": 0.4, "2": 0.5},
                "legend": {"0": "Not at all", "1": "Somewhat", "2": "Extremely"},
            },
        },
        "received": {"input_tokens": 11234},
    }


def _frames(
    seconds: float, *, sample_rate: int = 48000, num_channels: int = 1
) -> list[rtc.AudioFrame]:
    samples = int(seconds * sample_rate)
    data = np.zeros(samples * num_channels, dtype=np.int16)
    if num_channels == 2:
        data[0::2] = 1000
        data[1::2] = 3000
    return [
        rtc.AudioFrame(
            data=data.tobytes(),
            sample_rate=sample_rate,
            num_channels=num_channels,
            samples_per_channel=samples,
        )
    ]


@contextlib.asynccontextmanager
async def _hume(handler):  # type: ignore[no-untyped-def]
    app = web.Application(client_max_size=30 * 1024 * 1024)
    app.router.add_post("/v0/decisions/file", handler)
    server = TestServer(app)
    await server.start_server()
    session = aiohttp.ClientSession()
    try:
        yield str(server.make_url("")).rstrip("/"), session
    finally:
        await session.close()
        await server.close()


def _handler(
    responses: list[tuple[int, dict[str, Any], dict[str, str]]],
    captured: dict[str, Any] | None = None,
):  # type: ignore[no-untyped-def]
    calls = iter(responses)

    async def handler(request: web.Request) -> web.Response:
        if captured is not None:
            captured["headers"] = dict(request.headers)
            reader = await request.multipart()
            parts = []
            async for part in reader:
                assert isinstance(part, aiohttp.BodyPartReader)
                parts.append((part.name, part.filename, part.headers, await part.read()))
            captured["parts"] = parts
            captured["calls"] = captured.get("calls", 0) + 1
        status, body, headers = next(calls)
        return web.json_response(body, status=status, headers=headers)

    return handler


def _model(base_url: str, session: aiohttp.ClientSession) -> DecisionModel:
    return DecisionModel(api_key="test-key", base_url=base_url, http_session=session)


class TestConstruction:
    def test_env_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HUME_API_KEY", "env-key")
        assert DecisionModel()._api_key == "env-key"

    def test_missing_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("HUME_API_KEY", raising=False)
        with pytest.raises(ValueError, match="HUME_API_KEY"):
            DecisionModel()

    def test_identity(self) -> None:
        model = DecisionModel(api_key="k")
        assert model.provider == "Hume"
        assert model.capabilities.decision_kinds == ["probability", "choice", "score"]
        assert model.capabilities.audio_input


async def test_request_sends_questions_then_16k_mono_wav() -> None:
    captured: dict[str, Any] = {}
    handler = _handler([(200, response_body(), {"X-Request-Id": "req_1"})], captured)
    async with _hume(handler) as (url, session):
        response = await _model(url, session).evaluate(
            audio=_frames(6, sample_rate=48000, num_channels=2),
            decisions=QUESTIONS,
            input="A support call.",
        )

    assert captured["headers"]["X-Hume-Api-Key"] == "test-key"
    assert captured["headers"]["X-Hume-Client-Name"] == "livekit"

    (req_name, _, req_headers, req_body), (file_name, filename, file_headers, file_body) = captured[
        "parts"
    ]
    assert req_name == "request"
    assert req_headers["Content-Type"] == "application/json"
    assert json.loads(req_body) == {
        "context": "A support call.",
        "questions": {
            "frustrated": {"type": "noul", "instructions": "Does the caller sound frustrated?"},
            "tone": {
                "type": "choice",
                "instructions": "Which best describes the agent's tone?",
                "criteria": [
                    {"name": "warm", "description": "Warm and friendly"},
                    {"name": "curt"},
                ],
            },
            "naturalness": {
                "type": "score",
                "instructions": "How natural does the agent sound?",
                "criteria": ["Not at all", "Somewhat", "Extremely"],
            },
        },
    }

    assert file_name == "file"
    assert filename == "audio.wav"
    assert file_headers["Content-Type"] == "audio/wav"
    with wave.open(io.BytesIO(file_body)) as wav:
        assert wav.getnchannels() == 1
        assert wav.getframerate() == 16000
        assert wav.getnframes() == pytest.approx(6 * 16000, abs=160)
        samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16)
    # The two channels are averaged, not one of them dropped.
    assert abs(int(np.median(samples)) - 2000) <= 2

    assert response.results["frustrated"].value == 0.1274
    assert response.results["tone"].value == "warm"
    assert response.results["tone"].provider_data["confidence"] == 0.3
    score = response.results["naturalness"]
    assert score.kind == "score"
    assert score.value == 1.4
    assert score.probabilities == {0: 0.1, 1: 0.4, 2: 0.5}
    assert score.levels == ["Not at all", "Somewhat", "Extremely"]
    assert response.errors == {}
    assert response.provider == "Hume"
    assert response.request_id == "019a1f3c-7d2e-7b41-9c58-2e6f4a1d8b03"
    assert response.input_tokens == 11234


async def test_empty_input_sends_no_context_and_chat_context_is_a_transcript() -> None:
    captured: dict[str, Any] = {}
    handler = _handler([(200, response_body(), {}), (200, response_body(), {})], captured)
    chat_ctx = ChatContext.empty()
    chat_ctx.add_message(role="user", content="I've been on hold for an hour.")
    async with _hume(handler) as (url, session):
        model = _model(url, session)
        await model.evaluate(audio=_frames(5, sample_rate=16000), decisions=QUESTIONS)
        assert "context" not in json.loads(captured["parts"][0][3])
        await model.evaluate(audio=_frames(5), input=chat_ctx, decisions=QUESTIONS)
        assert json.loads(captured["parts"][0][3])["context"] == (
            "user: I've been on hold for an hour."
        )


async def test_text_only_evaluate_is_rejected() -> None:
    model = DecisionModel(api_key="k")
    with pytest.raises(ValueError, match="require audio"):
        await model.evaluate(input="The caller asked for a refund.", decisions=QUESTIONS)


async def test_input_longer_than_hume_context_is_rejected_locally() -> None:
    model = DecisionModel(api_key="k")
    with pytest.raises(ValueError, match="1000 characters"):
        await model.evaluate(audio=_frames(5), input="x" * 1001, decisions=QUESTIONS)


@pytest.mark.parametrize("seconds", [4.9, 300.5])
async def test_audio_outside_duration_limits_is_rejected_locally(seconds: float) -> None:
    model = DecisionModel(api_key="k")
    with pytest.raises(ValueError, match="5 to 300 seconds"):
        await model.evaluate(audio=_frames(seconds, sample_rate=8000), decisions=QUESTIONS)


async def test_unknown_answer_type_is_a_per_decision_error() -> None:
    body = response_body()
    body["answers"]["frustrated"] = {"type": "something_new", "value": 1}
    async with _hume(_handler([(200, body, {})])) as (url, session):
        response = await _model(url, session).evaluate(audio=_frames(5), decisions=QUESTIONS)
    assert "frustrated" not in response.results
    assert response.errors == {"frustrated": "invalid Hume decision answer"}
    assert set(response.results) == {"tone", "naturalness"}


async def test_client_error_is_not_retried_and_carries_hume_detail() -> None:
    captured: dict[str, Any] = {}
    error = {
        "type": "error",
        "code": "question_invalid",
        "message": "score criteria must define at least 2 levels",
        "param": "/questions/naturalness/criteria",
        "request_id": "req_01JB8XA3C5RD",
    }
    async with _hume(_handler([(400, error, {})], captured)) as (url, session):
        with pytest.raises(APIStatusError) as exc_info:
            await _model(url, session).evaluate(audio=_frames(5), decisions=QUESTIONS)

    assert captured["calls"] == 1
    exc = exc_info.value
    assert exc.status_code == 400
    assert not exc.retryable
    assert exc.request_id == "req_01JB8XA3C5RD"
    assert "question_invalid" in exc.message
    assert "/questions/naturalness/criteria" in exc.message


async def test_rate_limit_and_busy_are_retried() -> None:
    captured: dict[str, Any] = {}
    responses = [
        (429, {"message": "API rate limit exceeded"}, {"Retry-After": "1"}),
        (503, {"type": "error", "code": "server_busy", "message": "busy"}, {}),
        (200, response_body(), {}),
    ]
    async with _hume(_handler(responses, captured)) as (url, session):
        response = await _model(url, session).evaluate(
            audio=_frames(5),
            decisions=QUESTIONS,
            conn_options=APIConnectOptions(max_retry=2, retry_interval=0, timeout=10),
        )
    assert captured["calls"] == 3
    assert response.results["frustrated"].value == 0.1274
