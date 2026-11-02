"""Prism response metrics through the SDK and standard LiveKit events."""

from __future__ import annotations

import asyncio
import base64
from types import SimpleNamespace

import pytest
from rimelabs_sdk import RimeResourceLimitError

from livekit.agents import Agent, AgentSession
from livekit.agents.metrics import RealtimeModelMetrics
from livekit.plugins.rime.realtime import realtime_model

from .test_plugin_rime import peer_model, wait_until

pytestmark = pytest.mark.unit

USAGE = {"input_tokens": 12, "output_tokens": 8, "total_tokens": 20, "context_window": 100}


async def finish(peer, response_id, *, status="completed", usage=USAGE):
    response = {"id": response_id, "status": status}
    if usage is not None:
        response["usage"] = usage
    await peer.emit("response.done", response=response)


@pytest.mark.parametrize("playback", ["auto", "discard"])
async def test_response_metrics_measure_first_audio_and_monotonic_duration(monkeypatch, playback):
    clock = SimpleNamespace(now=10.0, wall=1000.0)
    monkeypatch.setattr(
        realtime_model,
        "time",
        SimpleNamespace(time=lambda: clock.wall, monotonic=lambda: clock.now),
    )
    async with peer_model(playback=playback) as (peer, model, _):
        session = model.session()
        metrics, events = [], []
        session.on("metrics_collected", metrics.append)
        session.on("rime_server_event_received", events.append)
        await session.update_instructions("test")
        response = await peer.start()
        await wait_until(lambda: any(e.payload.kind == "response.started" for e in events))
        await peer.emit(
            "response.output_item.added",
            response_id=response,
            output_index=0,
            item={"id": "message", "type": "message"},
        )
        fields = {
            "response_id": response,
            "output_index": 0,
            "item_id": "message",
            "content_index": 0,
        }
        clock.now = 11.0
        await peer.emit("response.text.delta", **fields, delta="Hello")
        await peer.emit("response.audio.delta", **fields, delta="")
        await wait_until(lambda: any(e.payload.kind == "audio.delta" for e in events))
        clock.now = 12.0
        await peer.emit("response.audio.delta", **fields, delta=base64.b64encode(b"\0\0").decode())
        await wait_until(lambda: sum(e.payload.kind == "audio.delta" for e in events) == 2)
        clock.now = 13.0
        await peer.emit("response.audio.delta", **fields, delta=base64.b64encode(b"\0\0").decode())
        await wait_until(lambda: sum(e.payload.kind == "audio.delta" for e in events) == 3)
        clock.now, clock.wall = 14.0, 900.0
        await finish(peer, response)
        await wait_until(lambda: len(metrics) == 1)
        metric = metrics[0]
        assert isinstance(metric, RealtimeModelMetrics)
        assert metric.request_id == response
        assert metric.timestamp == 1000.0
        assert metric.duration == 4.0
        assert metric.ttft == 2.0
        assert (metric.input_tokens, metric.output_tokens, metric.total_tokens) == (12, 8, 20)
        assert metric.tokens_per_second == 2.0
        assert not metric.cancelled
        assert metric.label == model.label
        assert metric.metadata.model_name == "prism"
        assert metric.metadata.model_provider == "rime"
        assert metric.input_token_details == RealtimeModelMetrics.InputTokenDetails()
        assert metric.output_token_details == RealtimeModelMetrics.OutputTokenDetails()
        await finish(peer, response)
        # A later response is a processing barrier for the repeated terminal event.
        later = await peer.start()
        await finish(peer, later, usage=None)
        await wait_until(lambda: len(metrics) == 2)
        assert [m.request_id for m in metrics] == [response, later]
        assert metrics[1].tokens_per_second == 0


@pytest.mark.parametrize("status", ["completed", "cancelled", "failed"])
async def test_no_audio_or_usage_has_defined_metrics(status):
    async with peer_model() as (peer, model, _):
        session = model.session()
        metrics = []
        session.on("metrics_collected", metrics.append)
        await session.update_instructions("test")
        response = await peer.start()
        await finish(peer, response, status=status, usage=None)
        await wait_until(lambda: len(metrics) == 1)
        metric = metrics[0]
        assert metric.ttft == -1
        assert metric.duration >= 0
        assert metric.cancelled == (status == "cancelled")
        assert (metric.input_tokens, metric.output_tokens, metric.total_tokens) == (0, 0, 0)
        assert metric.tokens_per_second == 0


async def test_agent_session_forwards_metrics_and_accumulates_usage():
    async with peer_model(playback="discard") as (peer, model, _):
        async with AgentSession(llm=model) as session:
            events = []
            session.on("metrics_collected", events.append)
            await session.start(Agent(instructions="test"))
            speech = session.generate_reply(user_input="Hello")
            response = await peer.start(await peer.next())
            await peer.message(response)
            await finish(peer, response)
            await wait_until(lambda: len(events) == 1)
            await asyncio.wait_for(speech.wait_for_playout(), 3)
            assert events[0].metrics.request_id == response
            assert events[0].metrics.total_tokens == 20
            usage = session.usage.model_usage[0]
            assert usage.input_tokens == 12
            assert usage.output_tokens == 8


async def test_interrupted_generation_retains_late_terminal_metrics(monkeypatch):
    # Exercise adapter eviction within the older SDK's larger reference window.
    monkeypatch.setattr(realtime_model, "_GENERATION_LIMIT", 3)
    async with peer_model() as (peer, model, _):
        session = model.session()
        metrics, events = [], []
        session.on("metrics_collected", metrics.append)
        session.on("rime_server_event_received", events.append)
        await session.update_instructions("test")
        first = await peer.start()
        await wait_until(lambda: any(e.payload.kind == "response.started" for e in events))
        peer.cancel_release.clear()
        session.interrupt()
        await asyncio.wait_for(peer.cancel_received.wait(), 3)
        assert not metrics
        try:
            responses = []
            for number in range(4):
                response = await peer.start()
                responses.append(response)
                await finish(peer, response)
                await wait_until(lambda number=number: len(metrics) == number + 1)
            await finish(peer, first, status="cancelled")
            await wait_until(lambda: len(metrics) == 5)
            assert metrics[-1].request_id == first
            assert metrics[-1].cancelled
            assert metrics[-1].total_tokens == 20
        finally:
            peer.cancel_release.set()
        # Process the repeated cancellation terminal event before checking metrics.
        later = await peer.start()
        await finish(peer, later)
        await wait_until(lambda: len(metrics) == 6)
        assert [m.request_id for m in metrics] == [*responses, first, later]


async def test_generation_limit_refuses_to_evict_unfinished_responses(monkeypatch):
    monkeypatch.setattr(realtime_model, "_GENERATION_LIMIT", 3)
    async with peer_model() as (peer, model, _):
        session = model.session()
        errors, metrics = [], []
        session.on("error", errors.append)
        session.on("metrics_collected", metrics.append)
        await session.update_instructions("test")
        for _ in range(4):
            await peer.start()
        await wait_until(lambda: errors)
        with pytest.raises(RimeResourceLimitError, match="generation limit"):
            await session.update_instructions("test")
        assert not metrics


async def test_terminal_duplicate_after_generation_eviction_is_ignored(monkeypatch):
    monkeypatch.setattr(realtime_model, "_GENERATION_LIMIT", 3)
    async with peer_model() as (peer, model, _):
        session = model.session()
        metrics = []
        session.on("metrics_collected", metrics.append)
        await session.update_instructions("test")
        responses = []
        for number in range(4):
            responses.append(await peer.start())
            await finish(peer, responses[-1])
            await wait_until(lambda number=number: len(metrics) == number + 1)
        await finish(peer, responses[0])
        responses.append(await peer.start())
        await finish(peer, responses[-1])
        await wait_until(lambda: len(metrics) == 5)
        assert [metric.request_id for metric in metrics] == responses
