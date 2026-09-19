from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from livekit.agents.telemetry import set_tracer_provider, tracer
from livekit.agents.voice.avatar import _types

pytestmark = [pytest.mark.unit, pytest.mark.no_concurrent]


@pytest.fixture
def exporter() -> Iterator[InMemorySpanExporter]:
    original = tracer._tracer_provider
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    set_tracer_provider(provider)
    try:
        yield exporter
    finally:
        set_tracer_provider(original)
        provider.shutdown()


class Avatar(_types.AvatarSession):
    @property
    def avatar_identity(self) -> str:
        return "avatar-test"

    @property
    def provider(self) -> str:
        return "test-provider"


@pytest.mark.parametrize("outcome", ["completed", "failed", "cancelled"])
async def test_join_trace(exporter, monkeypatch, outcome):
    import asyncio

    avatar = Avatar()
    avatar._room = MagicMock()
    avatar._agent_session = MagicMock()
    metrics = []
    avatar.on("metrics_collected", metrics.append)
    participant = AsyncMock()
    error = {"failed": RuntimeError("join failed"), "cancelled": asyncio.CancelledError()}
    publication = AsyncMock(side_effect=error.get(outcome))
    monkeypatch.setattr(_types.utils, "wait_for_participant", participant)
    monkeypatch.setattr(_types.utils, "wait_for_track_publication", publication)
    monkeypatch.setattr(_types, "time", SimpleNamespace(time=MagicMock(side_effect=[100, 103.25])))

    with tracer.start_as_current_span("job_entrypoint") as parent:
        # Task creation must retain the application/job trace before AgentSession.start.
        task = asyncio.create_task(avatar._wait_avatar_join())
        if outcome == "completed":
            await task
        else:
            with pytest.raises(type(error[outcome])):
                await task

    span = next(s for s in exporter.get_finished_spans() if s.name == "avatar_join")
    assert span.parent.span_id == parent.get_span_context().span_id
    assert span.attributes["lk.avatar.provider"] == "test-provider"
    assert span.attributes["lk.avatar.join_outcome"] == outcome
    participant.assert_awaited_once()
    publication.assert_awaited_once()
    if outcome == "completed":
        assert len(metrics) == 1
        assert span.attributes["lk.avatar.join_latency"] == 3.25
        assert metrics[0].session_started_time == 100
        assert metrics[0].avatar_joined_time == 103.25
    else:
        assert not metrics
        assert "lk.avatar.join_latency" not in span.attributes
    assert span.status.status_code == (
        StatusCode.ERROR if outcome == "failed" else StatusCode.UNSET
    )
