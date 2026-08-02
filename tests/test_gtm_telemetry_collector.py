from __future__ import annotations

import gc
from unittest.mock import MagicMock, patch

import pytest

from livekit.agents import Agent, AgentSession, RunContext, function_tool
from livekit.agents.beta.gtm_telemetry.collector import PostCallTelemetryCollector
from livekit.agents.llm import FunctionToolCall

from .fake_llm import FakeLLM, FakeLLMResponse

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]

_COLLECTOR_MOD = "livekit.agents.beta.gtm_telemetry.collector"


class _EchoAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="echo agent")

    @function_tool
    async def lookup(self, ctx: RunContext) -> str:
        """Look something up."""
        return "42"

    @function_tool
    async def boom(self, ctx: RunContext) -> str:
        """Always fails."""
        raise ValueError("kaboom")


def _mock_job_ctx() -> MagicMock:
    ctx = MagicMock()
    ctx.job.id = "job-1"
    ctx.job.room.sid = "room-sid-1"
    ctx.job.room.name = "room-1"
    ctx.local_participant_identity = "agent-identity"
    return ctx


# --- attach / detach lifecycle -----------------------------------------------------


@pytest.mark.asyncio
async def test_attach_idempotent_same_session_is_noop():
    collector = PostCallTelemetryCollector()
    async with AgentSession(llm=FakeLLM()) as sess:
        collector.attach(sess)
        collector.attach(sess)
        assert collector.attached


@pytest.mark.asyncio
async def test_attach_second_different_session_raises():
    collector = PostCallTelemetryCollector()
    async with AgentSession(llm=FakeLLM()) as sess1, AgentSession(llm=FakeLLM()) as sess2:
        collector.attach(sess1)
        with pytest.raises(RuntimeError):
            collector.attach(sess2)


@pytest.mark.asyncio
async def test_attach_after_detach_allows_new_session():
    collector = PostCallTelemetryCollector()
    async with AgentSession(llm=FakeLLM()) as sess1, AgentSession(llm=FakeLLM()) as sess2:
        collector.attach(sess1)
        collector.detach()
        collector.attach(sess2)
        assert collector.attached


@pytest.mark.asyncio
async def test_detach_unregisters_close_listener():
    collector = PostCallTelemetryCollector()
    async with AgentSession(llm=FakeLLM()) as sess:
        collector.attach(sess)
        collector.detach()
    # the session closed after detach; the auto-finalize-on-close listener must not
    # have fired, so nothing should be cached
    assert collector._cached_report is None


@pytest.mark.asyncio
async def test_detach_actually_removes_the_registered_close_handler():
    # Direct regression test for the EventEmitter.once()-vs-.on() bug: .once() stores an
    # internal wrapper closure, not the passed callback, so `session.off(event,
    # self._on_close)` would silently remove nothing and _on_close would still fire
    # after detach. The spy must be installed *before* attach() — patching afterwards
    # wouldn't touch whatever object was actually captured by the event registration.
    collector = PostCallTelemetryCollector()
    sess = AgentSession(llm=FakeLLM())
    with patch.object(collector, "_on_close") as spy:
        collector.attach(sess)
        collector.detach()
        await sess.start(_EchoAgent())
        await sess.aclose()
        spy.assert_not_called()


@pytest.mark.asyncio
async def test_attach_to_new_session_clears_stale_cached_report():
    collector = PostCallTelemetryCollector(metadata={"call": "first"})
    first_session = AgentSession(llm=FakeLLM())
    collector.attach(first_session)
    await first_session.start(_EchoAgent())
    await first_session.aclose()

    first_report = collector.finalize()
    assert first_report.metadata == {"call": "first"}

    collector.detach()
    collector._metadata = {"call": "second"}
    second_session = AgentSession(llm=FakeLLM())
    collector.attach(second_session)
    await second_session.start(_EchoAgent())
    await second_session.aclose()

    second_report = collector.finalize()
    assert second_report.metadata == {"call": "second"}
    assert second_report.report_id != first_report.report_id


def test_detach_before_attach_is_safe_noop():
    collector = PostCallTelemetryCollector()
    collector.detach()
    assert not collector.attached


def test_finalize_without_attach_raises():
    collector = PostCallTelemetryCollector()
    with pytest.raises(RuntimeError):
        collector.finalize()


@pytest.mark.asyncio
async def test_finalize_after_session_garbage_collected_raises_clear_error():
    collector = PostCallTelemetryCollector()
    sess = AgentSession(llm=FakeLLM())
    collector.attach(sess)
    del sess
    gc.collect()
    with pytest.raises(RuntimeError):
        collector.finalize()


# --- finalize semantics: preliminary vs authoritative -------------------------------


@pytest.mark.asyncio
async def test_finalize_before_session_started_is_empty_and_not_ended():
    collector = PostCallTelemetryCollector()
    sess = AgentSession(llm=FakeLLM())
    collector.attach(sess)
    report = collector.finalize()
    assert report.ended is False
    assert report.transcript == []
    assert report.tool_executions == []


@pytest.mark.asyncio
async def test_finalize_before_close_returns_snapshot_with_ended_false():
    collector = PostCallTelemetryCollector()
    async with AgentSession(llm=FakeLLM()) as sess:
        collector.attach(sess)
        await sess.start(_EchoAgent())
        report = collector.finalize()
        assert report.ended is False


@pytest.mark.asyncio
async def test_finalize_repeated_calls_before_close_are_not_cached():
    collector = PostCallTelemetryCollector()
    async with AgentSession(llm=FakeLLM()) as sess:
        collector.attach(sess)
        await sess.start(_EchoAgent())
        report1 = collector.finalize()
        report2 = collector.finalize()
        assert report1 is not report2
        assert report1.report_id != report2.report_id


@pytest.mark.asyncio
async def test_finalize_after_close_returns_cached_authoritative_report():
    collector = PostCallTelemetryCollector()
    sess = AgentSession(llm=FakeLLM())
    collector.attach(sess)
    await sess.start(_EchoAgent())
    await sess.aclose()

    report1 = collector.finalize()
    report2 = collector.finalize()
    assert report1.ended is True
    assert report1 is report2


@pytest.mark.asyncio
async def test_attach_registers_close_listener_that_auto_finalizes():
    collector = PostCallTelemetryCollector()
    sess = AgentSession(llm=FakeLLM())
    collector.attach(sess)
    await sess.start(_EchoAgent())
    await sess.aclose()

    assert collector._cached_report is not None
    assert collector._cached_report.ended is True


@pytest.mark.asyncio
async def test_finalize_does_not_perform_network_io():
    collector = PostCallTelemetryCollector()
    async with AgentSession(llm=FakeLLM()) as sess:
        collector.attach(sess)
        await sess.start(_EchoAgent())
        with patch("aiohttp.ClientSession") as mock_session_cls:
            collector.finalize()
            mock_session_cls.assert_not_called()


# --- job context resolution ---------------------------------------------------------


@pytest.mark.asyncio
async def test_finalize_with_job_context_populates_identity_fields():
    collector = PostCallTelemetryCollector()
    async with AgentSession(llm=FakeLLM()) as sess:
        collector.attach(sess, job_ctx=_mock_job_ctx())
        await sess.start(_EchoAgent())
        report = collector.finalize()
        assert report.job_id == "job-1"
        assert report.room_id == "room-sid-1"
        assert report.room_name == "room-1"
        assert report.participant_identity == "agent-identity"


@pytest.mark.asyncio
async def test_finalize_without_job_context_leaves_identity_fields_none():
    collector = PostCallTelemetryCollector()
    async with AgentSession(llm=FakeLLM()) as sess:
        collector.attach(sess, job_ctx=None)
        await sess.start(_EchoAgent())
        report = collector.finalize()
        assert report.job_id is None
        assert report.room_id is None
        assert report.room_name is None


@pytest.mark.asyncio
async def test_finalize_auto_detects_job_context_when_not_given():
    collector = PostCallTelemetryCollector()
    async with AgentSession(llm=FakeLLM()) as sess:
        collector.attach(sess)  # job_ctx left as NOT_GIVEN
        await sess.start(_EchoAgent())
        with patch(f"{_COLLECTOR_MOD}.get_job_context", return_value=_mock_job_ctx()) as mock_get:
            report = collector.finalize()
        mock_get.assert_called_once_with(required=False)
        assert report.job_id == "job-1"


@pytest.mark.asyncio
async def test_finalize_participant_identity_lookup_failure_is_swallowed():
    ctx = _mock_job_ctx()
    type(ctx).local_participant_identity = property(
        lambda self: (_ for _ in ()).throw(RuntimeError)
    )
    collector = PostCallTelemetryCollector()
    async with AgentSession(llm=FakeLLM()) as sess:
        collector.attach(sess, job_ctx=ctx)
        await sess.start(_EchoAgent())
        report = collector.finalize()
        assert report.participant_identity is None
        assert report.job_id == "job-1"  # other fields still populate


# --- redact hook ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_redact_hook_applied_before_caching():
    def _redact(report):
        return report.model_copy(update={"metadata": {"redacted": True}})

    collector = PostCallTelemetryCollector(metadata={"secret": "value"}, redact=_redact)
    sess = AgentSession(llm=FakeLLM())
    collector.attach(sess)
    await sess.start(_EchoAgent())
    await sess.aclose()

    report = collector.finalize()
    assert report.metadata == {"redacted": True}


# --- integration: transcript + successful/failed tool calls -------------------------


@pytest.mark.asyncio
async def test_integration_full_session_produces_transcript_and_tool_executions():
    llm = FakeLLM(
        fake_responses=[
            FakeLLMResponse(
                input="please look something up",
                content="",
                ttft=0.05,
                duration=0.05,
                tool_calls=[FunctionToolCall(name="lookup", arguments="{}", call_id="call_1")],
            ),
            # after the tool runs, FakeLLM looks up the next response by the tool's
            # own output string ("42", see fake_llm.py's _get_index_text)
            FakeLLMResponse(input="42", content="the answer is 42", ttft=0.05, duration=0.05),
        ]
    )

    collector = PostCallTelemetryCollector(metadata={"lead_id": "lead_123"})
    sess = AgentSession(llm=llm)
    collector.attach(sess, job_ctx=_mock_job_ctx())
    await sess.start(_EchoAgent())

    await sess.run(user_input="please look something up")
    await sess.aclose()

    report = collector.finalize()

    assert report.ended is True
    assert report.job_id == "job-1"
    assert any(t.role == "user" and t.text == "please look something up" for t in report.transcript)
    assert len(report.tool_executions) == 1
    tool_record = report.tool_executions[0]
    assert tool_record.name == "lookup"
    assert tool_record.status == "done"
    assert tool_record.is_error is False
    assert report.metadata == {"lead_id": "lead_123"}
