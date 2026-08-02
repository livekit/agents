from __future__ import annotations

import dataclasses
from enum import Enum

import pytest

from livekit.agents.beta.gtm_telemetry.models import (
    CollectorConfig,
    PostCallReport,
    to_json_safe,
)
from livekit.agents.llm import ChatContext, FunctionCall, FunctionCallOutput
from livekit.agents.metrics import LLMModelUsage
from livekit.agents.voice.events import (
    AgentEvent,
    CloseEvent,
    CloseReason,
    FunctionToolsExecutedEvent,
    ToolCallEnded,
    ToolCallStarted,
    ToolCallUpdated,
    ToolExecutionUpdatedEvent,
)

pytestmark = pytest.mark.unit


def _report(
    *,
    events: list[AgentEvent] | None = None,
    chat_history: ChatContext | None = None,
    model_usage: list | None = None,
    config: CollectorConfig | None = None,
    metadata: dict | None = None,
    started_at: float | None = 1_700_000_000.0,
) -> PostCallReport:
    return PostCallReport.from_session(
        job_id="job_1",
        room_id="room_sid_1",
        room_name="my-room",
        participant_identity="user_1",
        started_at=started_at,
        events=events or [],
        chat_history=chat_history if chat_history is not None else ChatContext(),
        model_usage=model_usage or [],
        config=config or CollectorConfig(),
        metadata=metadata or {},
    )


def _chat_ctx(*, include_system: bool = False) -> ChatContext:
    ctx = ChatContext()
    if include_system:
        ctx.add_message(role="system", content="be nice", created_at=1.0)
    ctx.add_message(role="user", content="hello", created_at=2.0)
    ctx.add_message(role="assistant", content="hi there", interrupted=True, created_at=3.0)
    return ctx


# --- transcript -------------------------------------------------------------------


def test_transcript_filters_by_role_default_user_assistant():
    report = _report(chat_history=_chat_ctx(include_system=True))
    assert [t.role for t in report.transcript] == ["user", "assistant"]


def test_transcript_includes_system_when_opted_in():
    report = _report(
        chat_history=_chat_ctx(include_system=True),
        config=CollectorConfig(include_system_messages=True),
    )
    assert "system" in [t.role for t in report.transcript]


def test_transcript_preserves_interrupted_flag():
    report = _report(chat_history=_chat_ctx())
    assistant_turn = next(t for t in report.transcript if t.role == "assistant")
    assert assistant_turn.interrupted is True


def test_transcript_uses_text_content_strips_expr_markup():
    ctx = ChatContext()
    ctx.add_message(role="assistant", content="<expr>happy</expr>hello", created_at=1.0)
    report = _report(chat_history=ctx)
    assert "<expr>" not in report.transcript[0].text


def test_transcript_empty_message_becomes_empty_string():
    ctx = ChatContext()
    ctx.add_message(role="user", content=[], created_at=1.0)
    report = _report(chat_history=ctx)
    assert report.transcript[0].text == ""


def test_transcript_order_matches_chat_history_order():
    ctx = _chat_ctx()
    report = _report(chat_history=ctx)
    assert [t.text for t in report.transcript] == ["hello", "hi there"]


def test_transcript_ignores_function_call_items():
    ctx = ChatContext()
    ctx.add_message(role="user", content="hi", created_at=1.0)
    ctx.insert(FunctionCall(call_id="c1", name="foo", arguments="{}", created_at=2.0))
    report = _report(chat_history=ctx)
    assert len(report.transcript) == 1


# --- tool executions ----------------------------------------------------------------


def _started(call_id: str, name: str = "lookup", arguments: str = "{}", at: float = 1.0):
    return ToolExecutionUpdatedEvent(
        update=ToolCallStarted(
            function_call=FunctionCall(
                call_id=call_id, name=name, arguments=arguments, created_at=at
            )
        ),
        created_at=at,
    )


def _ended(call_id: str, status: str, message: str | None = None, at: float = 2.0):
    return ToolExecutionUpdatedEvent(
        update=ToolCallEnded(id=call_id, call_id=call_id, message=message, status=status),
        created_at=at,
    )


def _progress(call_id: str, message: str, at: float = 1.5):
    return ToolExecutionUpdatedEvent(
        update=ToolCallUpdated(id=call_id, call_id=call_id, message=message),
        created_at=at,
    )


def test_tool_execution_status_done():
    report = _report(events=[_started("c1"), _ended("c1", "done")])
    (record,) = report.tool_executions
    assert record.status == "done"
    assert record.is_error is False


def test_tool_execution_status_error():
    report = _report(events=[_started("c1"), _ended("c1", "error", message="boom")])
    (record,) = report.tool_executions
    assert record.status == "error"
    assert record.is_error is True
    assert record.error == "boom"


def test_tool_execution_status_cancelled():
    report = _report(events=[_started("c1"), _ended("c1", "cancelled")])
    (record,) = report.tool_executions
    assert record.status == "cancelled"
    assert record.is_error is False


def test_tool_execution_missing_terminal_event_becomes_interrupted():
    report = _report(events=[_started("c1")])
    (record,) = report.tool_executions
    assert record.status == "interrupted"


def test_tool_execution_missing_start_event_still_recorded():
    report = _report(events=[_ended("c1", "done")])
    (record,) = report.tool_executions
    assert record.call_id == "c1"
    assert record.status == "done"
    assert record.started_at is None


def test_tool_execution_out_of_order_events_still_reduce_correctly():
    # ended arrives before started in the snapshot list
    report = _report(events=[_ended("c1", "done"), _started("c1")])
    (record,) = report.tool_executions
    assert record.name == "lookup"
    assert record.status == "done"


def test_tool_execution_duplicate_started_event_does_not_duplicate_record():
    report = _report(events=[_started("c1"), _started("c1"), _ended("c1", "done")])
    assert len(report.tool_executions) == 1


def test_tool_execution_progress_updates_ordered():
    report = _report(
        events=[
            _started("c1"),
            _progress("c1", "step 1", at=1.1),
            _progress("c1", "step 2", at=1.2),
            _ended("c1", "done"),
        ]
    )
    (record,) = report.tool_executions
    assert [u.message for u in record.progress_updates] == ["step 1", "step 2"]


def test_tool_execution_same_name_different_call_ids_are_separate_records():
    report = _report(
        events=[_started("c1"), _ended("c1", "done"), _started("c2"), _ended("c2", "error")]
    )
    assert {r.call_id for r in report.tool_executions} == {"c1", "c2"}
    assert len(report.tool_executions) == 2


def test_tool_execution_arguments_parsed_as_json():
    report = _report(events=[_started("c1", arguments='{"x": 1}'), _ended("c1", "done")])
    (record,) = report.tool_executions
    assert record.arguments == {"x": 1}


def test_tool_execution_non_json_arguments_kept_as_raw_string():
    report = _report(events=[_started("c1", arguments="not-json"), _ended("c1", "done")])
    (record,) = report.tool_executions
    assert record.arguments == "not-json"


def test_tool_execution_arguments_omitted_when_disabled():
    report = _report(
        events=[_started("c1", arguments='{"x": 1}'), _ended("c1", "done")],
        config=CollectorConfig(include_tool_arguments=False),
    )
    (record,) = report.tool_executions
    assert record.arguments is None


def test_tool_execution_enriched_with_result_from_function_tools_executed():
    events = [
        _started("c1"),
        _ended("c1", "done"),
        FunctionToolsExecutedEvent(
            function_calls=[
                FunctionCall(call_id="c1", name="lookup", arguments="{}", created_at=1.0)
            ],
            function_call_outputs=[
                FunctionCallOutput(
                    call_id="c1", output='{"weather": "sunny"}', is_error=False, created_at=2.0
                )
            ],
        ),
    ]
    report = _report(events=events)
    (record,) = report.tool_executions
    assert record.result == {"weather": "sunny"}


def test_tool_execution_result_omitted_when_disabled():
    events = [
        _started("c1"),
        _ended("c1", "done"),
        FunctionToolsExecutedEvent(
            function_calls=[
                FunctionCall(call_id="c1", name="lookup", arguments="{}", created_at=1.0)
            ],
            function_call_outputs=[
                FunctionCallOutput(call_id="c1", output="ok", is_error=False, created_at=2.0)
            ],
        ),
    ]
    report = _report(events=events, config=CollectorConfig(include_tool_results=False))
    (record,) = report.tool_executions
    assert record.result is None


def test_tool_execution_none_output_in_batch_event_is_tolerated():
    events = [
        _started("c1"),
        _ended("c1", "done"),
        FunctionToolsExecutedEvent(
            function_calls=[
                FunctionCall(call_id="c1", name="lookup", arguments="{}", created_at=1.0)
            ],
            function_call_outputs=[None],
        ),
    ]
    report = _report(events=events)
    (record,) = report.tool_executions
    assert record.result is None
    assert record.status == "done"


def test_tool_execution_function_tools_executed_alone_infers_error_status():
    # simulates the collector attaching after the start/end events were already missed
    events = [
        FunctionToolsExecutedEvent(
            function_calls=[
                FunctionCall(call_id="c1", name="lookup", arguments="{}", created_at=1.0)
            ],
            function_call_outputs=[
                FunctionCallOutput(call_id="c1", output="failed", is_error=True, created_at=2.0)
            ],
        ),
    ]
    report = _report(events=events)
    (record,) = report.tool_executions
    assert record.is_error is True
    assert record.status == "error"


# --- metrics --------------------------------------------------------------------------


def test_metrics_model_usage_passthrough():
    usage = [LLMModelUsage(provider="openai", model="gpt-4o", input_tokens=10, output_tokens=5)]
    report = _report(model_usage=usage)
    assert report.metrics.model_usage == usage


def test_metrics_no_latency_samples_means_empty_latency_dict():
    report = _report(chat_history=ChatContext())
    assert report.metrics.latency == {}


def test_metrics_latency_aggregate_math():
    ctx = ChatContext()
    ctx.add_message(
        role="assistant",
        content="a",
        created_at=1.0,
        metrics={"llm_node_ttft": 0.1},
    )
    ctx.add_message(
        role="assistant",
        content="b",
        created_at=2.0,
        metrics={"llm_node_ttft": 0.3},
    )
    report = _report(chat_history=ctx)
    agg = report.metrics.latency["llm_node_ttft"]
    assert agg.count == 2
    assert agg.min == pytest.approx(0.1)
    assert agg.max == pytest.approx(0.3)
    assert agg.sum == pytest.approx(0.4)
    assert agg.mean == pytest.approx(0.2)


def test_metrics_tool_call_and_error_counts():
    report = _report(
        events=[_started("c1"), _ended("c1", "done"), _started("c2"), _ended("c2", "error")]
    )
    assert report.metrics.tool_call_count == 2
    assert report.metrics.tool_error_count == 1


# --- end_reason / ended / duration -----------------------------------------------------


def test_ended_and_end_reason_from_close_event():
    close = CloseEvent(reason=CloseReason.USER_INITIATED, created_at=1_700_000_010.0)
    report = _report(events=[close])
    assert report.ended is True
    assert report.end_reason == "user_initiated"


def test_not_ended_without_close_event():
    report = _report(events=[])
    assert report.ended is False
    assert report.end_reason is None


def test_duration_from_close_event_created_at_when_ended():
    close = CloseEvent(reason=CloseReason.TASK_COMPLETED, created_at=1_700_000_010.0)
    report = _report(events=[close], started_at=1_700_000_000.0)
    assert report.duration == pytest.approx(10.0)


def test_duration_is_none_when_started_at_is_none():
    report = _report(started_at=None)
    assert report.duration is None


def test_duration_clamped_nonnegative():
    close = CloseEvent(reason=CloseReason.ERROR, created_at=100.0)
    report = _report(events=[close], started_at=200.0)
    assert report.duration == 0.0


# --- to_json_safe -----------------------------------------------------------------------


class _Color(Enum):
    RED = "red"


@dataclasses.dataclass
class _Point:
    x: int
    y: int


class _Unserializable:
    def __repr__(self) -> str:
        return f"<_Unserializable object at 0x{id(self):x}>"


def test_to_json_safe_primitives_passthrough():
    assert to_json_safe(None) is None
    assert to_json_safe(True) is True
    assert to_json_safe(1) == 1
    assert to_json_safe("s") == "s"


def test_to_json_safe_nan_and_infinity_become_sentinel_strings():
    assert to_json_safe(float("nan")) == "<nan>"
    assert to_json_safe(float("inf")) == "<inf>"
    assert to_json_safe(float("-inf")) == "<-inf>"


def test_to_json_safe_enum_uses_value():
    assert to_json_safe(_Color.RED) == "red"


def test_to_json_safe_dataclass_recursed():
    assert to_json_safe(_Point(1, 2)) == {"x": 1, "y": 2}


def test_to_json_safe_pydantic_model_recursed():
    usage = LLMModelUsage(provider="openai", model="gpt-4o")
    safe = to_json_safe(usage)
    assert safe["provider"] == "openai"


def test_to_json_safe_bytes_base64_encoded():
    safe = to_json_safe(b"hello")
    assert safe == {"__bytes_b64__": "aGVsbG8="}


def test_to_json_safe_tuple_and_set_become_list():
    assert to_json_safe((1, 2)) == [1, 2]
    assert isinstance(to_json_safe({1, 2}), list)


def test_to_json_safe_exception_becomes_type_and_message_never_traceback():
    safe = to_json_safe(ValueError("bad value"))
    assert safe == {"type": "ValueError", "message": "bad value"}


def test_to_json_safe_unrecognized_object_never_uses_repr_or_memory_address():
    safe = to_json_safe(_Unserializable())
    assert isinstance(safe, str)
    assert "0x" not in safe
    assert safe == "<unserializable:_Unserializable>"


def test_to_json_safe_circular_reference_detected():
    d: dict = {}
    d["self"] = d
    assert to_json_safe(d) == {"self": "<circular>"}


def test_to_json_safe_max_depth_guarded():
    nested = None
    for _ in range(50):
        nested = [nested]
    safe = to_json_safe(nested)
    # must terminate and produce the sentinel somewhere in the structure
    assert _contains(safe, "<max-depth-exceeded>")


def _contains(value, needle) -> bool:
    if value == needle:
        return True
    if isinstance(value, list):
        return any(_contains(v, needle) for v in value)
    if isinstance(value, dict):
        return any(_contains(v, needle) for v in value.values())
    return False


# --- report-level ------------------------------------------------------------------------


def test_report_metadata_is_normalized_copy_not_shared():
    meta = {"lead_id": "123"}
    report = _report(metadata=meta)
    report.metadata["lead_id"] = "mutated"
    assert meta["lead_id"] == "123"


def test_report_json_roundtrip_is_deterministic():
    report = _report(chat_history=_chat_ctx())
    first = report.model_dump_json()
    second = report.model_dump_json()
    assert first == second


def test_report_serializes_to_valid_json():
    import json

    report = _report(chat_history=_chat_ctx(), events=[_started("c1"), _ended("c1", "done")])
    parsed = json.loads(report.model_dump_json())
    assert parsed["schema_version"] == "1"
    assert isinstance(parsed["started_at"], float)


def test_report_id_is_stable_within_one_report():
    report = _report()
    assert report.report_id == report.model_copy().report_id
