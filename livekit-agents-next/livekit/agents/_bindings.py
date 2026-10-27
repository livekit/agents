"""The hand-written method bodies of the generated models (``_generated/*.py`` call them)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

if TYPE_CHECKING:
    from ._generated.events import FunctionToolsExecutedEvent
    from .llm import FunctionCall, FunctionCallOutput


def function_tools_executed_zipped(
    ev: FunctionToolsExecutedEvent,
) -> list[tuple[FunctionCall, FunctionCallOutput]]:
    return list(zip(ev.function_calls, ev.function_call_outputs, strict=False))


def function_tools_executed_cancel_tool_reply(ev: FunctionToolsExecutedEvent) -> None:
    for out in ev.function_call_outputs:
        out.reply_required = False


def function_tools_executed_cancel_agent_handoff(ev: FunctionToolsExecutedEvent) -> None:
    ev._handoff_required = False


def function_tools_executed_has_tool_reply(ev: FunctionToolsExecutedEvent) -> bool:
    return any(out.reply_required for out in ev.function_call_outputs)


def function_tools_executed_has_agent_handoff(ev: FunctionToolsExecutedEvent) -> bool:
    return ev._handoff_required


def verify_lists_length(ev: FunctionToolsExecutedEvent) -> FunctionToolsExecutedEvent:
    if len(ev.function_calls) != len(ev.function_call_outputs):
        raise ValueError("The number of function_calls and function_call_outputs must match.")
    return ev


def chat_chunk_has_response(chunk: Any) -> bool:
    return bool(chunk.delta and (chunk.delta.content or chunk.delta.tool_calls))


def metrics_repr(metrics: BaseModel) -> str:
    fields = metrics.model_dump(exclude_defaults=True)
    return f"{type(metrics).__name__}({', '.join(f'{k}={v!r}' for k, v in fields.items())})"


def model_usage_repr(usage: BaseModel) -> str:
    # skip zeros for concise display
    fields = {k: v for k, v in usage.model_dump().items() if v != 0 and v != 0.0}
    return f"{type(usage).__name__}({', '.join(f'{k}={v!r}' for k, v in fields.items())})"


def _default_total_tokens(metrics: Any) -> Any:
    if "total_tokens" not in metrics.model_fields_set:
        metrics.total_tokens = metrics.input_tokens + metrics.output_tokens
    return metrics


def speech_data_post_init(data: Any) -> None:
    from .language import LanguageCode

    def code(lang: Any) -> Any:
        return (
            LanguageCode(lang)
            if isinstance(lang, str) and not isinstance(lang, LanguageCode)
            else lang
        )

    data.language = code(data.language)
    for name in ("source_languages", "target_languages"):
        if (langs := getattr(data, name)) is not None:
            setattr(data, name, [code(lang) for lang in langs])


def serialize_source(source: Any) -> Any:
    from .llm import LLM, RealtimeModel
    from .stt import STT
    from .tts import TTS

    if isinstance(source, LLM | STT | TTS | RealtimeModel):
        return {"model": source.model, "provider": source.provider}
    if isinstance(source, BaseModel):
        return source.model_dump()
    return repr(source)


def serialize_error(error: Any) -> Any:
    if isinstance(error, BaseModel):
        return error.model_dump()
    return repr(error)
