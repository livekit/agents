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


def speech_data_post_init(data: Any) -> None:
    from .language import LanguageCode

    data.language = LanguageCode(data.language)
    for name in ("source_languages", "target_languages"):
        if (langs := getattr(data, name)) is not None:
            setattr(data, name, [LanguageCode(lang) for lang in langs])


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
