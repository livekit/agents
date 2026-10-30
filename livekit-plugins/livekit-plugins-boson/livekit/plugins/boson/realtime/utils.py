"""Wire-format helpers for the Boson realtime model."""

from __future__ import annotations

from typing import Any, Literal
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from openai.types.realtime import ConversationItemCreateEvent
from pydantic import BaseModel, ConfigDict

from livekit.agents import llm, utils
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

# The same refusals as an `error` event, which the server sends just before it
# closes with 4429. Reconnecting cannot clear any of them, so they end the
# session rather than being reported as recoverable. Kept separate from the base
# class's own fatal set, which would catch only `insufficient_quota` here: it
# reads `code` in preference to `type`, and three of these four codes are Boson's
# own. `type` is `insufficient_quota` for all of them.
_BOSON_FATAL_ERROR_CODES = frozenset(
    {"insufficient_quota", "monthly_cap_reached", "contract_ended", "no_billing_account"}
)
_BOSON_FATAL_ERROR_TYPES = frozenset({"insufficient_quota"})

# Roles the server's conversation store cannot hold; see
# _livekit_item_to_boson_item and _create_update_chat_ctx_events.
_UNSUPPORTED_ITEM_ROLES = frozenset({"system", "developer"})


def _normalize_ws_url(url: str, query_params: dict[str, str]) -> str:
    parsed = urlparse(url)
    scheme = parsed.scheme
    if scheme == "http":
        scheme = "ws"
    elif scheme == "https":
        scheme = "wss"
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query.update(query_params)
    return urlunparse(
        (scheme, parsed.netloc, parsed.path, parsed.params, urlencode(query), parsed.fragment)
    )


def _set_message_text(message: llm.ChatMessage, text: str) -> None:
    """Replace the first text part of ``message`` (or append one) with ``text``."""
    text_index = next(
        (idx for idx, content in enumerate(message.content) if isinstance(content, str)),
        None,
    )
    if text_index is None:
        message.content.append(text)
    else:
        message.content[text_index] = text


def _is_boson_fatal_error(error: dict[str, Any]) -> bool:
    """Whether this error is an account-level refusal a reconnect cannot clear."""
    return error.get("code") in _BOSON_FATAL_ERROR_CODES or (
        error.get("type") in _BOSON_FATAL_ERROR_TYPES
    )


def _session_update_event(event_prefix: str, session: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "session.update",
        "event_id": utils.shortuuid(event_prefix),
        "session": {"type": "realtime", **session},
    }


def _merge_session_value(base: Any, update: Any) -> Any:
    """Apply ``update`` to ``base`` the way the server merges ``session.update``.

    Dicts merge key by key, recursively; anything else, ``None`` included,
    replaces the old value outright.
    """
    if not isinstance(base, dict) or not isinstance(update, dict):
        return update
    merged = dict(base)
    for key, value in update.items():
        merged[key] = _merge_session_value(merged.get(key), value)
    return merged


def _copy_dict_or_none(value: Any | None) -> dict[str, Any] | None:
    # `False` disables turn detection the same way `None` does (matches the
    # README and Boson's Pipecat client); `dict(False)` would otherwise raise
    # TypeError since bool isn't iterable.
    if value is None or value is False:
        return None
    return dict(value)


def _normalize_input_audio_transcription(
    value: Any | None,
) -> NotGivenOr[dict[str, Any]]:
    if value is None:
        return NOT_GIVEN
    transcription = dict(value)
    transcription.pop("prompt", None)
    return transcription


def _build_input_audio_transcription(
    *,
    input_audio_transcription: NotGivenOr[dict[str, Any] | None],
    model: str,
    language: str | None,
) -> NotGivenOr[dict[str, Any]]:
    has_convenience_options = bool(model) or language is not None
    if not has_convenience_options and (
        not is_given(input_audio_transcription) or input_audio_transcription is None
    ):
        return NOT_GIVEN

    transcription = (
        dict(input_audio_transcription)
        if is_given(input_audio_transcription) and input_audio_transcription is not None
        else {}
    )
    # A transcription `prompt` is not part of the supported wire config; drop
    # it even if the caller passed it through a raw transcription dict.
    transcription.pop("prompt", None)
    if model:
        transcription["model"] = model
    if language is not None:
        transcription["language"] = language
    return transcription


def _input_audio_transcription_enabled(transcription: NotGivenOr[dict[str, Any]]) -> bool:
    """Whether the server will emit user-transcription events for this config.

    The server returns transcript events only when the client sets a non-empty
    ``model``. Omitting the transcription block, sending ``null``, or sending a
    block without a model all run ASR server-side (for the LLM) but emit no
    client-facing ``conversation.item.input_audio_transcription.completed``
    events.
    """
    return is_given(transcription) and bool(transcription.get("model"))


def _resolve_output_modalities(
    output_modalities: list[Literal["text", "audio"]] | None,
) -> list[Literal["text", "audio"]]:
    """Validate ``output_modalities`` to exactly ``["text"]`` or ``["audio"]``.

    The server rejects mixed (``["text", "audio"]``) and empty lists — output is
    single-modality. ``None`` defaults to ``["audio"]``.
    """
    if output_modalities is None:
        return ["audio"]
    if len(output_modalities) != 1 or output_modalities[0] not in ("text", "audio"):
        raise ValueError(
            "output_modalities must be exactly one of ['text'] or ['audio'] "
            f"(got {output_modalities!r}); mixed and empty lists are not supported."
        )
    return list(output_modalities)


def _build_noise_reduction(value: NotGivenOr[Any | None]) -> dict[str, Any] | None:
    """Normalize ``input_audio_noise_reduction`` to the OpenAI object form.

    Accepts a bare type string (``"near_field"`` / ``"far_field"``) or a dict
    (``{"type": ...}``). ``NOT_GIVEN`` and ``None`` both disable it (nothing is
    sent, which the server treats as disabled).
    """
    if not is_given(value) or value is None:
        return None
    if isinstance(value, str):
        return {"type": value}
    return dict(value)


def _server_vad_auto_interrupts_response(turn_detection: dict[str, Any] | None) -> bool:
    """Whether server VAD will auto-cancel the active response on speech start.

    The wire ``interrupt_response`` field does not currently change Higgs
    Realtime's actual behavior: with server VAD enabled the server always
    cancels the active response when it detects speech, regardless of what the
    client sends for this field. So the client's own duplicate-cancel
    suppression must key off whether server VAD is enabled at all, not off the
    (currently inert) ``interrupt_response`` value.
    """
    return turn_detection is not None


def _tools_to_boson(tools: list[llm.Tool]) -> list[dict[str, Any]]:
    boson_tools: list[dict[str, Any]] = []
    for tool in tools:
        if isinstance(tool, llm.FunctionTool):
            boson_tools.append(llm.utils.build_legacy_openai_schema(tool, internally_tagged=True))
        elif isinstance(tool, llm.RawFunctionTool):
            raw_schema = dict(tool.info.raw_schema)
            raw_schema.pop("meta", None)
            raw_schema["type"] = "function"
            boson_tools.append(raw_schema)
    return boson_tools


def _tool_choice_to_boson(tool_choice: llm.ToolChoice | None) -> Any:
    if tool_choice is None:
        return "auto"
    if isinstance(tool_choice, str):
        return tool_choice
    function = tool_choice.get("function", {})
    name = function.get("name")
    if name:
        return {"type": "function", "name": name}
    return "auto"


class _BosonConversationItem(BaseModel):
    """A conversation item in the Boson wire shape.

    Typed only as far as the base ``update_chat_ctx`` machinery needs
    (``item.id`` for echo correlation); the payload rides in extra fields
    because it deviates from the GA models (assistant content uses type
    ``"text"``, which their literals reject).
    """

    model_config = ConfigDict(extra="allow")
    id: str


class _BosonConversationItemCreateEvent(ConversationItemCreateEvent):
    item: _BosonConversationItem  # type: ignore[assignment]


def _livekit_item_to_boson_item(item: llm.ChatItem) -> dict[str, Any] | None:
    if item.type == "message":
        if item.role in _UNSUPPORTED_ITEM_ROLES:
            # Conversation items only accept role "assistant" or "user", so a
            # system/developer item can never be synced as one. Drop it
            # instead of sending a create that would be rejected; use
            # session-level instructions for persistent directives.
            return None
        content_text = _text_from_content(item.content)
        if not content_text:
            return None
        content_type = "text" if item.role == "assistant" else "input_text"
        return {
            "id": item.id,
            "object": "realtime.item",
            "type": "message",
            "role": item.role,
            "content": [{"type": content_type, "text": content_text}],
        }
    if item.type == "function_call":
        return {
            "id": item.id,
            "object": "realtime.item",
            "type": "function_call",
            "call_id": item.call_id,
            "name": item.name,
            "arguments": item.arguments,
            "status": "completed",
        }
    if item.type == "function_call_output":
        return {
            "id": item.id,
            "object": "realtime.item",
            "type": "function_call_output",
            "call_id": item.call_id,
            "output": item.output,
            "status": "completed",
        }
    return None


def _text_from_content(content: list[llm.ChatContent]) -> str:
    parts: list[str] = []
    for part in content:
        if isinstance(part, str):
            parts.append(part)
        elif isinstance(part, llm.AudioContent) and part.transcript:
            parts.append(part.transcript)
    return "\n".join(parts)


def _format_error_message(error: dict[str, Any], event_id: str | None) -> str:
    message = error.get("message") or "Boson realtime API error"
    details = {
        "type": error.get("type"),
        "code": error.get("code"),
        "event_id": event_id,
    }
    details = {key: value for key, value in details.items() if value is not None}
    if details:
        return f"{message} ({', '.join(f'{key}={value}' for key, value in details.items())})"
    return message
