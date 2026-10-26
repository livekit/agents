"""Keyterm biasing: the core runs the detector; this resolves its LLM and hands over its tool."""

from __future__ import annotations

from typing import Any

from .. import _core
from .._generated.keyterms import KeytermDetectionOptions, KeytermsOptions, STTContextOptions
from ..llm import LLM, function_tool
from ..log import logger

__all__ = ["KeytermDetectionOptions", "KeytermsOptions", "STTContextOptions"]


@function_tool(name="record_keyterms")
async def _record_keyterms(pending: list[str], confirm: list[str], remove: list[str]) -> None:
    """Update the STT keyterms based on the latest transcript.

    Args:
        pending: Distinctive terms seen but not yet trusted — tracked, not applied.
        confirm: Pending terms the transcript has now corroborated — applied.
        remove: Only a spelling the user corrected away; applied terms are otherwise sticky.
    """
    # only used to elicit a structured tool call; never executed


def _detection_llm(stt_context: dict[str, Any] | None) -> LLM | None:
    """The detection LLM ``stt_context_options`` asks for: the ``LLM`` given, else the model the
    core names (``_core.resolve_keyterm_detection``) through the inference gateway. ``None`` with
    detection off, or when the model cannot be created (detection is then skipped)."""
    config = (stt_context or {}).get("keyterm_detection")
    resolved = _core.resolve_keyterm_detection(config)
    if not resolved["enabled"]:
        return None
    if (model := resolved["model"]) is None:
        llm = config["llm"]  # type: ignore[index]
        return llm if isinstance(llm, LLM) else None
    try:
        from ..inference import LLM as InferenceLLM

        return InferenceLLM.from_model_string(model)
    except Exception:  # noqa: BLE001 — never let detection setup break the session
        logger.warning("keyterm detection: could not create detection LLM %r; skipping", model)
        return None
