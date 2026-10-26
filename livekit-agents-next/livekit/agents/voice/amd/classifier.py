"""The classifier's public types and the tools its LLM answers with.

The classifier itself is the core's (``_core.LiveSession.attach_amd``); the tools here are the
objects a host LLM is handed beside the core's specs, and are never executed.
"""

from __future__ import annotations

from ... import _core
from ..._generated.amd import AMDCategory, AMDPredictionEvent
from ...llm.tool_context import function_tool

__all__ = ["AMD_PROMPT", "AMDCategory", "AMDPredictionEvent"]

AMD_PROMPT: str = _core.amd_constants()["prompt"]


@function_tool
async def save_prediction(label: AMDCategory) -> None:
    """Save the prediction to the verdict."""


@function_tool
async def postpone_termination(seconds: float) -> str:
    """Postpone the termination of the classification task.
    Use when the transcript is ambiguous and more audio is expected.

    Args:
        seconds: Additional seconds to wait (max 10).
    """
    return ""


CLASSIFIER_TOOLS = [save_prediction, postpone_termination]
