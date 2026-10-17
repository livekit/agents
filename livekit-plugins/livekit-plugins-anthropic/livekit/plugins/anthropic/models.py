from typing import Literal

from anthropic.types import ThinkingConfigParam

# https://docs.anthropic.com/en/docs/about-claude/model-deprecations#model-status

ChatModels = Literal[
    "claude-3-5-sonnet-20240620",  # deprecated
    "claude-3-opus-20240229",  # deprecated
    "claude-3-5-sonnet-20241022",  # deprecated
    "claude-3-haiku-20240307",
    "claude-3-5-haiku-20241022",
    "claude-3-7-sonnet-20250219",
    "claude-sonnet-4-20250514",
    "claude-sonnet-4-6",
    "claude-sonnet-5",
    "claude-sonnet-5-5",
    "claude-haiku-5-5",
    "claude-opus-4-20250514",
    "claude-opus-4-1-20250805",
    "claude-opus-4-6",
    "claude-opus-4-7",
    "claude-opus-4-8",
    "claude-opus-5",
    "claude-opus-5-5",
]

# Default ``thinking`` setting per model when the caller does not choose one.
#
# Claude 5 generation models think by default when a request carries no ``thinking``
# field. The plugin caps each response at 1024 tokens by default, and ``max_tokens``
# covers thinking and the reply together, so on default settings Sonnet 5 and Haiku 5.5
# routinely spend the whole budget thinking and return no text at all, or take long
# enough that the per-request timeout fires first. Turning thinking off restores an
# immediate reply on both.
#
# The table is keyed by exact model id on purpose. Models that accept ``thinking`` but
# behave acceptably on API defaults (Opus 5, Sonnet 5.5, Opus 5.5) and models that
# reject ``{"type": "disabled"}`` outright (Sonnet 5.5, Opus 5.5) are deliberately not
# listed, and any id the plugin has not been measured against gets the API default
# instead of a value the API may refuse.
_DEFAULT_THINKING: dict[str, ThinkingConfigParam] = {
    "claude-sonnet-5": {"type": "disabled"},
    "claude-haiku-5-5": {"type": "disabled"},
}


def default_thinking_for_model(model: str) -> ThinkingConfigParam | None:
    """Return the plugin's default ``thinking`` setting for ``model``, or ``None`` to send
    no ``thinking`` field and let the API apply its own default."""
    return _DEFAULT_THINKING.get(model)
