from typing import Literal, Union

# https://docs.anthropic.com/en/docs/about-claude/model-deprecations#model-status

ChatModels = Literal[
    "claude-3-5-sonnet-20240620",  # deprecated
    "claude-3-opus-20240229",  # deprecated
    "claude-3-5-sonnet-20241022",  # deprecated
    "claude-3-haiku-20240307",
    "claude-3-5-haiku-20241022",
    "claude-3-7-sonnet-20250219",
    "claude-sonnet-4-20250514",
    "claude-opus-4-20250514",
    "claude-opus-4-1-20250805",
    "claude-sonnet-4-5-20250929",
    "claude-haiku-4-5-20251001",
    "claude-opus-4-5-20251101",
]


# Models that support extended thinking
# https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking#supported-models
THINKING_MODELS: set[str] = {
    "claude-3-7-sonnet-20250219",
    "claude-sonnet-4-20250514",
    "claude-opus-4-20250514",
    "claude-opus-4-1-20250805",
    "claude-sonnet-4-5-20250929",
    "claude-haiku-4-5-20251001",
    "claude-opus-4-5-20251101",
}


def _supports_thinking(model: Union[ChatModels, str]) -> bool:
    """Check if the model supports extended thinking."""
    return model in THINKING_MODELS
