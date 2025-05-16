from typing import Literal

from . import anthropic, aws, google, openai

LLMFormatName = Literal["openai", "google", "aws", "anthropic"]


__all__ = ["openai", "google", "aws", "anthropic", "LLMFormatName"]
