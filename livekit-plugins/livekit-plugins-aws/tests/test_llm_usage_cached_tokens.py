"""Bedrock usage must satisfy the contract CompletionUsage states for itself.

`CompletionUsage.prompt_tokens` is documented as "The number of input tokens used (includes
cached tokens)", and `total_tokens` as "completion + prompt tokens". Bedrock reports
`inputTokens` net of the cache and puts the cached tokens in `cacheReadInputTokens` and
`cacheWriteInputTokens`, so assigning `inputTokens` straight through leaves every cached
token out of `prompt_tokens` and makes `total_tokens` disagree with its own definition.

The anthropic plugin already adds them back for the same models served directly:

    prompt_token = self._input_tokens + self._cache_creation_tokens + self._cache_read_tokens

This pins the Bedrock path to the same behaviour.
"""

from __future__ import annotations

import pytest

from livekit.plugins.aws.llm import LLMStream

pytestmark = pytest.mark.unit


def _parse_metadata(usage: dict) -> object:
    """Drive _parse_chunk's metadata branch without constructing a live stream."""
    chunk = {"metadata": {"usage": usage}}
    return LLMStream._parse_chunk(object.__new__(LLMStream), "req-1", chunk)


class TestBedrockCachedTokens:
    def test_cache_reads_are_included_in_prompt_tokens(self):
        """2000 of the 2100 billed input tokens came from the cache."""
        result = _parse_metadata(
            {
                "inputTokens": 100,
                "outputTokens": 50,
                "totalTokens": 2150,
                "cacheReadInputTokens": 2000,
            }
        )
        assert result.usage.prompt_tokens == 2100
        assert result.usage.prompt_cached_tokens == 2000
        assert result.usage.cache_read_tokens == 2000

    def test_cache_writes_are_included_too(self):
        """Writing the cache is billed, at a premium, and must not vanish."""
        result = _parse_metadata(
            {
                "inputTokens": 100,
                "outputTokens": 50,
                "totalTokens": 1650,
                "cacheWriteInputTokens": 1500,
            }
        )
        assert result.usage.prompt_tokens == 1600
        assert result.usage.cache_creation_tokens == 1500

    def test_total_equals_prompt_plus_completion(self):
        """total_tokens is documented as completion + prompt, so it must agree."""
        result = _parse_metadata(
            {
                "inputTokens": 100,
                "outputTokens": 50,
                "totalTokens": 999999,  # deliberately inconsistent upstream value
                "cacheReadInputTokens": 300,
                "cacheWriteInputTokens": 200,
            }
        )
        assert result.usage.prompt_tokens == 600
        assert result.usage.total_tokens == 650
        assert (
            result.usage.total_tokens == result.usage.prompt_tokens + result.usage.completion_tokens
        )

    def test_uncached_request_is_unchanged(self):
        """No cache in play means the previous arithmetic was already right."""
        result = _parse_metadata({"inputTokens": 100, "outputTokens": 50, "totalTokens": 150})
        assert result.usage.prompt_tokens == 100
        assert result.usage.completion_tokens == 50
        assert result.usage.total_tokens == 150
        assert result.usage.prompt_cached_tokens == 0
        assert result.usage.cache_creation_tokens == 0

    def test_null_cache_fields_are_treated_as_zero(self):
        """Bedrock may send the keys with null rather than omitting them."""
        result = _parse_metadata(
            {
                "inputTokens": 100,
                "outputTokens": 50,
                "totalTokens": 150,
                "cacheReadInputTokens": None,
                "cacheWriteInputTokens": None,
            }
        )
        assert result.usage.prompt_tokens == 100
        assert result.usage.total_tokens == 150
