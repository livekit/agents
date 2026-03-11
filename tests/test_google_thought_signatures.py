import pytest

from livekit.plugins.google.llm import (
    _is_gemini_3_flash_model,
    _is_gemini_3_model,
    _requires_thought_signatures,
)


class TestGeminiModelDetection:
    """Tests for Gemini model detection helper functions."""

    @pytest.mark.parametrize(
        "model,expected",
        [
            # Gemini 3 models - should return True
            ("gemini-3-pro-preview", True),
            ("gemini-3-flash-preview", True),
            ("gemini-3-flash", True),
            ("gemini-3-pro", True),
            ("GEMINI-3-FLASH", True),  # case insensitive
            # Gemini 2.5 models - should return False
            ("gemini-2.5-flash", False),
            ("gemini-2.5-pro-preview-05-06", False),
            # Gemini 2.0 models - should return False
            ("gemini-2.0-flash-001", False),
            # Gemini 1.5 models - should return False
            ("gemini-1.5-pro", False),
            # Other models - should return False
            ("gpt-4", False),
            ("claude-3", False),
        ],
    )
    def test_is_gemini_3_model(self, model: str, expected: bool):
        assert _is_gemini_3_model(model) == expected

    @pytest.mark.parametrize(
        "model,expected",
        [
            # Gemini 3 Flash models - should return True
            ("gemini-3-flash-preview", True),
            ("gemini-3-flash", True),
            ("GEMINI-3-FLASH", True),  # case insensitive
            # Gemini 3 Pro models - should return False
            ("gemini-3-pro-preview", False),
            ("gemini-3-pro", False),
            # Other models - should return False
            ("gemini-2.5-flash", False),
            ("gemini-2.0-flash-001", False),
        ],
    )
    def test_is_gemini_3_flash_model(self, model: str, expected: bool):
        assert _is_gemini_3_flash_model(model) == expected

    @pytest.mark.parametrize(
        "model,expected",
        [
            # Gemini 3 models - should return True (requires thought signatures)
            ("gemini-3-pro-preview", True),
            ("gemini-3-flash-preview", True),
            ("gemini-3-flash", True),
            ("gemini-3-pro", True),
            ("GEMINI-3-FLASH", True),  # case insensitive
            # Gemini 2.5 models - should return True (requires thought signatures)
            ("gemini-2.5-flash", True),
            ("gemini-2.5-pro-preview-05-06", True),
            ("gemini-2.5-flash-preview-04-17", True),
            ("gemini-2.5-flash-preview-05-20", True),
            ("GEMINI-2.5-FLASH", True),  # case insensitive
            # Gemini 2.0 models - should return False (does not require thought signatures)
            ("gemini-2.0-flash-001", False),
            ("gemini-2.0-flash-lite-preview-02-05", False),
            ("gemini-2.0-pro-exp-02-05", False),
            # Gemini 1.5 models - should return False
            ("gemini-1.5-pro", False),
            # Other models - should return False
            ("gpt-4", False),
            ("claude-3", False),
        ],
    )
    def test_requires_thought_signatures(self, model: str, expected: bool):
        """Test that thought_signature handling is enabled for Gemini 2.5+ models.

        This is the critical fix for the thought_signature error that occurs when using
        Gemini 2.5 Flash with multi-turn function calling. Google's API requires
        thought_signatures to be stored and passed back in subsequent requests.
        """
        assert _requires_thought_signatures(model) == expected
