import pytest

from livekit.agents.inference.tts import (
    TTS,
    FallbackModel,
    _normalize_fallback,
    _parse_model_string,
)


def _make_tts(**kwargs):
    """Helper to create TTS with required credentials."""
    defaults = {
        "model": "cartesia/sonic",
        "api_key": "test-key",
        "api_secret": "test-secret",
        "base_url": "https://example.livekit.cloud",
    }
    defaults.update(kwargs)
    return TTS(**defaults)


class TestParseModelString:
    def test_simple_model_without_voice(self):
        """Model string without voice suffix returns None for voice."""
        model, voice = _parse_model_string("cartesia")
        assert model == "cartesia"
        assert voice is None

    def test_model_with_voice_suffix(self):
        """Model string with :voice suffix extracts the voice."""
        model, voice = _parse_model_string("cartesia:my-voice-id")
        assert model == "cartesia"
        assert voice == "my-voice-id"

    def test_provider_model_format_without_voice(self):
        """Provider/model format without voice suffix."""
        model, voice = _parse_model_string("cartesia/sonic")
        assert model == "cartesia/sonic"
        assert voice is None

    def test_provider_model_format_with_voice(self):
        """Provider/model format with voice suffix."""
        model, voice = _parse_model_string("cartesia/sonic:my-voice-id")
        assert model == "cartesia/sonic"
        assert voice == "my-voice-id"

    @pytest.mark.parametrize(
        "model_str,expected_model,expected_voice",
        [
            ("elevenlabs/eleven_flash_v2:voice123", "elevenlabs/eleven_flash_v2", "voice123"),
            ("rime:speaker-a", "rime", "speaker-a"),
            ("rime/mistv2:narrator", "rime/mistv2", "narrator"),
            ("inworld/inworld-tts-1:character", "inworld/inworld-tts-1", "character"),
            ("cartesia/sonic-turbo:deep-voice", "cartesia/sonic-turbo", "deep-voice"),
        ],
    )
    def test_various_providers_and_voices(self, model_str, expected_model, expected_voice):
        """Test various provider/model combinations with different voices."""
        model, voice = _parse_model_string(model_str)
        assert model == expected_model
        assert voice == expected_voice

    def test_empty_voice_after_colon(self):
        """Empty string after colon still counts as voice."""
        model, voice = _parse_model_string("cartesia/sonic:")
        assert model == "cartesia/sonic"
        assert voice == ""


class TestNormalizeFallback:
    def test_single_string_model(self):
        """Single string model becomes a list with one FallbackModel."""
        result = _normalize_fallback("cartesia/sonic")
        assert result == [{"model": "cartesia/sonic", "voice": ""}]

    def test_single_string_model_with_voice(self):
        """Single string model with voice suffix extracts voice."""
        result = _normalize_fallback("cartesia/sonic:my-voice")
        assert result == [{"model": "cartesia/sonic", "voice": "my-voice"}]

    def test_single_fallback_model_dict(self):
        """Single FallbackModel dict becomes a list with that dict."""
        fallback = FallbackModel(model="cartesia/sonic", voice="narrator")
        result = _normalize_fallback(fallback)
        assert result == [{"model": "cartesia/sonic", "voice": "narrator"}]

    def test_list_of_string_models(self):
        """List of string models becomes list of FallbackModels."""
        result = _normalize_fallback(["cartesia/sonic", "elevenlabs/eleven_flash_v2"])
        assert result == [
            {"model": "cartesia/sonic", "voice": ""},
            {"model": "elevenlabs/eleven_flash_v2", "voice": ""},
        ]

    def test_list_of_string_models_with_voices(self):
        """List of string models with voice suffixes."""
        result = _normalize_fallback(["cartesia/sonic:voice1", "elevenlabs:voice2"])
        assert result == [
            {"model": "cartesia/sonic", "voice": "voice1"},
            {"model": "elevenlabs", "voice": "voice2"},
        ]

    def test_list_of_fallback_model_dicts(self):
        """List of FallbackModel dicts is preserved."""
        fallbacks = [
            FallbackModel(model="cartesia/sonic", voice="narrator"),
            FallbackModel(model="elevenlabs", voice=""),
        ]
        result = _normalize_fallback(fallbacks)
        assert result == [
            {"model": "cartesia/sonic", "voice": "narrator"},
            {"model": "elevenlabs", "voice": ""},
        ]

    def test_mixed_list_strings_and_dicts(self):
        """Mixed list of strings and FallbackModel dicts."""
        fallbacks = [
            "cartesia/sonic:voice1",
            FallbackModel(model="elevenlabs/eleven_flash_v2", voice="custom"),
            "rime/mist",
        ]
        result = _normalize_fallback(fallbacks)
        assert result == [
            {"model": "cartesia/sonic", "voice": "voice1"},
            {"model": "elevenlabs/eleven_flash_v2", "voice": "custom"},
            {"model": "rime/mist", "voice": ""},
        ]

    def test_fallback_model_with_extra_kwargs(self):
        """FallbackModel with extra_kwargs is preserved."""
        fallback = FallbackModel(
            model="cartesia/sonic",
            voice="narrator",
            extra_kwargs={"duration": 30.0, "speed": "fast"},
        )
        result = _normalize_fallback(fallback)
        assert result == [
            {
                "model": "cartesia/sonic",
                "voice": "narrator",
                "extra_kwargs": {"duration": 30.0, "speed": "fast"},
            }
        ]

    def test_list_with_extra_kwargs_preserved(self):
        """List with FallbackModels containing extra_kwargs."""
        fallbacks = [
            FallbackModel(model="cartesia/sonic", voice="v1", extra_kwargs={"speed": "slow"}),
            "elevenlabs:voice2",
            FallbackModel(model="rime/mist", voice="", extra_kwargs={"custom": True}),
        ]
        result = _normalize_fallback(fallbacks)
        assert result == [
            {"model": "cartesia/sonic", "voice": "v1", "extra_kwargs": {"speed": "slow"}},
            {"model": "elevenlabs", "voice": "voice2"},
            {"model": "rime/mist", "voice": "", "extra_kwargs": {"custom": True}},
        ]

    def test_empty_list(self):
        """Empty list returns empty list."""
        result = _normalize_fallback([])
        assert result == []

    def test_fallback_model_with_none_voice(self):
        """FallbackModel with explicit None voice."""
        fallback = FallbackModel(model="cartesia/sonic", voice="")
        result = _normalize_fallback(fallback)
        assert result == [{"model": "cartesia/sonic", "voice": ""}]
