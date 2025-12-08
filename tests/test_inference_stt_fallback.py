import pytest

from livekit.agents.inference.stt import (
    STT,
    FallbackModel,
    _normalize_fallback,
    _parse_model_string,
)
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
)


def _make_stt(**kwargs):
    """Helper to create STT with required credentials."""
    defaults = {
        "model": "deepgram",
        "api_key": "test-key",
        "api_secret": "test-secret",
        "base_url": "https://example.livekit.cloud",
    }
    defaults.update(kwargs)
    return STT(**defaults)


class TestParseModelString:
    def test_simple_model_without_language(self):
        """Model string without language suffix returns NOT_GIVEN for language."""
        model, language = _parse_model_string("deepgram")
        assert model == "deepgram"
        assert language is NOT_GIVEN

    def test_model_with_language_suffix(self):
        """Model string with :language suffix extracts the language."""
        model, language = _parse_model_string("deepgram:en")
        assert model == "deepgram"
        assert language == "en"

    def test_provider_model_format_without_language(self):
        """Provider/model format without language suffix."""
        model, language = _parse_model_string("deepgram/nova-3")
        assert model == "deepgram/nova-3"
        assert language is NOT_GIVEN

    def test_provider_model_format_with_language(self):
        """Provider/model format with language suffix."""
        model, language = _parse_model_string("deepgram/nova-3:en")
        assert model == "deepgram/nova-3"
        assert language == "en"

    @pytest.mark.parametrize(
        "model_str,expected_model,expected_lang",
        [
            ("cartesia/ink-whisper:de", "cartesia/ink-whisper", "de"),
            ("assemblyai:es", "assemblyai", "es"),
            ("deepgram/nova-2-medical:ja", "deepgram/nova-2-medical", "ja"),
            ("deepgram/nova-3:multi", "deepgram/nova-3", "multi"),
            ("cartesia:zh", "cartesia", "zh"),
        ],
    )
    def test_various_providers_and_languages(self, model_str, expected_model, expected_lang):
        """Test various provider/model combinations with different languages."""
        model, language = _parse_model_string(model_str)
        assert model == expected_model
        assert language == expected_lang

    def test_auto_model(self):
        """Auto model without language."""
        model, language = _parse_model_string("auto")
        assert model == "auto"
        assert language is NOT_GIVEN

    def test_auto_model_with_language(self):
        """Auto model with language suffix."""
        model, language = _parse_model_string("auto:pt")
        assert model == "auto"
        assert language == "pt"


class TestNormalizeFallback:
    def test_single_string_model(self):
        """Single string model becomes a list with one FallbackModel."""
        result = _normalize_fallback("deepgram/nova-3")
        assert result == [{"model": "deepgram/nova-3"}]

    def test_single_fallback_model_dict(self):
        """Single FallbackModel dict becomes a list with that dict."""
        fallback = FallbackModel(model="deepgram/nova-3")
        result = _normalize_fallback(fallback)
        assert result == [{"model": "deepgram/nova-3"}]

    def test_list_of_string_models(self):
        """List of string models becomes list of FallbackModels."""
        result = _normalize_fallback(["deepgram/nova-3", "cartesia/ink-whisper"])
        assert result == [
            {"model": "deepgram/nova-3"},
            {"model": "cartesia/ink-whisper"},
        ]

    def test_list_of_fallback_model_dicts(self):
        """List of FallbackModel dicts is preserved."""
        fallbacks = [
            FallbackModel(model="deepgram/nova-3"),
            FallbackModel(model="assemblyai"),
        ]
        result = _normalize_fallback(fallbacks)
        assert result == [
            {"model": "deepgram/nova-3"},
            {"model": "assemblyai"},
        ]

    def test_mixed_list_strings_and_dicts(self):
        """Mixed list of strings and FallbackModel dicts."""
        fallbacks = [
            "deepgram/nova-3",
            FallbackModel(model="cartesia/ink-whisper"),
            "assemblyai",
        ]
        result = _normalize_fallback(fallbacks)
        assert result == [
            {"model": "deepgram/nova-3"},
            {"model": "cartesia/ink-whisper"},
            {"model": "assemblyai"},
        ]

    def test_string_with_language_suffix_discards_language(self):
        """Language suffix in string model is discarded."""
        result = _normalize_fallback("deepgram/nova-3:en")
        assert result == [{"model": "deepgram/nova-3"}]

    def test_fallback_model_with_extra_kwargs(self):
        """FallbackModel with extra_kwargs is preserved."""
        fallback = FallbackModel(
            model="deepgram/nova-3",
            extra_kwargs={"keywords": [("livekit", 1.5)], "punctuate": True},
        )
        result = _normalize_fallback(fallback)
        assert result == [
            {
                "model": "deepgram/nova-3",
                "extra_kwargs": {"keywords": [("livekit", 1.5)], "punctuate": True},
            }
        ]

    def test_list_with_extra_kwargs_preserved(self):
        """List with FallbackModels containing extra_kwargs."""
        fallbacks = [
            FallbackModel(model="deepgram/nova-3", extra_kwargs={"punctuate": True}),
            "cartesia/ink-whisper",
            FallbackModel(model="assemblyai", extra_kwargs={"format_turns": True}),
        ]
        result = _normalize_fallback(fallbacks)
        assert result == [
            {"model": "deepgram/nova-3", "extra_kwargs": {"punctuate": True}},
            {"model": "cartesia/ink-whisper"},
            {"model": "assemblyai", "extra_kwargs": {"format_turns": True}},
        ]

    def test_empty_list(self):
        """Empty list returns empty list."""
        result = _normalize_fallback([])
        assert result == []

    def test_multiple_colons_in_model_string(self):
        """Multiple colons in model string - splits on last, discards language."""
        result = _normalize_fallback("some:model:part:fr")
        assert result == [{"model": "some:model:part"}]


class TestSTTConstructorFallbackAndConnectOptions:
    """Tests for STT constructor focusing on fallback and connect_options args."""

    def test_fallback_not_given(self):
        """When fallback is not provided, _opts.fallback is NOT_GIVEN."""
        stt = _make_stt()
        assert stt._opts.fallback is NOT_GIVEN

    def test_fallback_single_string(self):
        """Single string fallback is normalized to list of FallbackModel."""
        stt = _make_stt(fallback="cartesia/ink-whisper")
        assert stt._opts.fallback == [{"model": "cartesia/ink-whisper"}]

    def test_fallback_list_of_strings(self):
        """List of string fallbacks is normalized."""
        stt = _make_stt(fallback=["deepgram/nova-3", "assemblyai"])
        assert stt._opts.fallback == [
            {"model": "deepgram/nova-3"},
            {"model": "assemblyai"},
        ]

    def test_fallback_single_fallback_model(self):
        """Single FallbackModel is normalized to list."""
        stt = _make_stt(fallback=FallbackModel(model="deepgram/nova-3"))
        assert stt._opts.fallback == [{"model": "deepgram/nova-3"}]

    def test_fallback_with_extra_kwargs(self):
        """FallbackModel with extra_kwargs is preserved in _opts."""
        stt = _make_stt(
            fallback=FallbackModel(
                model="deepgram/nova-3",
                extra_kwargs={"punctuate": True, "keywords": [("livekit", 1.5)]},
            )
        )
        assert stt._opts.fallback == [
            {
                "model": "deepgram/nova-3",
                "extra_kwargs": {"punctuate": True, "keywords": [("livekit", 1.5)]},
            }
        ]

    def test_fallback_mixed_list(self):
        """Mixed list of strings and FallbackModels is normalized."""
        stt = _make_stt(
            fallback=[
                "deepgram/nova-3",
                FallbackModel(model="cartesia", extra_kwargs={"min_volume": 0.5}),
                "assemblyai",
            ]
        )
        assert stt._opts.fallback == [
            {"model": "deepgram/nova-3"},
            {"model": "cartesia", "extra_kwargs": {"min_volume": 0.5}},
            {"model": "assemblyai"},
        ]

    def test_fallback_string_with_language_discarded(self):
        """Language suffix in fallback string is discarded."""
        stt = _make_stt(fallback="deepgram/nova-3:en")
        assert stt._opts.fallback == [{"model": "deepgram/nova-3"}]

    def test_connect_options_not_given_uses_default(self):
        """When connect_options is not provided, uses DEFAULT_API_CONNECT_OPTIONS."""
        stt = _make_stt()
        assert stt._opts.conn_options == DEFAULT_API_CONNECT_OPTIONS

    def test_connect_options_custom_timeout(self):
        """Custom connect_options with timeout is stored."""
        custom_opts = APIConnectOptions(timeout=30.0)
        stt = _make_stt(conn_options=custom_opts)
        assert stt._opts.conn_options == custom_opts
        assert stt._opts.conn_options.timeout == 30.0

    def test_connect_options_custom_max_retry(self):
        """Custom conn_options with max_retry is stored."""
        custom_opts = APIConnectOptions(max_retry=5)
        stt = _make_stt(conn_options=custom_opts)
        assert stt._opts.conn_options == custom_opts
        assert stt._opts.conn_options.max_retry == 5

    def test_connect_options_full_custom(self):
        """Fully custom connect_options is stored correctly."""
        custom_opts = APIConnectOptions(timeout=60.0, max_retry=10, retry_interval=2.0)
        stt = _make_stt(conn_options=custom_opts)
        assert stt._opts.conn_options == custom_opts
        assert stt._opts.conn_options.timeout == 60.0
        assert stt._opts.conn_options.max_retry == 10
        assert stt._opts.conn_options.retry_interval == 2.0
