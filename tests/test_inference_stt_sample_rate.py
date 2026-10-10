import pytest

from livekit.agents.inference.stt import STT

pytestmark = pytest.mark.unit


def _make_stt(**kwargs):
    """Helper to create STT with required credentials."""
    defaults = {
        "model": "deepgram/nova-3",
        "api_key": "test-key",
        "api_secret": "test-secret",
        "base_url": "https://example.livekit.cloud",
    }
    defaults.update(kwargs)
    return STT(**defaults)


class TestDefaultSampleRate:
    def test_openai_model_defaults_to_24khz(self):
        """OpenAI transcription sessions accept only 24 kHz PCM, so openai/* must not
        inherit the 16 kHz default the other inference providers expect."""
        assert _make_stt(model="openai/gpt-live-transcribe")._opts.sample_rate == 24000

    def test_openai_model_with_language_suffix_defaults_to_24khz(self):
        """The :language suffix is stripped before the rate is resolved."""
        assert _make_stt(model="openai/gpt-live-transcribe:en")._opts.sample_rate == 24000

    @pytest.mark.parametrize(
        "model",
        ["deepgram/nova-3", "cartesia/ink-whisper", "assemblyai/u3-rt-pro", "auto"],
    )
    def test_non_openai_models_keep_16khz(self, model):
        assert _make_stt(model=model)._opts.sample_rate == 16000

    def test_explicit_sample_rate_wins(self):
        assert (
            _make_stt(model="openai/gpt-live-transcribe", sample_rate=8000)._opts.sample_rate
            == 8000
        )
