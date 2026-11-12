"""Fish Audio plugin: TTS request building."""

from livekit.agents.types import NOT_GIVEN
from livekit.plugins.fishaudio.tts import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    _build_tts_request,
    _TTSOptions,
)


def test_normalize_loudness_with_speed_and_volume() -> None:
    """https://github.com/livekit/agents/pull/6520"""
    opts = _TTSOptions(
        model=DEFAULT_MODEL,
        output_format="wav",
        sample_rate=24000,
        voice_id=NOT_GIVEN,
        base_url=DEFAULT_BASE_URL,
        api_key="test-key",
        latency_mode="balanced",
        chunk_length=100,
        speed=1.2,
        volume=-3.0,
        temperature=0.7,
        top_p=0.7,
        mp3_bitrate=64,
        opus_bitrate=64000,
        normalize=True,
        normalize_loudness=True,
        max_new_tokens=NOT_GIVEN,
        min_chunk_length=NOT_GIVEN,
        condition_on_previous_chunks=NOT_GIVEN,
        early_stop_threshold=NOT_GIVEN,
    )

    prosody = _build_tts_request(opts, text="hi")["prosody"]

    assert prosody == {"speed": 1.2, "volume": -3.0, "normalize_loudness": True}
