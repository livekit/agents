from __future__ import annotations

import pytest

from livekit.plugins.deepgram import TTS

pytestmark = pytest.mark.plugin("deepgram")


def test_deepgram_tts_partial_update_preserves_sibling_options() -> None:
    # Initialize with comprehensive non-default configuration
    tts = TTS(
        model="aura-helios-en",
        encoding="mp3",
        sample_rate=48000,
        bit_rate=128000,
        speed=1.2,
        api_key="test-api-key",
    )

    # 1. Update only model -> preserves encoding, sample_rate, bit_rate, speed
    tts.update_options(model="aura-luna-en")
    assert tts._opts.model == "aura-luna-en"
    assert tts._opts.encoding == "mp3"
    assert tts._opts.sample_rate == 48000
    assert tts.sample_rate == 48000
    assert tts._opts.bit_rate == 128000
    assert tts._opts.speed == 1.2
    assert tts._opts.api_key == "test-api-key"

    # 2. Update only encoding -> preserves model, sample_rate, bit_rate, speed
    tts.update_options(encoding="linear16")
    assert tts._opts.encoding == "linear16"
    assert tts._opts.model == "aura-luna-en"
    assert tts._opts.sample_rate == 48000
    assert tts.sample_rate == 48000
    assert tts._opts.bit_rate == 128000
    assert tts._opts.speed == 1.2

    # 3. Update only sample_rate -> preserves model, encoding, bit_rate, speed
    tts.update_options(sample_rate=24000)
    assert tts._opts.sample_rate == 24000
    assert tts.sample_rate == 24000
    assert tts._opts.model == "aura-luna-en"
    assert tts._opts.encoding == "linear16"
    assert tts._opts.bit_rate == 128000
    assert tts._opts.speed == 1.2

    # 4. Update only speed -> preserves model, encoding, sample_rate, bit_rate
    tts.update_options(speed=0.9)
    assert tts._opts.speed == 0.9
    assert tts._opts.model == "aura-luna-en"
    assert tts._opts.encoding == "linear16"
    assert tts._opts.sample_rate == 24000
    assert tts._opts.bit_rate == 128000

    # 5. Update only bit_rate -> preserves model, encoding, sample_rate, speed
    tts.update_options(bit_rate=64000)
    assert tts._opts.bit_rate == 64000
    assert tts._opts.speed == 0.9
    assert tts._opts.model == "aura-luna-en"
    assert tts._opts.encoding == "linear16"
    assert tts._opts.sample_rate == 24000

    # 6. Clear speed with None -> preserves sibling options
    tts.update_options(speed=None)
    assert tts._opts.speed is None
    assert tts._opts.bit_rate == 64000
    assert tts._opts.model == "aura-luna-en"
    assert tts._opts.encoding == "linear16"
    assert tts._opts.sample_rate == 24000

    # 7. Clear bit_rate with None -> preserves sibling options
    tts.update_options(bit_rate=None)
    assert tts._opts.bit_rate is None
    assert tts._opts.speed is None
    assert tts._opts.model == "aura-luna-en"
    assert tts._opts.encoding == "linear16"
    assert tts._opts.sample_rate == 24000


def test_deepgram_tts_update_options_accepts_integer_speed() -> None:
    # update_options with integer speed (e.g. speed=1) must convert to float
    tts = TTS(api_key="test-api-key")
    tts.update_options(speed=1)
    assert isinstance(tts._opts.speed, float)
    assert tts._opts.speed == 1.0

    # __init__ with integer speed
    tts_int = TTS(speed=1, api_key="test-api-key")
    assert isinstance(tts_int._opts.speed, float)
    assert tts_int._opts.speed == 1.0


def test_deepgram_tts_update_options_speed_boundaries() -> None:
    tts = TTS(api_key="test-api-key")

    # Valid boundaries
    tts.update_options(speed=0.7)
    assert tts._opts.speed == 0.7

    tts.update_options(speed=1.5)
    assert tts._opts.speed == 1.5

    # Out of range (< 0.7 or > 1.5)
    with pytest.raises(ValueError, match="speed must be between 0.7 and 1.5"):
        tts.update_options(speed=0.69)

    with pytest.raises(ValueError, match="speed must be between 0.7 and 1.5"):
        tts.update_options(speed=1.51)

    # Boolean speed must be rejected with ValueError
    with pytest.raises(ValueError, match="speed must be a float or int"):
        tts.update_options(speed=True)

    with pytest.raises(ValueError, match="speed must be a float or int"):
        tts.update_options(speed=False)

    # Non-numeric string speed must be rejected with ValueError
    with pytest.raises(ValueError, match="speed must be a float or int"):
        tts.update_options(speed="not-a-number")  # type: ignore[arg-type]


def test_deepgram_tts_update_options_atomic_on_failure() -> None:
    tts = TTS(
        model="aura-helios-en",
        encoding="mp3",
        sample_rate=48000,
        bit_rate=128000,
        speed=1.2,
        api_key="test-api-key",
    )

    # 1. Invalid speed when updating model: model must NOT be updated
    with pytest.raises(ValueError):
        tts.update_options(model="aura-luna-en", speed=99.0)
    assert tts._opts.model == "aura-helios-en"
    assert tts._opts.speed == 1.2

    # 2. Invalid boolean speed: encoding must NOT be updated
    with pytest.raises(ValueError):
        tts.update_options(encoding="linear16", speed=True)
    assert tts._opts.encoding == "mp3"
    assert tts._opts.speed == 1.2

    # 3. Invalid sample_rate (negative): model must NOT be updated
    with pytest.raises(ValueError, match="sample_rate must be a positive integer"):
        tts.update_options(model="aura-luna-en", sample_rate=-8000)
    assert tts._opts.model == "aura-helios-en"
    assert tts._opts.sample_rate == 48000
    assert tts.sample_rate == 48000

    # 4. Invalid sample_rate (zero): model must NOT be updated
    with pytest.raises(ValueError, match="sample_rate must be a positive integer"):
        tts.update_options(model="aura-luna-en", sample_rate=0)
    assert tts._opts.model == "aura-helios-en"
    assert tts._opts.sample_rate == 48000

    # 5. Invalid bit_rate (negative): model must NOT be updated
    with pytest.raises(ValueError, match="bit_rate must be a positive integer"):
        tts.update_options(model="aura-luna-en", bit_rate=-1)
    assert tts._opts.model == "aura-helios-en"
    assert tts._opts.bit_rate == 128000

    # 6. Empty model string: must be rejected and preserve existing model
    with pytest.raises(ValueError, match="model must be a non-empty string"):
        tts.update_options(model="")
    assert tts._opts.model == "aura-helios-en"

    # 7. Empty encoding string: must be rejected and preserve existing encoding
    with pytest.raises(ValueError, match="encoding must be a non-empty string"):
        tts.update_options(encoding="")
    assert tts._opts.encoding == "mp3"


def test_deepgram_tts_update_options_pool_invalidation() -> None:
    tts = TTS(api_key="test-api-key")
    calls: list[bool] = []
    tts._pool.invalidate = lambda: calls.append(True)  # type: ignore[method-assign]

    # No arguments: pool must NOT be invalidated
    tts.update_options()
    assert calls == []

    # Same option value: pool must NOT be invalidated
    tts.update_options(model="aura-2-andromeda-en")
    assert calls == []

    # Changed model: pool must be invalidated
    tts.update_options(model="aura-luna-en")
    assert calls == [True]
    calls.clear()

    # Failed validation: pool must NOT be invalidated
    with pytest.raises(ValueError):
        tts.update_options(model="aura-zeus-en", speed=99.0)
    assert calls == []
    assert tts._opts.model == "aura-luna-en"
