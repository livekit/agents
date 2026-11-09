from __future__ import annotations

from typing import Any

import pytest

from livekit.plugins.minimax import TTS

pytestmark = pytest.mark.unit


def _tts(**kwargs: Any) -> TTS:
    return TTS(api_key="test-key", **kwargs)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"speed": 0.4}, "speed must be between 0.5 and 2.0"),
        ({"speed": 2.5}, "speed must be between 0.5 and 2.0"),
        ({"intensity": 101}, "intensity must be between -100 and 100"),
        ({"intensity": -101}, "intensity must be between -100 and 100"),
        ({"timbre": 150}, "timbre must be between -100 and 100"),
        ({"emotion": "fluent"}, '"fluent" emotion is only supported by speech-2.6-* models'),
    ],
)
def test_update_options_rejects_what_the_constructor_rejects(
    kwargs: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError) as ctor_error:
        _tts(**kwargs)
    assert message in str(ctor_error.value)

    tts = _tts()
    snapshot = dict(vars(tts._opts))
    with pytest.raises(ValueError) as update_error:
        tts.update_options(**kwargs)
    assert message in str(update_error.value)
    # a rejected update leaves every option unchanged
    assert vars(tts._opts) == snapshot


def test_update_options_rejects_switching_a_fluent_tts_to_another_model() -> None:
    tts = _tts(model="speech-2.6-hd", emotion="fluent")

    with pytest.raises(ValueError, match='"fluent" emotion is only supported'):
        tts.update_options(model="speech-02-hd")
    assert tts._opts.model == "speech-2.6-hd"

    tts.update_options(model="speech-02-hd", emotion="happy")
    assert tts._opts.model == "speech-02-hd"
    assert tts._opts.emotion == "happy"


def test_update_options_accepts_in_range_values() -> None:
    tts = _tts(model="speech-2.6-turbo")

    tts.update_options(speed=2.0, intensity=-100, timbre=100, emotion="fluent")

    assert tts._opts.speed == 2.0
    assert tts._opts.intensity == -100
    assert tts._opts.timbre == 100
    assert tts._opts.emotion == "fluent"
