from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from livekit import rtc
from livekit.agents import NOT_GIVEN
from livekit.agents.voice.backchannel import (
    DEFAULT_BACKCHANNEL_OPTIONS,
    DEFAULT_BACKCHANNEL_SOURCE,
    BackchannelConfig,
    _BackchannelEmitter,
    _opts_fingerprint,
    resolve_backchannel_options,
)
from livekit.agents.voice.events import _AgentBackchannelOpportunityEvent

from .fake_tts import FakeTTS

pytestmark = [pytest.mark.unit]


def _ev(*, eot_probability: float, eot_threshold: float = 0.5) -> _AgentBackchannelOpportunityEvent:
    return _AgentBackchannelOpportunityEvent(
        probability=0.9,
        threshold=0.5,
        end_of_turn_probability=eot_probability,
        end_of_turn_threshold=eot_threshold,
    )


def _frame(value: int = 0, n: int = 160) -> rtc.AudioFrame:
    samples = np.full(n, value, dtype=np.int16)
    return rtc.AudioFrame(
        data=samples.tobytes(), sample_rate=24000, num_channels=1, samples_per_channel=n
    )


class _FakeActivity:
    def __init__(self, tts: FakeTTS | None = None) -> None:
        self.tts = tts
        self.said: list[SimpleNamespace] = []

    def say(self, text, *, audio, allow_interruptions, add_to_chat_ctx):  # type: ignore[no-untyped-def]
        self.said.append(
            SimpleNamespace(
                text=text,
                audio=audio,
                allow_interruptions=allow_interruptions,
                add_to_chat_ctx=add_to_chat_ctx,
            )
        )


async def _drain(gen) -> list[rtc.AudioFrame]:  # type: ignore[no-untyped-def]
    return [f async for f in gen]


# --- BackchannelConfig.calculate_weight ---------------------------------------------------------


def test_calculate_weight_band() -> None:
    cfg = BackchannelConfig("yeah", eot_range=(0.0, 0.15), probability=0.7)
    # eligible: eot_prob / threshold = 0.1 / 1.0 = 0.1 < 0.15
    assert cfg.calculate_weight(eot_probability=0.1, eot_threshold=1.0) == 0.7
    # excluded: 0.4 / 1.0 = 0.4 outside [0, 0.15)
    assert cfg.calculate_weight(eot_probability=0.4, eot_threshold=1.0) == 0.0
    # upper bound is exclusive
    assert cfg.calculate_weight(eot_probability=0.15, eot_threshold=1.0) == 0.0


def test_calculate_weight_zero_threshold_guard() -> None:
    cfg = BackchannelConfig("yeah")
    assert cfg.calculate_weight(eot_probability=0.0, eot_threshold=0.0) == 0.0


def test_default_tiers_are_mutually_exclusive() -> None:
    def eligible(frac: float) -> set[str | object]:
        return {
            c.source
            for c in DEFAULT_BACKCHANNEL_SOURCE
            if isinstance(c, BackchannelConfig)
            and c.calculate_weight(eot_probability=frac, eot_threshold=1.0) > 0
        }

    # at a low-eot pause only the risky words are eligible
    assert eligible(0.05) == {"okay", "yeah", "right", "i see"}
    # above the 0.15 boundary only the safe sounds are eligible (tiers are contiguous)
    assert eligible(0.3) == {"mm-hmm", "uh-huh", "hmm"}
    assert eligible(0.8) == {"mm-hmm", "uh-huh", "hmm"}


# --- resolve_backchannel_options ----------------------------------------------------------------


def test_resolve_options() -> None:
    assert resolve_backchannel_options(NOT_GIVEN) == DEFAULT_BACKCHANNEL_OPTIONS
    assert resolve_backchannel_options(True) == DEFAULT_BACKCHANNEL_OPTIONS
    assert resolve_backchannel_options(False) is None

    custom_list = ["yeah", BackchannelConfig("ok")]
    assert resolve_backchannel_options(custom_list) == {
        **DEFAULT_BACKCHANNEL_OPTIONS,
        "source": custom_list,
    }

    merged = resolve_backchannel_options({"frequency": 0.2})
    assert merged is not None
    assert merged["frequency"] == 0.2
    assert merged["source"] == DEFAULT_BACKCHANNEL_SOURCE


# --- _BackchannelEmitter ------------------------------------------------------------------------


async def test_frequency_pregate_off() -> None:
    emitter = _BackchannelEmitter({"frequency": 0.0, "source": [BackchannelConfig("yeah")]})
    emitter._cache["yeah"] = [_frame()]  # pre-warm so only the pre-gate can block
    activity = _FakeActivity()
    emitter.maybe_emit(_ev(eot_probability=0.0), activity)
    assert activity.said == []


async def test_empty_pool_no_emit() -> None:
    # frequency 1.0 always attempts, but the risky band excludes this pause
    emitter = _BackchannelEmitter(
        {"frequency": 1.0, "source": [BackchannelConfig("yeah", eot_range=(0.0, 0.15))]}
    )
    emitter._cache["yeah"] = [_frame()]
    activity = _FakeActivity()
    emitter.maybe_emit(_ev(eot_probability=0.4), activity)  # 0.4/0.5 = 0.8 outside band
    assert activity.said == []


async def test_first_use_renders_then_emits_and_caches() -> None:
    tts = FakeTTS(fake_audio_duration=0.05)
    emitter = _BackchannelEmitter(
        {"frequency": 1.0, "source": [BackchannelConfig("yeah", eot_range=(0.0, 1.0))]}
    )
    activity = _FakeActivity(tts)

    # first occurrence: cache miss -> render in the background, then emit when ready
    emitter.maybe_emit(_ev(eot_probability=0.0), activity)
    assert activity.said == []  # not emitted synchronously
    assert emitter._tasks
    for task in list(emitter._tasks):
        await task

    assert "yeah" in emitter._cache  # cached for instant replay
    assert len(activity.said) == 1  # emitted once rendered (no skipped round)
    call = activity.said[0]
    assert call.text == "yeah"  # transcript carries the spoken word
    assert call.allow_interruptions is False
    assert call.add_to_chat_ctx is False
    frames = await _drain(call.audio)
    assert frames


async def test_synth_ttff_timeout_skips_and_does_not_cache() -> None:
    tts = FakeTTS(fake_timeout=0.5)  # first frame arrives well past the 300ms budget
    emitter = _BackchannelEmitter(
        {"frequency": 1.0, "source": [BackchannelConfig("yeah", eot_range=(0.0, 1.0))]}
    )
    activity = _FakeActivity(tts)

    emitter.maybe_emit(_ev(eot_probability=0.0), activity)
    for task in list(emitter._tasks):
        await task

    assert activity.said == []
    assert "yeah" not in emitter._cache  # slow synth not cached → retried next time


async def test_anti_repeat_no_consecutive_duplicates() -> None:
    emitter = _BackchannelEmitter(
        {
            "frequency": 1.0,
            "source": [
                BackchannelConfig("aa", eot_range=(0.0, 1.0)),
                BackchannelConfig("bb", eot_range=(0.0, 1.0)),
            ],
        }
    )
    emitter._cache["aa"] = [_frame()]
    emitter._cache["bb"] = [_frame()]
    activity = _FakeActivity()

    for _ in range(6):
        emitter.maybe_emit(_ev(eot_probability=0.0), activity)

    picks = [c.text for c in activity.said]
    assert len(picks) == 6
    # hotness zeroes the just-emitted clip
    assert all(a != b for a, b in zip(picks, picks[1:], strict=False))


async def test_volume_gain_applied() -> None:
    emitter = _BackchannelEmitter(
        {"frequency": 1.0, "source": [BackchannelConfig("aa", eot_range=(0.0, 1.0), volume=0.5)]}
    )
    emitter._cache["aa"] = [_frame(value=1000)]
    activity = _FakeActivity()

    emitter.maybe_emit(_ev(eot_probability=0.0), activity)
    frames = await _drain(activity.said[0].audio)
    data = np.frombuffer(frames[0].data, dtype=np.int16)
    assert np.allclose(data, 500, atol=1)  # 1000 * 0.5


# --- cross-session disk cache -------------------------------------------------------------------


@dataclass
class _FakeOpts:
    voice: str
    model: str = "m"
    api_key: str = "secret"  # denylisted — must not affect the fingerprint


def _tts_with_opts(opts: _FakeOpts, **kwargs) -> FakeTTS:  # type: ignore[no-untyped-def]
    tts = FakeTTS(**kwargs)
    tts._opts = opts  # type: ignore[attr-defined]
    return tts


class _RaisingTTS(FakeTTS):
    def synthesize(self, text, *args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("synthesize must not be called on a disk-cache hit")


def test_opts_fingerprint_stable_and_sensitive() -> None:
    fa = _opts_fingerprint(_tts_with_opts(_FakeOpts(voice="v1")))
    assert fa is not None
    assert fa == _opts_fingerprint(_tts_with_opts(_FakeOpts(voice="v1")))  # stable
    assert fa != _opts_fingerprint(_tts_with_opts(_FakeOpts(voice="v2")))  # voice matters
    # api_key is denylisted, so rotating it does not change the key
    assert fa == _opts_fingerprint(_tts_with_opts(_FakeOpts(voice="v1", api_key="rotated")))


def test_opts_fingerprint_none_without_opts() -> None:
    assert _opts_fingerprint(FakeTTS()) is None  # no dataclass _opts → disk cache disabled


async def test_disk_cache_roundtrip(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("LIVEKIT_BACKCHANNEL_CACHE_DIR", str(tmp_path))
    source = [BackchannelConfig("yeah", eot_range=(0.0, 1.0))]

    # first session: synthesize and persist to disk (realistic clip length; a sub-100ms
    # clip would decode back to zero frames in PyAV)
    tts1 = _tts_with_opts(_FakeOpts(voice="v1"), fake_audio_duration=0.5)
    emitter1 = _BackchannelEmitter({"frequency": 1.0, "source": source})
    activity1 = _FakeActivity(tts1)
    emitter1.maybe_emit(_ev(eot_probability=0.0), activity1)
    for task in list(emitter1._tasks):
        await task

    assert len(activity1.said) == 1
    assert len(list(tmp_path.rglob("*.wav"))) == 1  # written to disk

    # second session: fresh emitter + a TTS that fails if synthesized → must load from disk
    tts2 = _RaisingTTS()
    tts2._opts = _FakeOpts(voice="v1")  # same fingerprint as tts1
    emitter2 = _BackchannelEmitter({"frequency": 1.0, "source": source})
    activity2 = _FakeActivity(tts2)
    emitter2.maybe_emit(_ev(eot_probability=0.0), activity2)
    for task in list(emitter2._tasks):
        await task

    assert len(activity2.said) == 1  # loaded from disk, no synthesis


async def test_disk_cache_disabled(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("LIVEKIT_BACKCHANNEL_CACHE_DIR", str(tmp_path))
    tts = _tts_with_opts(_FakeOpts(voice="v1"), fake_audio_duration=0.05)
    emitter = _BackchannelEmitter(
        {
            "frequency": 1.0,
            "cache": False,
            "source": [BackchannelConfig("yeah", eot_range=(0.0, 1.0))],
        }
    )
    activity = _FakeActivity(tts)
    emitter.maybe_emit(_ev(eot_probability=0.0), activity)
    for task in list(emitter._tasks):
        await task

    assert len(activity.said) == 1  # still emits (in-memory)
    assert list(tmp_path.rglob("*.wav")) == []  # but nothing persisted
