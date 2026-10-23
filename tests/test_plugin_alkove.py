from __future__ import annotations

import itertools

import alkove
import numpy as np
import pytest

from livekit import rtc
from livekit.agents import vad as agents_vad
from livekit.agents.voice import room_io
from livekit.plugins import alkove as alkove_plugin

from . import utils

# downloads Focus and Cue from Hugging Face
pytestmark = pytest.mark.plugin("alkove")


@pytest.fixture(scope="module")
def models() -> alkove.Models:
    return alkove.Models.load(["focus", "cue"])


async def speech(rate: int, secs: float = 12.0) -> np.ndarray:
    frames, *_ = await utils.make_test_speech(sample_rate=rate)
    pcm = np.frombuffer(rtc.combine_audio_frames(frames).data, dtype=np.int16)
    return pcm[: int(rate * secs)]


def noisy(pcm: np.ndarray) -> np.ndarray:
    noise = np.random.default_rng(0).normal(0, 1500, len(pcm))
    return np.clip(pcm + noise, -32768, 32767).astype(np.int16)


def frames(pcm: np.ndarray, rate: int, ms: int = 50) -> list[rtc.AudioFrame]:
    n = rate * ms // 1000
    return [
        rtc.AudioFrame(
            data=pcm[i : i + n].tobytes(),
            sample_rate=rate,
            num_channels=1,
            samples_per_channel=len(pcm[i : i + n]),
        )
        for i in range(0, len(pcm), n)
    ]


async def test_focus_gives_each_frame_back_at_its_size_20_ms_behind_live(models) -> None:
    pcm = noisy(await speech(24000))
    focus = alkove_plugin.Focus(models=models)
    room_io.AudioInputOptions(noise_cancellation=focus)
    ins = frames(pcm, 24000)
    out = [focus._process(f) for f in ins]
    assert [f.samples_per_channel for f in out] == [f.samples_per_channel for f in ins]
    assert all(f.sample_rate == 24000 and f.num_channels == 1 for f in out)
    # the gain control a session runs behind a selector takes them: whole 10 ms
    apm = rtc.AudioProcessingModule(auto_gain_control=True)
    for f in out[:-1]:
        apm.process_stream(
            rtc.AudioFrame(
                data=bytearray(f.data),
                sample_rate=f.sample_rate,
                num_channels=1,
                samples_per_channel=f.samples_per_channel,
            )
        )
    got = np.concatenate([np.frombuffer(f.data, dtype=np.int16) for f in out])
    live = alkove.Live(models, 24000)
    want = np.concatenate(
        [np.zeros(480, np.int16)]
        + [alkove.to_int16(live.push(pcm[i : i + 1200])[0]) for i in range(0, len(pcm), 1200)]
    )
    assert np.array_equal(got, want[: len(got)])
    assert np.abs(got[480:].astype(np.int32) - pcm[: len(got) - 480]).max() > 1000


async def test_focus_starts_over_on_a_new_rate_track_or_after_a_pause(models) -> None:
    focus = alkove_plugin.Focus(models=models)
    f48 = frames(noisy(await speech(48000, 1.0)), 48000)
    first = [focus._process(f).data.tobytes() for f in f48[:3]]
    assert focus._process(f48[0]).sample_rate == 48000
    focus._close()
    assert [focus._process(f).data.tobytes() for f in f48[:3]] == first
    focus._on_stream_info_updated(room_name="r", participant_identity="p", publication_sid="TR_2")
    assert [focus._process(f).data.tobytes() for f in f48[:3]] == first
    focus.enabled = False
    focus.enabled = True
    assert [focus._process(f).data.tobytes() for f in f48[:3]] == first
    f16 = frames(await speech(16000, 1.0), 16000)
    assert focus._process(f16[0]).sample_rate == 16000


async def run_vad(
    vad: alkove_plugin.VAD, fs: list[rtc.AudioFrame], flush_after: int | None = None
) -> list[agents_vad.VADEvent]:
    stream = vad.stream()
    for i, f in enumerate(fs):
        stream.push_frame(f)
        if flush_after is not None and i == flush_after:
            stream.flush()
    stream.end_input()
    return [e async for e in stream]


async def test_vad_says_what_the_processor_says(models) -> None:
    s, quiet = await speech(24000), np.zeros(24000, np.int16)
    pcm = np.concatenate([quiet, s[: 6 * 24000], quiet, s[6 * 24000 :], quiet])
    events = await run_vad(alkove_plugin.VAD.load(models=models), frames(pcm, 24000))
    starts = [e for e in events if e.type == agents_vad.VADEventType.START_OF_SPEECH]
    ends = [e for e in events if e.type == agents_vad.VADEventType.END_OF_SPEECH]
    p = alkove.Processor(models, cue=True, rate=24000)
    want = [s for i in range(0, len(pcm), 1200) for s in p.push(pcm[i : i + 1200]).speech]
    assert [round(e.timestamp - e.silence_duration - e.speech_duration, 9) for e in ends] == [
        round(s.start, 9) for s in want
    ]
    assert len(starts) == len(ends) == len(want) >= 2
    inference = [e for e in events if e.type == agents_vad.VADEventType.INFERENCE_DONE]
    assert abs(len(inference) * 0.03 - len(pcm) / 24000) < 0.1
    assert all(0 <= e.probability <= 1 and e.frames[0].sample_rate == 24000 for e in inference)
    assert sum(e.frames[0].samples_per_channel for e in inference) == round(
        len(inference) * 0.03 * 24000
    )
    assert all(e.frames[0].samples_per_channel >= 0.5 * 24000 for e in starts)
    assert max(e.raw_accumulated_silence for e in inference) >= 0.9
    assert alkove_plugin.VAD.load(models=models).min_silence_duration == 0.6


async def test_vad_flush_starts_a_fresh_segment(models) -> None:
    fs = frames(await speech(16000, 4.0), 16000)
    events = await run_vad(alkove_plugin.VAD.load(models=models), fs, flush_after=len(fs) // 2)
    stamps = [e.timestamp for e in events if e.type == agents_vad.VADEventType.INFERENCE_DONE]
    assert any(b < a for a, b in itertools.pairwise(stamps))


async def test_focus_keeps_the_live_track_whole_when_pre_connect_audio_comes_at_another_rate(
    models,
) -> None:
    live24, pre48 = (
        frames(noisy(await speech(24000, 3.0)), 24000),
        frames(noisy(await speech(48000, 3.0)), 48000),
    )
    alone = alkove_plugin.Focus(models=models)
    want = [alone._process(f).data.tobytes() for f in live24]
    focus = alkove_plugin.Focus(models=models)
    got = []
    for a, b in zip(live24, pre48, strict=True):
        got.append(focus._process(a).data.tobytes())
        assert focus._process(b).sample_rate == 48000
    assert got == want


def test_focus_wants_focus_and_cue() -> None:
    with pytest.raises(ValueError, match="focus"):
        alkove_plugin.Focus(models=alkove.Models.load(["cue"]))


async def test_vad_start_of_speech_keeps_its_prefix_whatever_start_secs(models) -> None:
    pcm = np.concatenate([np.zeros(48000, np.int16), await speech(24000, 6.0)])
    vad = alkove_plugin.VAD.load(models=models, speech=alkove.SpeechParams(start_secs=1.2))
    starts = [
        e
        for e in await run_vad(vad, frames(pcm, 24000))
        if e.type == agents_vad.VADEventType.START_OF_SPEECH
    ]
    assert starts
    for e in starts:
        want = round((e.speech_duration + vad.prefix_padding_duration) * 24000)
        assert abs(e.frames[0].samples_per_channel - want) <= 1


async def test_vad_long_speech_keeps_its_start_up_to_max_buffered_speech(models, caplog) -> None:
    pcm = np.concatenate(
        [np.zeros(24000, np.int16), await speech(24000, 12.0), np.zeros(24000, np.int16)]
    )
    vad = alkove_plugin.VAD.load(models=models, max_buffered_speech=2.0)
    ends = [
        e
        for e in await run_vad(vad, frames(pcm, 24000))
        if e.type == agents_vad.VADEventType.END_OF_SPEECH
    ]
    long = [e for e in ends if e.speech_duration > 3]
    assert long
    for e in long:
        start = e.timestamp - e.silence_duration - e.speech_duration
        first = round((start - vad.prefix_padding_duration) * 24000)
        got = np.frombuffer(e.frames[0].data, dtype=np.int16)
        assert len(got) == round(2.5 * 24000)
        assert np.array_equal(got, pcm[first : first + len(got)])
    assert "max_buffered_speech reached" in caplog.text


def test_vad_min_silence_duration_follows_its_speech_params(models) -> None:
    vad = alkove_plugin.VAD.load(models=models)
    assert vad.min_silence_duration == 0.6
    vad.speech = alkove.SpeechParams(stop_secs=0.3)
    assert vad.min_silence_duration == 0.3


async def test_vad_refuses_a_frame_at_another_rate(models, caplog) -> None:
    fs = frames(await speech(16000, 2.0), 16000)
    other = frames(await speech(24000, 0.5), 24000)
    events = await run_vad(alkove_plugin.VAD.load(models=models), fs[:20] + other + fs[20:])
    inference = [e for e in events if e.type == agents_vad.VADEventType.INFERENCE_DONE]
    assert all(e.frames[0].sample_rate == 16000 for e in inference)
    assert abs(len(inference) * 0.03 - 2.0) < 0.1
    assert "another sample rate" in caplog.text
