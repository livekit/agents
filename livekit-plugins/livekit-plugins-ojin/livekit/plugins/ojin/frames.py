from __future__ import annotations

from livekit import rtc
from ojin.stv import STVAudioFrame, STVVideoFrame

_BYTES_PER_SAMPLE = 2  # int16 PCM


def is_silence(frame: STVAudioFrame) -> bool:
    """Whether the frame's pcm is nothing but zero bytes.

    The playback loop emits one audio frame per tick forever, synthesizing fill of
    exactly this shape on every tick with no real audio to play. A real chunk that
    happens to be all zeros (a provider's pause padding) is indistinguishable and
    is dropped too: a documented, accepted 40 ms accounting shift per occurrence.
    """
    return bool(frame.pcm.strip(b"\x00") == b"")


def to_video_frame(frame: STVVideoFrame) -> rtc.VideoFrame | None:
    """Convert an Ojin video frame into a LiveKit one.

    Returns None when the frame carries no decoded pixels. The WebSocket client
    never emits such a frame (it skips the tick instead), but the STVOutput
    protocol permits it, so the guard stays. Size is read per frame: it belongs to
    the Ojin model, not the plugin (1024x1024 and 736x1216 both occur).
    """
    if frame.rgb is None:
        return None

    return rtc.VideoFrame(
        width=frame.width,
        height=frame.height,
        type=rtc.VideoBufferType.RGB24,
        data=frame.rgb,
    )


def to_audio_frame(frame: STVAudioFrame) -> rtc.AudioFrame:
    """Convert an Ojin audio frame into a LiveKit one.

    ``frame.pcm`` is the original TTS audio at the rate it was sent, so the sample
    count is derived from the payload rather than assumed.
    """
    return rtc.AudioFrame(
        data=frame.pcm,
        sample_rate=frame.sample_rate,
        num_channels=frame.num_channels,
        samples_per_channel=len(frame.pcm) // (_BYTES_PER_SAMPLE * frame.num_channels),
    )


def downmix_to_mono(data: bytes, num_channels: int) -> bytes:
    """Average interleaved int16 channels down to mono; mono passes through.

    The framework resamples the rate it hands us but never the channel count, and
    ``AvatarOptions.audio_channels`` is fixed at construction, so the plugin makes
    the input mono itself.
    """
    if num_channels == 1:
        return data

    import numpy as np  # available via ojin-client[stv]

    samples = np.frombuffer(data, dtype=np.int16).reshape(-1, num_channels)
    # Annotated rather than cast: numpy is typed or not depending on whether the
    # stub packages are installed, and this stays correct either way.
    mono: bytes = samples.mean(axis=1).astype(np.int16).tobytes()
    return mono
