from __future__ import annotations

import array as _array
import math

from livekit import rtc


class VolumeAmplifierProcessor(rtc.FrameProcessor[rtc.AudioFrame]):
    """A ``FrameProcessor`` that scales audio sample amplitude by a fixed gain.

    Designed to be used as ``AudioInputOptions.audio_post_processor`` so it runs
    after native FFI-level filters (e.g. Krisp BVC / BVCTelephony):

    .. code-block:: python

        from livekit.agents.utils.audio_processors import VolumeAmplifierProcessor
        from livekit.plugins import noise_cancellation

        AudioInputOptions(
            noise_cancellation=noise_cancellation.BVCTelephony(),
            audio_post_processor=VolumeAmplifierProcessor(gain_db=6.0),
        )

    Args:
        gain_db: Gain in decibels (positive = louder, negative = quieter).
            Defaults to 6 dB (~2× amplitude).
    """

    def __init__(self, gain_db: float = 6.0) -> None:
        self._gain = 10 ** (gain_db / 20.0)
        self._enabled = True

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    @property
    def gain_db(self) -> float:
        """Current gain in decibels."""
        return 20.0 * math.log10(self._gain)

    @gain_db.setter
    def gain_db(self, value: float) -> None:
        self._gain = 10 ** (value / 20.0)

    def _process(self, frame: rtc.AudioFrame) -> rtc.AudioFrame:
        # AudioFrame data is interleaved int16 samples.
        samples = _array.array("h", frame.data)
        gain = self._gain
        samples = _array.array(
            "h",
            (max(-32768, min(32767, int(s * gain))) for s in samples),
        )
        return rtc.AudioFrame(
            data=samples.tobytes(),
            sample_rate=frame.sample_rate,
            num_channels=frame.num_channels,
            samples_per_channel=frame.samples_per_channel,
        )

    def _close(self) -> None:
        pass
