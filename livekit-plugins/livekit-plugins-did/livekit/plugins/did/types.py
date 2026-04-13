from dataclasses import dataclass

DEFAULT_SAMPLE_RATE = 24000


@dataclass
class AudioConfig:
    """Configuration for the audio sent to the D-ID avatar.

    Attributes:
        sample_rate: Sample rate in Hz. Supported values: 16000, 24000, 48000.
    """

    sample_rate: int = DEFAULT_SAMPLE_RATE
