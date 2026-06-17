from typing import Literal

TTSModels = Literal["openbmb/VoxCPM2"] | str
TTSVoices = Literal["default"] | str

DEFAULT_MODEL: TTSModels = "openbmb/VoxCPM2"
DEFAULT_VOICE: TTSVoices = "default"
DEFAULT_SAMPLE_RATE = 48_000
NUM_CHANNELS = 1
