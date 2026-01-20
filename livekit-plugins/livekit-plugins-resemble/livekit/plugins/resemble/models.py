from enum import Enum
from typing import Literal


class Precision(str, Enum):
    PCM_16 = "PCM_16"


TTSModels = Literal["resemble", "chatterbox", "chatterbox-turbo"]
