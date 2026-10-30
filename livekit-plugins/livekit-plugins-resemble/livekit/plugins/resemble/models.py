from enum import StrEnum
from typing import Literal


class Precision(StrEnum):
    PCM_16 = "PCM_16"


TTSModels = Literal["chatterbox", "chatterbox-turbo"]
