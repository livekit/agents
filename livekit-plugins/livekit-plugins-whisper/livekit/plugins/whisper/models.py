from enum import Enum


class Model(Enum):
    """Class of basic Whisper model selection options"""

    TINY = "tiny"
    TINY_EN = "tiny.en"
    BASE = "base"
    BASE_EN = "base.en"
    SMALL = "small"
    SMALL_EN = "small.en"
    DISTIL_SMALL_EN = "distil-small.en"
    MEDIUM = "medium"
    MEDIUM_EN = "medium.en"
    DISTIL_MEDIUM_EN = "distil-medium.en"
    LARGE_V1 = "large-v1"
    LARGE_V2 = "large-v2"
    LARGE_V3 = "large-v3"
    LARGE = "large"
    DISTIL_LARGE_V2 = "distil-large-v2"
    DISTIL_LARGE_V3 = "distil-large-v3"
    LARGE_V3_TURBO = "large-v3-turbo"
    TURBO = "turbo"
