from typing import Literal

TTSModels = Literal["mistv2", "mistv3", "arcana"]

# https://docs.rime.ai/api-reference/voices
ArcanaVoices = Literal[
    "luna", "celeste", "orion", "ursa", "astra", "esther", "estelle", "andromeda"
]

DefaultMistVoice = "cove"
