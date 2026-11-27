from typing import Literal

TTSModels = Literal["koyal-v1"]

TTSLanguages = Literal["hi", "bn", "ta", "te", "mr", "gu", "kn", "en-IN", "en-US", "en-GB", "en-AU"]
"""Language codes that the TTS model accepts. Each voice speaks some of these languages."""

TTSSampleRates = Literal[8000, 16000, 22050, 24000, 32000, 44100, 48000]
"""Sample rates, in Hz, of the 16-bit PCM audio."""

TTSDefaultVoiceId = "lC7qzMoRE7SSjaNCVeff"  # Hailey, a US English voice
