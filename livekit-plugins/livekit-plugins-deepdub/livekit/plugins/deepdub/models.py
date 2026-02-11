from typing import Literal

TTSModels = Literal["dd-etts-1.1", "dd-etts-2.5", "dd-etts-3.0"]
TTSAudioFormat = Literal["mp3", "opus", "mulaw", "headerless-wav", "s16le"]
TTSSampleRate = Literal[8000, 16000, 22050, 24000, 44100, 48000]
