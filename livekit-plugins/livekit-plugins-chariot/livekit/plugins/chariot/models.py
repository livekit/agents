from typing import Literal

TTSModels = Literal["chariot-multilingual-v1", "chariot-v0"]

STTModels = Literal["chariot-1-asr-realtime"]

# Languages the speech-to-text model transcribes. Leave the language unset to detect it.
STTLanguages = Literal["en", "hi", "bn", "gu", "kn", "ml", "mr", "or", "pa", "sa", "ta", "te"]

# "balanced" favors latency. "high_accuracy" is more accurate (it writes numbers as digits,
# for example); its interim transcripts update less often.
STTModes = Literal["balanced", "high_accuracy"]

# "codemix" writes each word in the script it is usually written in, so English words in
# Hindi speech come out in Latin script. "native" writes everything in the language's script.
STTStyles = Literal["codemix", "native"]

# Mono 16-bit PCM rates the streaming API accepts.
STTSampleRates = Literal[8000, 16000, 22050, 24000, 32000, 44100, 48000]
