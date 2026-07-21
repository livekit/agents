from typing import Literal

###############################################################################
#                                    STT                                       #
###############################################################################

GnaniSTTLanguages = Literal[
    "bn-IN",
    "en-IN",
    "gu-IN",
    "hi-IN",
    "kn-IN",
    "ml-IN",
    "mr-IN",
    "pa-IN",
    "ta-IN",
    "te-IN",
    "en-IN,hi-IN",
]
"""BCP-47 locale codes accepted by Prisma STT. A comma-separated pair such as
``"en-IN,hi-IN"`` enables auto-detection. See
https://docs.gnani.ai/api/STT/stt-websocket#supported-languages"""

###############################################################################
#                                    TTS                                       #
###############################################################################

DEFAULT_MODEL = "timbre-v2.5"

GnaniTTSModels = Literal["timbre-v2.0", "timbre-v2.5"]

# Default-model (timbre-v2.0) voices, for ``voice=`` editor autocomplete. The
# full voice list depends on the model — see
# https://docs.gnani.ai/api/TTS/tts-sse#available-voices — so any voice string
# is accepted and validated by the Gnani API at request time.
GnaniTTSVoices = Literal["Pranav", "Kaveri", "Shubhra", "Deepak"]

GnaniTTSLanguages = Literal[
    "auto",
    "hi-IN",
    "en-IN",
    "ta-IN",
    "te-IN",
    "kn-IN",
    "ml-IN",
    "mr-IN",
    "bn-IN",
    "gu-IN",
    "pa-IN",
]
"""``language`` is honored by ``timbre-v2.5`` only. See
https://docs.gnani.ai/api/TTS/tts-inference#supported-languages"""

GnaniTTSEncodings = Literal["linear_pcm", "oggopus", "pcm_mulaw", "pcm_alaw"]
GnaniTTSContainers = Literal["raw", "mp3", "wav", "mulaw", "alaw", "ogg"]
GnaniTTSBitrates = Literal["32k", "64k", "96k", "128k", "192k"]
