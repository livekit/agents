from typing import Literal

###############################################################################
#                                    STT                                       #
###############################################################################

GnaniSTTModels = Literal["prisma-v2.5", "prisma-v3.0"]
"""Prisma model. The ``3`` in the ``/stt/v3`` URL is the API version, *not* the model
version: leaving ``model`` unset selects the server default, ``prisma-v2.5``."""

PRISMA_V25 = "prisma-v2.5"
PRISMA_V30 = "prisma-v3.0"
STT_MODELS = (PRISMA_V25, PRISMA_V30)
DEFAULT_STT_MODEL = PRISMA_V25
"""What the API uses when no ``model`` is sent."""

INDIC_AUTO = "indic-auto"

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
    "indic-auto",
]
"""BCP-47 locale codes accepted by Prisma STT, plus ``"indic-auto"`` (the language is
detected per utterance). With ``model="prisma-v3.0"`` you can also pass 2-4 codes as a
list (``["hi-IN", "en-IN"]``) or a comma-separated string (``"hi-IN,en-IN"``) to restrict
detection to those languages. See
https://docs.gnani.ai/api/STT/stt-websocket#supported-languages"""

GnaniSTTFormats = Literal["verbatim", "transcribe", "codemix", "transliterate"]
"""Output style. ``codemix`` and ``transliterate`` need ``model="prisma-v3.0"``."""

###############################################################################
#                                    TTS                                       #
###############################################################################

GnaniTTSModels = Literal["timbre-v2.5", "timbre-v3.0"]

DEFAULT_MODEL = "timbre-v2.5"
"""``timbre-v2.0`` has been retired by the Gnani API and is no longer accepted."""

RETIRED_MODELS = frozenset({"timbre-v2.0"})
"""Models the API rejects with ``400 ... is deprecated and is no longer available``."""

# Voices are bound to a model: using a ``timbre-v3.0`` voice with ``timbre-v2.5``
# (or the other way round) is rejected by the API with a 400. The tuples below
# are used for editor autocomplete and to fail fast on a *known* cross-model
# mismatch only. Any other voice string is passed through, so voices that Gnani
# adds later keep working without a plugin release. The API stays the source of
# truth. Catalogues as returned by the API on 2026-10-30.
TIMBRE_V25_VOICES = (
    "Aarohi",
    "Abhimanyu",
    "Ambuja",
    "Asmita",
    "Bhavini",
    "Bhavna",
    "Brinda",
    "Charvi",
    "Chirag",
    "Chitra",
    "Damini",
    "Deepak",
    "Deven",
    "Devika",
    "Dhruva",
    "Eishani",
    "Falak",
    "Falguni",
    "Farhan",
    "Gaurav",
    "Gauri",
    "Girish",
    "Hansika",
    "Harshit",
    "Hemraj",
    "Iravati",
    "Ishaan",
    "Jalaj",
    "Janaki",
    "Jatin",
    "Jwala",
    "Kamakshi",
    "Kartik",
    "Kaveri",
    "Kavin",
    "Kirra",
    "Lalita",
    "Lavanya",
    "Lehara",
    "Madhuri",
    "Mehuli",
    "Nalini",
    "Nayana",
    "Noopur",
    "Omkar",
    "Oviya",
    "Parvati",
    "Poorvi",
    "Pranav",
    "Radhika",
    "Reshma",
    "Riyaan",
    "Roopesh",
    "Saanvi",
    "Shlok",
    "Shweta",
    "Suhana",
    "Tanvi",
    "Trisha",
    "Trupti",
    "Urmila",
    "Urvashi",
    "Varuni",
    "Vedika",
    "Veera",
    "Vidya",
    "Vikrant",
    "Wamika",
    "Yamini",
    "Yashvi",
    "Yukti",
    "Zahira",
    "Zayan",
)

TIMBRE_V30_VOICES = (
    "Avinash",
    "Chetana",
    "Devansh",
    "Harshad",
    "Kalpana",
    "Lakshmi",
    "Prakash",
    "Riyaa",
    "Shivani",
    "Shubham",
    "Tarika",
    "Vasudha",
    "Yashraj",
)

DEFAULT_VOICE = "Pranav"
DEFAULT_VOICES: dict[str, str] = {
    "timbre-v2.5": "Pranav",
    "timbre-v3.0": "Avinash",
}
"""Default voice per model; used when ``voice`` is not given."""

GnaniTTSVoices = Literal[
    "Avinash",
    "Chetana",
    "Devansh",
    "Harshad",
    "Kalpana",
    "Lakshmi",
    "Prakash",
    "Riyaa",
    "Shivani",
    "Shubham",
    "Tarika",
    "Vasudha",
    "Yashraj",
    "Aarohi",
    "Abhimanyu",
    "Ambuja",
    "Asmita",
    "Bhavini",
    "Bhavna",
    "Brinda",
    "Charvi",
    "Chirag",
    "Chitra",
    "Damini",
    "Deepak",
    "Deven",
    "Devika",
    "Dhruva",
    "Eishani",
    "Falak",
    "Falguni",
    "Farhan",
    "Gaurav",
    "Gauri",
    "Girish",
    "Hansika",
    "Harshit",
    "Hemraj",
    "Iravati",
    "Ishaan",
    "Jalaj",
    "Janaki",
    "Jatin",
    "Jwala",
    "Kamakshi",
    "Kartik",
    "Kaveri",
    "Kavin",
    "Kirra",
    "Lalita",
    "Lavanya",
    "Lehara",
    "Madhuri",
    "Mehuli",
    "Nalini",
    "Nayana",
    "Noopur",
    "Omkar",
    "Oviya",
    "Parvati",
    "Poorvi",
    "Pranav",
    "Radhika",
    "Reshma",
    "Riyaan",
    "Roopesh",
    "Saanvi",
    "Shlok",
    "Shweta",
    "Suhana",
    "Tanvi",
    "Trisha",
    "Trupti",
    "Urmila",
    "Urvashi",
    "Varuni",
    "Vedika",
    "Veera",
    "Vidya",
    "Vikrant",
    "Wamika",
    "Yamini",
    "Yashvi",
    "Yukti",
    "Zahira",
    "Zayan",
]
"""Known voices across ``timbre-v2.5`` and ``timbre-v3.0``. Any string is accepted."""

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
    "hi-en",
]
"""``language`` is honored by ``timbre-v2.5`` and ``timbre-v3.0``; ``auto`` detects
it from the text. See https://docs.gnani.ai/api/TTS/tts-inference#supported-languages"""

MIN_SPEED = 0.85
MAX_SPEED = 1.15

GnaniTTSEncodings = Literal["linear_pcm", "oggopus"]
GnaniTTSContainers = Literal["raw", "mp3", "wav", "ogg"]
GnaniTTSBitrates = Literal["96k", "128k", "192k"]
