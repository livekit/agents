from typing import Literal

# Output containers Quickdial can return over REST.
TTSEncoding = Literal["pcm", "wav", "opus"]

# A few well-known voices (see GET /v1/voices for the full, live list).
TTSVoices = Literal[
    "alba", "jane", "charles", "anna", "george", "vera",
    "estelle", "giovanni", "juergen", "lola", "rafael",
]

STTLanguages = Literal["en", "fr", "de", "es", "it", "pt"]
