from typing import Literal

TURN_DETECTION = Literal["HIGH", "MEDIUM", "LOW"]
MODALITIES = Literal["audio", "mixed"]
REALTIME_MODELS = Literal["amazon.nova-sonic-v1:0", "amazon.nova-2-sonic-v1:0"]

SONIC1_VOICES = Literal[
    "matthew",  # English (US) - Masculine
    "tiffany",  # English (US) - Feminine
    "amy",  # English (GB) - Feminine
    "lupe",  # Spanish - Feminine
    "carlos",  # Spanish - Masculine
    "ambre",  # French - Feminine
    "florian",  # French - Masculine
    "greta",  # German - Feminine
    "lennart",  # German - Masculine
    "beatrice",  # Italian - Feminine
    "lorenzo",  # Italian - Masculine
]

SONIC2_VOICES = Literal[
    "matthew",  # English (US) - Masculine - Polyglot
    "tiffany",  # English (US) - Feminine - Polyglot
    "amy",  # English (GB) - Feminine
    "olivia",  # English (US) - Feminine
    "lupe",  # Spanish - Feminine
    "carlos",  # Spanish - Masculine
    "ambre",  # French - Feminine
    "florian",  # French - Masculine
    "tina",  # German - Feminine
    "lennart",  # German - Masculine
    "beatrice",  # Italian - Feminine
    "lorenzo",  # Italian - Masculine
    "carolina",  # Portuguese (Brazilian) - Feminine
    "leo",  # Portuguese (Brazilian) - Masculine
    "arjun",  # Hindi - Masculine
    "kiara",  # Hindi - Feminine
]
