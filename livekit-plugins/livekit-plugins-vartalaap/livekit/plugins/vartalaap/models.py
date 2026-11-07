from typing import Literal

TTSModels = Literal["narad"]
"""The Vartalaap model family. ``narad`` is the current Hinglish / Indian-English model."""

TTSVoices = Literal[
    "saanvi",
    "sahil",
    "meera",
    "mahesh",
    "ananya",
    "arjun",
    "kiara",
    "kevin",
]
"""A few of the voice ids. ``GET /v1/voices`` on the API lists every voice of the key with its
language, gender and the expressions it carries; any id from that list is accepted."""

TTSExpressions = Literal["neutral", "excited", "sad", "whisper"]
"""Expression of the delivery (the API's ``style`` field). ``neutral`` sends nothing and plays the
voice as is. A voice without the requested expression plays neutral; an unknown name is a 400.
Newer models may add expressions; the value is passed through unchanged."""

TTSDigits = Literal["auto", "hi", "mixed", "en"]
"""How digits are read in Hinglish text: ``hi`` Hindi number words, ``mixed`` codes/OTPs/phones in
English digit names with amounts in Hindi, ``en`` English number words with Indian grouping,
``auto`` picks ``en`` when the text has Latin words and ``hi`` otherwise. English voices ignore it."""

TTSProfiles = Literal["default", "telephony"]
"""Output chain profile. ``telephony`` adds a 300 Hz high-pass, a presence lift and a de-esser tuned
for 8 kHz mu-law / PSTN delivery."""
