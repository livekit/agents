from typing import Literal

ChatModels = Literal[
    "mistral-large-latest",
    "mistral-large-4",
    "mistral-large-2512",
    "mistral-medium-latest",
    "mistral-medium-3-5",
    "mistral-small-latest",
    "mistral-small-2603",
]

STTModels = Literal[
    "voxtral-mini-transcribe-realtime-3",
    "voxtral-mini-transcribe-3",
    "voxtral-small-transcribe-3",
    "voxtral-mini-transcribe-realtime-2602",
    "voxtral-mini-2602",
]

TTSModels = Literal[
    "voxtral-mini-tts-3",
    "voxtral-mini-tts-latest",
    "voxtral-mini-tts-2603",
]

TTSVoxtral2Voices = Literal[
    "gb_jane_confident",
    "gb_jane_confused",
    "gb_jane_curious",
    "gb_jane_frustrated",
    "gb_jane_jealousy",
    "gb_jane_neutral",
    "gb_jane_sad",
    "gb_jane_sarcasm",
    "gb_jane_shameful",
    "fr_marie_angry",
    "fr_marie_curious",
    "fr_marie_excited",
    "fr_marie_happy",
    "fr_marie_neutral",
    "fr_marie_sad",
    "gb_oliver_angry",
    "gb_oliver_cheerful",
    "gb_oliver_confident",
    "gb_oliver_curious",
    "gb_oliver_excited",
    "gb_oliver_neutral",
    "gb_oliver_sad",
    "en_paul_angry",
    "en_paul_cheerful",
    "en_paul_confident",
    "en_paul_excited",
    "en_paul_frustrated",
    "en_paul_happy",
    "en_paul_neutral",
    "en_paul_sad",
]

# only supported by the Voxtral 3 models, which also accept the Voxtral 2 voices
TTSVoxtral3Voices = Literal[
    "voice_479",  # Maryam, ar
    "voice_3143",  # Jordi, ca
    "voice_481",  # Heidi, de
    "voice_1788",  # David, en-GB
    "voice_591",  # Fred, en-GB
    "voice_590",  # Harriet, en-GB
    "voice_3147",  # Howard, en-GB
    "voice_3149",  # Hugo, en-GB
    "voice_1787",  # Mary, en-GB
    "voice_1786",  # Sky, en-GB
    "voice_3150",  # James, en-US
    "voice_447",  # Margaret, en-US
    "voice_485",  # Rob, en-US
    "voice_478",  # Sheila, en-US
    "voice_483",  # Elena, es
    "voice_3141",  # Camille, fr
    "voice_589",  # Lea, fr
    "voice_3137",  # Manon, fr
    "voice_592",  # Marie, fr
    "voice_3139",  # Ananya, hi
    "voice_3153",  # Himanshu, hi
    "voice_3146",  # Olivia, hu
    "voice_482",  # Lorenzo, it
    "voice_3133",  # Haruto, ja
    "voice_3135",  # Jin, ko
    "voice_3136",  # Sung, ko
    "voice_3145",  # Lotte, nl
    "voice_3134",  # Lars, no
    "voice_3154",  # Leon, pl
    "voice_480",  # Gabriela, pt
    "voice_3155",  # Irina, ro
    "voice_3132",  # Tatiana, ru
    "voice_3138",  # Oran, tr
    "voice_3140",  # Taras, uk
    "voice_3144",  # Huong, vi
    "voice_3142",  # Mei, zh
]

TTSVoices = TTSVoxtral2Voices | TTSVoxtral3Voices
