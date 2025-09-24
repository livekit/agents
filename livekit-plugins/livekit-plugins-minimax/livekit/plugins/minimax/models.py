# livekit/plugins/minimax/models.py

from typing import Literal

# Minimax TTS Models
# See official documentation for details:
# https://www.minimax.io/platform/document/T2A%20V2?key=66719005a427f0c8a5701643#k6wO

TTSEncoding = Literal[
    "pcm_s16le",
    # Not yet supported by this plugin
    # "pcm_f32le",
    # "pcm_mulaw",
    # "pcm_alaw",
]
TTSModels = Literal["speech-02-turbo", "speech-2.5-turbo-preview"]
TTSLanguages = Literal['Chinese', 'Chinese,Yue', 'English', 'Arabic', 'Russian', 'Spanish', 'French', 'Portuguese', 'German', 'Turkish', 'Dutch', 'Ukrainian', 'Vietnamese', 'Indonesian', 'Japanese', 'Italian', 'Korean', 'Thai', 'Polish', 'Romanian', 'Greek', 'Czech', 'Finnish', 'Hindi', 'Bulgarian', 'Danish', 'Hebrew', 'Malay', 'Persian', 'Slovak', 'Swedish', 'Croatian', 'Filipino', 'Hungarian', 'Norwegian', 'Slovenian', 'Catalan', 'Nynorsk', 'Tamil', 'Afrikaans', 'auto']
TTSDefaultLanguage = "English"
TTSDefaultVoiceId = "English_radiant_girl"
TTSVoiceEmotions = Literal["happy", "sad", "angry", "fearful", "disgusted", "surprised", "neutral"]

# Minimax TTS Voice IDs
# Defines all supported voices using a Literal type for static analysis.
TTSVoices = Literal[
    "voice_agent_Female_Phone_4",
    "voice_agent_Male_Phone_1",
    "English_StressedLady",
    "English_SentimentalLady",
    "English_WiseScholar",
    "English_radiant_girl",
    "moss_audio_84f32de9-2363-11f0-b7ab-d255fae1f27b",
    "japanese_male_social_media_1_v2",
    "japanese_female_social_media_1_v2",
    "French_CasualMan",
    "French_Female Journalist",
    "Spanish_Narrator",
    "Spanish_WiseScholar",
    "Spanish_ThoughtfulMan",
    "Arabic_CalmWoman",
    "Arabic_FriendlyGuy",
    "Portuguese_ThoughtfulLady",
    "German_PlayfulMan",
    "German_SweetLady"
]

# A list of supported voices for runtime checks and internal logic.
# This should match the TTSVoices Literal above.
SUPPORTED_VOICES = [
    
]
TTSSubtitleType = Literal["word", "sentence"]
TTSDefaultEmotion = None
TTSEmotion = Literal["", "happy", "sad", "angry", "fearful", "disgusted", "surprised", "neutral"]

# Audio Format Encodings
# https://www.minimax.io/platform/document/T2A%20V2?key=66719005a427f0c8a5701643#Wb2j
TTSAudioFormats = Literal[
    "pcm",
    "mp3",
    "flac",
    "wav"
]

TTSDefaultAudioFormats = "pcm"

# Sample Rates for PCM format
TTSSampleRates = Literal[
    8000, 16000, 22050, 24000, 32000, 44100
]
TTSDefaultSampleRates = 32000


# Bit Rates (or bit depth) for PCM format
TTSBitRates = Literal[32000, 64000, 128000, 256000]
TTSDefaultBitRates = 128000