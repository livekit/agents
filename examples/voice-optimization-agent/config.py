from dataclasses import dataclass
from typing import Optional, Union
from livekit.agents.stt import STT
from livekit.agents.tts import TTS
from livekit.agents.vad import VAD
from livekit.plugins.turn_detector.multilingual import MultilingualModel

@dataclass
class AgentConfig:
    stt: STT
    tts: TTS
    vad: Optional[VAD] = None
    turn_detection: Union[str, MultilingualModel] = "vad"
    min_endpointing_delay: float = 0.5
    max_endpointing_delay: float = 6.0
    allow_interruptions: bool = True
    use_streaming_stt: bool = True
    use_ssml: bool = True
    # STT-specific parameters
    end_of_turn_confidence_threshold: float = 0.8
    min_end_of_turn_silence_when_confident: float = 0.3
    max_turn_silence: float = 1.0