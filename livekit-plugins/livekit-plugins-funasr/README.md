# LiveKit Plugins FunASR

Self-hosted speech-to-text for LiveKit Agents using [FunASR](https://github.com/modelscope/FunASR) — SenseVoice, Paraformer, Fun-ASR-Nano. Runs **locally, no cloud API**, strong on Chinese and 50+ languages.

## Install
```bash
pip install livekit-plugins-funasr
```

## Usage
```python
from livekit.plugins import funasr

# ModelScope (default hub="ms")
stt = funasr.STT(model="iic/SenseVoiceSmall", device="cuda")

# HuggingFace
stt = funasr.STT(model="FunAudioLLM/SenseVoiceSmall", hub="hf", device="cuda")
```

Non-streaming STT; LiveKit wraps it with a VAD `StreamAdapter` for real-time agents.
