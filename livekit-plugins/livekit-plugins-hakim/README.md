# Hakim plugin for LiveKit Agents

Support for Arabic-first realtime speech-to-text and text-to-speech from [Hakim](https://tryhakim.ai).

See [https://tryhakim.ai/docs](https://tryhakim.ai/docs) for more information.

## Installation

```bash
pip install livekit-plugins-hakim
```

## Pre-requisites

You'll need a Hakim API key. It can be set as an environment variable: `HAKIM_API_KEY`.

## Usage

```python
from livekit.plugins import hakim

session = AgentSession(
    vad=silero.VAD.load(),
    stt=hakim.STT(language="ar"),
    llm=your_llm_plugin,
    tts=hakim.TTS(voice="your-voice-id"),
)
```
