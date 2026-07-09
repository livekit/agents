# Tencent Cloud plugin for LiveKit Agents

Support for streaming speech-to-text with Tencent Cloud ASR and text-to-speech
with Tencent Cloud TTS.

## Installation

```bash
pip install livekit-plugins-tencent
```

## Authentication

Set the Tencent Cloud ASR credentials in your environment:

```bash
TENCENT_ASR_APP_ID=...
TENCENT_ASR_SECRET_ID=...
TENCENT_ASR_SECRET_KEY=...
```

Set the Tencent Cloud TTS credentials separately:

```bash
TENCENT_TTS_APP_ID=...
TENCENT_TTS_SECRET_ID=...
TENCENT_TTS_SECRET_KEY=...
```

## ASR Usage

```python
from livekit.agents import AgentSession
from livekit.plugins import tencent

session = AgentSession(
    stt=tencent.STT(),
)
```

With explicit options:

```python
from livekit.plugins import tencent

stt = tencent.STT(
    engine_model_type="16k_zh_en",
    vad_silence_time=500,
    max_speak_time=15000,
)
```

Tencent Cloud ASR currently supports streaming recognition in this plugin. Batch
recognition via `recognize()` is not implemented.

## TTS Usage

```python
from livekit.agents import AgentSession
from livekit.plugins import tencent

session = AgentSession(
    tts=tencent.TTS(),
)
```

With explicit options:

```python
from livekit.plugins import tencent

tts = tencent.TTS(
    voice_type=601010,
    codec="pcm",
    sample_rate=24000,
    speed=0.0,
    volume=0.0,
)
```
