# Meta plugin for LiveKit Agents

Support for Meta [Muse Voice Transcribe](https://ai.meta.com/) using its realtime streaming speech-to-text interface.

## Installation

```bash
pip install livekit-plugins-meta
```

## Pre-requisites

Set a Meta Model API key in your environment:

```bash
MODEL_API_KEY=<your_model_api_key>
```

You can also pass the key directly with `api_key=`.

## Usage

```python
from livekit.agents import AgentSession
from livekit.plugins import meta

session = AgentSession(
    stt=meta.STT(
        keywords=["LiveKit", "Muse"],
        language_bias=["en"],
    ),
    # ... llm, tts, etc.
)
```

The plugin supports streaming recognition with server-side endpointing, cumulative interim transcripts, and mono PCM16 audio at 24 kHz. `keywords` and `language_bias` are static hints applied when a stream starts. Batch recognition, diarization, detected-language metadata, and active-stream keyterm updates are not supported.
