# Soniox plugin for LiveKit Agents

Support for Soniox [Speech-to-Text](https://soniox.com/docs/stt) and [Text-to-Speech](https://soniox.com/docs/tts) APIs, using WebSocket streaming interfaces.

See [STT documentation](https://docs.livekit.io/agents/models/stt/soniox/) and [TTS documentation](https://docs.livekit.io/agents/models/tts/soniox/) for more information.

## Installation

```bash
pip install livekit-plugins-soniox
```

## Pre-requisites

The Soniox plugin requires an API key to authenticate. You can get your Soniox API key [here](https://console.soniox.com/).

Set API key in your `.env` file:

```
SONIOX_API_KEY=<your_soniox_api_key>
```

## Usage

### Speech-to-Text (STT)

Use Soniox STT in an `AgentSession` or as a standalone transcription service:

```python
from livekit.plugins import soniox

session = AgentSession(
    stt=soniox.STT(),
    # ... llm, tts, etc.
)
```

### Text-to-Speech (TTS)

Use Soniox TTS for real-time speech synthesis:

```python
from livekit.plugins import soniox

session = AgentSession(
    tts=soniox.TTS(
        language="en",
        voice="Maya",
    ),
    # ... stt, llm, etc.
)
```

The TTS supports real-time streaming from LLM - text chunks are tokenized and sent to Soniox as words are formed, enabling low-latency speech synthesis.

## More information and reference

Explore integration details and find comprehensive examples:
- [Soniox STT LiveKit integration guide](https://soniox.com/docs/integrations/livekit)
- [Soniox API reference](https://soniox.com/docs/api-reference)
- [Soniox STT languages](https://soniox.com/docs/stt/concepts/supported-languages) and [Soniox TTS languages](https://soniox.com/docs/tts/concepts/languages)
- [Soniox TTS voices](https://soniox.com/docs/tts/concepts/voices)
