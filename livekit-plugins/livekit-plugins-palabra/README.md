# Palabra plugin for LiveKit Agents

Support for Palabra realtime [Speech-to-Text](https://platform.palabra.ai/docs/speech-to-text/realtime-stt) and [Text-to-Speech](https://platform.palabra.ai/docs/text-to-speech/realtime-tts) APIs, using WebSocket streaming interfaces.

See [STT documentation](https://docs.livekit.io/agents/models/stt/palabra/) and [TTS documentation](https://docs.livekit.io/agents/models/tts/palabra/) for more information.

## Installation

```bash
pip install livekit-plugins-palabra
```

## Pre-requisites

The Palabra plugin requires an API key to authenticate. You can create one at [platform.palabra.ai/api-keys](https://platform.palabra.ai/api-keys).

Set the API key in your `.env` file:

```
PALABRA_API_KEY=<your_palabra_api_key>
```



## Usage



### Speech-to-Text (STT)

Use Palabra STT in an `AgentSession` or as a standalone streaming transcription service. The spoken language is auto-detected by default:

```python
from livekit.agents import AgentSession
from livekit.plugins import palabra

session = AgentSession(
    stt=palabra.STT(),
    # ... llm, tts, etc.
)
```



### Live translation

Palabra STT can translate speech on the fly. With `translate_languages` set, each emitted `SpeechData` maps as: `language`/`text` = the translation; `source_languages`/`source_texts` = the original speech. Final transcripts arrive already in the target language, so a translating agent works without an LLM in the loop:

```python
stt = palabra.STT(translate_languages=["es"])
```



### Text-to-Speech (TTS)

Use Palabra TTS for real-time speech synthesis:

```python
from livekit.plugins import palabra

session = AgentSession(
    tts=palabra.TTS(
        language="en",
        voice_id="default_low",  # or "default_high", or any Palabra voice id
        speed=0.5,  # 0.0-1.0; 0.5 is natural conversational pace
    ),
    # ... stt, llm, etc.
)
```

The TTS streams over one persistent WebSocket session: text chunks coming from the LLM are tokenized into sentences and synthesized as they arrive, and an interruption cancels the in-flight synthesis server-side.

## More information and reference

- [Palabra realtime STT API](https://platform.palabra.ai/docs/speech-to-text/realtime-stt)
- [Palabra realtime TTS API](https://platform.palabra.ai/docs/text-to-speech/realtime-tts)
- [Palabra platform](https://platform.palabra.ai) — API keys and usage
- [palabra-ai Python SDK](https://pypi.org/project/palabra-ai/) — the transport this plugin builds on

