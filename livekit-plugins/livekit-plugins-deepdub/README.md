# LiveKit Plugins Deepdub

**Deepdub** streaming text-to-speech and real-time speech-to-text for [LiveKit Agents](https://docs.livekit.io/agents/).

Deepdub makes emotional text-to-speech (eTTS) in dozens of languages and real-time recognition for Hebrew and English. See [deepdub.ai](https://deepdub.ai) and the [API docs](https://docs.deepdub.ai).

## Installation

```bash
pip install livekit-plugins-deepdub
```

Set `DEEPDUB_API_KEY`, or pass `api_key=` to `TTS` and `STT`.

## Usage

### In an agent

```python
from livekit.agents import AgentSession
from livekit.plugins import deepdub, openai, silero

session = AgentSession(
    stt=deepdub.STT(language="en"),
    llm=openai.LLM(),
    tts=deepdub.TTS(voice_prompt_id="b2abb241-ac92-48bf-a890-fec03f43e209", locale="en-US"),
    vad=silero.VAD.load(),
)
```

### Text-to-speech

`TTS` uses Deepdub's streaming API (`wss://wss.deepdub.ai/ws`): the LLM's tokens are sent as they arrive and the audio streams back while it is generated. Deepdub buffers the text and chooses the segment boundaries itself. Each agent reply is one Deepdub turn, closed with `end-stream` when the agent flushes or ends its input; an interrupted reply is aborted with `cancel`. `synthesize()` speaks one complete text as a single turn.

```python
tts = deepdub.TTS(locale="en-US")
async with tts:
    stream = tts.stream()
    stream.push_text("Hello, ")
    stream.push_text("world!")
    stream.end_input()
    async for audio in stream:
        print(f"Got audio: {audio.frame.duration:.2f}s")
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key` | `str` | env `DEEPDUB_API_KEY` | Deepdub API key |
| `model` | `str` | `"dd-etts-4.0"` | Deepdub model |
| `voice_prompt_id` | `str` | English "Call Center Agent (Neutral)" | Voice prompt; see the [voice presets](https://docs.deepdub.ai/voice-presets) |
| `locale` | `str` | `"en-US"` | Locale of the text, such as `he-IL` or `es-MX`. A bare language such as `en` uses its default locale |
| `sample_rate` | `int` | `24000` | Output sample rate: 8000, 16000, 22050, 24000, 32000, 36000, 44100 or 48000 Hz |
| `enable_logging` | `bool \| None` | account default | Whether Deepdub may keep the session's text in its server-side logs |
| `base_url` | `str` | `"wss://wss.deepdub.ai"` | API host; `wss://wss.eu.deepdub.ai` for EU accounts |

### Speech-to-text

`STT` streams audio to `wss://wsgw.deepdub.ai/transcribe-websocket`: interim transcripts while the user speaks, then a final transcript when the server detects the end of the turn. A flush from the agent finalizes what has been heard so far. Only streaming recognition is supported.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key` | `str` | env `DEEPDUB_API_KEY` | Deepdub API key |
| `model` | `str` | `"stt-egf"` | Deepdub speech-to-text model |
| `language` | `str` | detected | `en` or `he` to lock the language; otherwise it is detected per utterance and may switch within a session |
| `language_hints_strict` | `bool` | `False` | Restrict detection to the hinted language |
| `context_terms` | `list[str] \| None` | `None` | Words to boost, such as names and product terms (up to 200) |
| `context_text` | `str \| None` | `None` | Earlier conversation, used as recognition history |
| `max_endpoint_delay_ms` | `int \| None` | server default | Pause, from 500 to 3000 ms, that ends a turn |
| `sample_rate` | `int` | `16000` | Sample rate of the audio sent to Deepdub |
| `client_reference_id` | `str \| None` | `None` | Your own identifier, echoed in Deepdub logs |

## License

Apache 2.0
