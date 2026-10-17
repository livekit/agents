# AddisAI plugin for LiveKit Agents

Support for [AddisAI](https://addisassistant.com/) speech and realtime voice in
LiveKit Agents.

The plugin includes:

- Batch STT for Amharic (`am`), Afaan Oromo (`om`), and Tigrinya (`ti`)
- Addis Scribe file and live Amharic transcription
- Addis Voices 2 completed-clip and persistent streaming TTS
- Full-duplex conversations with `አሌፍ-1.2-realtime-audio`

## Installation

```bash
uv add "livekit-agents[addisai]~=1.8"
```

Or install the plugin package directly:

```bash
pip install livekit-plugins-addisai
```

Set the API key on the server:

```bash
export ADDIS_API_KEY="your-api-key"
```

## Voice pipeline

The legacy `STT` uses batch recognition, so pair it with a VAD:

```python
from livekit.agents import AgentSession
from livekit.plugins import addisai, silero

session = AgentSession(
    vad=silero.VAD.load(),
    stt=addisai.STT(language="am"),
    llm="openai/gpt-4.1-mini",
    tts=addisai.TTS(language="am", voice="am-hamen"),
)
```

`STT` calls `/api/v2/stt` and returns final transcripts. Its `language` can be
`am`, `om`, or `ti`.

## Addis Scribe

Use Scribe for current Amharic file transcription and native live interim
results:

```python
stt = addisai.ScribeSTT(backend="turbo", chunk="320ms")
```

Scribe accepts 16 kHz mono PCM and creates a scoped, one-use WebSocket ticket
for each live utterance. Set `speakers=True` with the Turbo backend to retain
the provider's speaker-labelled result in transcript metadata.

## Addis Voices 2

Use `TTS` for completed clips:

```python
tts = addisai.TTS(
    language="om",
    voice="om-bikila",
    output_format="pcm_16000",
    speed=1.0,
)
```

Voice availability is dynamic. Query the AddisAI voice catalog and select a
voice matching `am`, `om`, or `ti`.

Use `StreamingTTS` for token-in/audio-out synthesis over a persistent socket:

```python
tts = addisai.StreamingTTS(language="am", voice="am-hamen")
```

It creates a scoped ticket, streams text with `text.append`, commits one stable
request ID, and consumes events through `speech.completed` so billing and the
saved clip are confirmed.

## Conversational realtime

The microphone-conversation relay is separate from streaming TTS. Use its
full-duplex model through LiveKit's duplex adapter:

```python
from livekit.agents import AgentSession, llm
from livekit.plugins import addisai

session = AgentSession(
    llm=llm.DuplexRealtimeAdapter(addisai.RealtimeModel()),
)
```

The relay accepts 16 kHz PCM and returns 24 kHz PCM. It owns turn detection,
interruptions, and response generation. Its current public protocol does not
expose custom instructions, chat history, tools, or transcripts.

## Language switching

Completed-clip clients can be updated between requests:

```python
stt.update_options(language="om")
tts.update_options(language="om", voice="om-bikila")
```

## API boundaries

`/api/v1/realtime/*` is Addis Voices 2 streaming TTS. The conversational
Realtime API uses `relay.addisassistant.com`; the two services share a name but
not a protocol. The plugin implements the current official SDK contracts using
LiveKit's asynchronous HTTP client.

## Resources

- [Speech-to-text](https://docs.addisassistant.com/docs/capabilities/speech-to-text)
- [Scribe live transcription](https://docs.addisassistant.com/docs/capabilities/speech-to-text/live)
- [Text-to-speech](https://docs.addisassistant.com/docs/capabilities/text-to-speech)
- [Streaming speech](https://docs.addisassistant.com/docs/capabilities/text-to-speech/streaming)
- [Conversational realtime](https://docs.addisassistant.com/docs/capabilities/realtime)
- [Python SDK](https://github.com/Addis-AI-Org/addisai-py)
- [LiveKit Agents](https://docs.livekit.io/agents/)
