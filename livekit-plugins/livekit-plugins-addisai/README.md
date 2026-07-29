# AddisAI plugin for LiveKit Agents

Support for [AddisAI](https://addisassistant.com/) speech-to-text and
text-to-speech services in LiveKit Agents.

The plugin supports Amharic (`am`) and Afaan Oromo (`om`):

- Batch speech recognition with `addis-whisper`
- Non-streaming speech synthesis with Addis Voices 2

## Installation

```bash
pip install livekit-plugins-addisai
```

## Prerequisites

Create an AddisAI API key and set it as an environment variable:

```bash
export ADDIS_API_KEY="your-api-key"
```

See the [AddisAI quick start](https://docs.addisassistant.com/docs/get-started/quickstart)
for account and API-key setup.

## Voice pipeline

AddisAI STT is non-streaming, so include a VAD in the agent session. The LLM can
be any model supported by LiveKit:

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

## Speech-to-text

```python
from livekit.plugins import addisai

stt = addisai.STT(language="am")
```

AddisAI STT is a batch API. In a LiveKit voice pipeline, supply a VAD so LiveKit
can segment incoming speech and adapt it to batch recognition. Transcripts are
final-only; interim transcripts are not available.

Use `language="om"` for Afaan Oromo.

## Text-to-speech

```python
from livekit.plugins import addisai

tts = addisai.TTS(
    language="am",
    voice="am-hamen",
)
```

Addis Voices 2 does not stream partial audio. LiveKit automatically adapts it by
synthesizing complete sentence chunks. The default `pcm_16000` format is
WAV-wrapped 16 kHz PCM for speech pipelines.

Voice availability is dynamic. Query the
[AddisAI voice catalog](https://docs.addisassistant.com/docs/capabilities/text-to-speech)
and pass an available voice ID matching the selected language. `am-hamen` is the
canonical Amharic example.

Optional synthesis settings include:

```python
tts = addisai.TTS(
    language="om",
    voice="your-available-oromo-voice-id",
    output_format="pcm_16000",
    speed=50.0,
)
```

The plugin preserves one AddisAI `client_request_id` across LiveKit retry
attempts to prevent duplicate generation and billing.

## Language switching

Both clients can be updated between requests:

```python
stt.update_options(language="om")
tts.update_options(language="om", voice="your-available-oromo-voice-id")
```

Only `am` and `om` are accepted.

## Other AddisAI capabilities

AddisAI also offers text generation, multimodal input, and a bidirectional
realtime audio API. They are not included in this initial plugin:

- Text streaming cannot currently be combined with tools, attachments, or audio
  input, and the raw streaming transport contract is not publicly documented.
- The realtime public protocol does not yet expose the session instructions,
  transcripts, tool calls, and interruption controls required for a complete
  LiveKit Realtime Model integration.

These capabilities can be added in focused follow-up contributions as their
public contracts mature.
