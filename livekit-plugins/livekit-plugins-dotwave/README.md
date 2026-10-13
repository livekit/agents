# LiveKit Agents Plugin for .wave

Support for [.wave](https://dotwave.ai/)'s streaming speech-to-text in LiveKit Agents.

The plugin speaks the .wave `/v1/listen` WebSocket, which follows Deepgram's streaming
protocol. See the [.wave Deepgram-compatible API docs](https://dotwave.ai/docs/deepgram/) for the
full message reference.

## Installation

```bash
pip install livekit-plugins-dotwave
```

## Pre-requisites

You'll need an API key from .wave. It can be set as an environment variable: `DOTWAVE_API_KEY`

## Usage

```python
from livekit.agents import AgentSession
from livekit.plugins import dotwave

session = AgentSession(
    stt=dotwave.STT(),  # automatic language detection
    # ... llm, tts, vad
)

# or pin a language
stt = dotwave.STT(language="pt-BR", utterance_end_ms=1000)
```

Options:

- `model`: defaults to `nemotron-asr-streaming`.
- `language`: a language tag such as `en-US` or `pt-BR`. Leave it as `None` (the default) for
  automatic language detection; the detected language is reported on each transcript.
- `interim_results`: emit interim transcripts (default `True`).
- `sample_rate`: `16000` (default) or `24000`. Input audio at another rate is converted to it.
- `utterance_end_ms`: between 1000 and 3200 to have .wave end a turn after that much silence.
  Leave it unset to let the agent's own turn detection decide; the plugin sends `Finalize` when
  the agent flushes the stream.

A .wave session stays open while audio flows; one that receives no audio for 30 seconds is closed.
In a LiveKit room the participant's audio (silence included) is forwarded continuously, so this
only matters when you stop pushing frames. `KeepAlive` messages alone do not keep a session open.

Pre-recorded (non-streaming) recognition is not available; use `stream()`.
