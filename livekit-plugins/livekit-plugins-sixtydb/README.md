# 60db TTS plugin for LiveKit Agents

60db is a hosted speech API with workspace-scoped voices. This optional plugin
uses its authenticated HTTP synthesis endpoint and outputs mono PCM16 at 24 kHz.

```bash
pip install livekit-plugins-sixtydb
export SIXTYDB_API_KEY="your-workspace-api-key"
```

```python
from livekit.agents import AgentSession
from livekit.plugins import sixtydb

session = AgentSession(
    tts=sixtydb.TTS(voice_id="your-workspace-voice-id"),
    # Configure STT and LLM separately for your application.
)
```

Obtain a voice ID from your workspace's [voice catalog](https://docs.60db.ai).
An explicit `api_key` overrides the environment variable. Optional `speed`
ranges from 0.5 to 2.0. `update_options(voice_id=..., speed=...)` affects future
requests; active requests retain their original settings.

`AgentSession` adapts incremental text into sentence synthesis requests. Native
WebSocket input streaming is not implemented. Responses are buffered (maximum
32 MiB) and validated before playback, so time to first audio includes the full
HTTP synthesis, including all pieces of a long sentence. Longer input is split
into requests of at most 5000 characters while preserving speech text and audio
order. Trailing whitespace and whitespace-only pieces are not synthesized.
Each request has a 60-second total
timeout, and uses LiveKit's connection options for connect/read timeouts.

NDJSON encoding declarations apply until another declaration or a recognized WAV
container is encountered. Unlabeled PCM chunks retain their sample bytes, even
when they begin with an audio-file signature. Explicit `pcm` or `wav` metadata
resolves ambiguous payloads. Binary `audio/pcm` responses likewise retain their
sample bytes; an audio-file prefix cannot identify compression in declared raw
PCM. Untyped binary responses are inferred and reject known compressed prefixes.

An injected `http_session` belongs to the caller and is not closed by the plugin.
Without one, the plugin reuses LiveKit's managed HTTP session. Text is sent to
60db when this provider is selected; keep API keys on the agent server.

API reference: https://docs.60db.ai/api-reference/tts/text-to-speech
