# HeyGen plugin for LiveKit Agents

Support for voice synthesis with [HeyGen Voice](https://developers.heygen.com/docs/voices/heygen-voice-speech).

Voice agents stream text through a WebSocket and get word timing with the audio. Set `use_tts_aligned_transcript=True` on `AgentSession` to use it. `synthesize()` uses HTTP.

## Installation

```bash
pip install livekit-plugins-heygen
```

## Prerequisites

You'll need an API key from HeyGen. It can be set as an environment variable: `HEYGEN_API_KEY`

## Usage

```python
from livekit.agents import AgentSession
from livekit.plugins import heygen

session = AgentSession(
    tts=heygen.TTS(),
    # ... stt, llm, vad
)
```

The defaults are the `heygen-voice-1-turbo` model and Nadine, a public voice. `model` can be `heygen-voice-1`, `heygen-voice-1-turbo` or `heygen-voice-1-turbo-free`. Turbo Free is free of charge and has no latency or availability guarantee.

## Voices

Pass any voice in your workspace as `voice_id`. To clone a voice from one recording:

```bash
curl -X POST https://api.heygen.com/v3/models/audio/voices \
  -H "x-api-key: $HEYGEN_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "mode": "instant",
    "name": "My voice",
    "language": "en",
    "audio": [{"type": "url", "url": "https://example.com/recording.mp3"}]
  }'
```

The response contains the new `voice_id`. Use it once `GET /v3/models/audio/voices/{voice_id}` reports the voice as `ACTIVE`.

Professional voice clones are not served over the WebSocket or by the Turbo models. Use `heygen-voice-1` with `streaming=False`, which sends one HTTP request per sentence:

```python
tts = heygen.TTS(voice_id="<professional voice id>", model="heygen-voice-1", streaming=False)
```

## Rate limits

HeyGen allows 30 speech requests per minute for each workspace member, and 120 on enterprise plans. Every WebSocket handshake and every turn counts as a request, so one API key carries about 6 live calls. A socket is reused when turns start less than 30 seconds apart, and an interrupted turn closes its socket. The plugin does not retry a rate limit.
