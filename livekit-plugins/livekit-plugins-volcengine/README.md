# Volcengine plugin for LiveKit Agents

Support for text-to-speech using the
[Volcengine bidirectional WebSocket API](https://www.volcengine.com/docs/6561/1329505).

## Installation

```bash
pip install livekit-plugins-volcengine
```

## Authentication

Set `VOLCENGINE_API_KEY` and `VOLCENGINE_TTS_VOICE` in your `.env` file. You can also set
`VOLCENGINE_TTS_RESOURCE_ID`; it defaults to `seed-tts-2.0`.

## Usage

```python
from livekit.plugins import volcengine

tts = volcengine.TTS()
```

Or configure the synthesis options explicitly:

```python
from livekit.plugins import volcengine

tts = volcengine.TTS(
    voice="zh_female_cancan_mars_bigtts",
    resource_id="seed-tts-1.0",
    sample_rate=24000,
    speech_rate=0,
    loudness_rate=0,
)
```

The resource ID must match the selected voice. For example, voices ending in
`_mars_bigtts` use `seed-tts-1.0`.

The plugin uses raw PCM audio and streams text directly to Volcengine without adding its own
sentence buffering.
