# Giggy TTS plugin for LiveKit Agents

Native Python integration with the [Giggy speech API](https://giggy.ai/docs/speech-api).
This contribution is not an officially released package until upstream accepts and releases it.

For development, install this directory from the Agents workspace:

```sh
uv pip install -e livekit-plugins/livekit-plugins-giggy
```

Set `GIGGY_API_KEY` and `GIGGY_VOICE_ID` to your API key and voice UUID, or pass
them explicitly:

```python
from livekit.plugins import giggy

provider = giggy.TTS(voice="YOUR_GIGGY_VOICE_UUID", speed=1.0)
```

Use `provider` as the TTS component of your existing `AgentSession`.
The model is `giggyspeech`; synthesis submits complete text to
`POST https://giggy.ai/v1/audio/speech` and delivers progressive, signed PCM16
little-endian audio at 24 kHz, mono. Speed ranges from 0.25 to 4.

Incremental text input and aligned transcripts are not supported. LiveKit's
capabilities therefore report `streaming=False`, even though audio bytes arrive
progressively over HTTP. No WebSocket or alternate provider is used.

Synthesis uses paid Streaming admission. Automatic retries are disabled even
when the caller requests them, because an interrupted request may already have
incurred a charge. No `Idempotency-Key` is sent. Cancellation closes the HTTP
response. An injected `aiohttp.ClientSession` remains owned by the caller;
otherwise the plugin uses LiveKit's shared HTTP session.

Offline tests do not require Giggy credentials or call the production service:

```sh
uv run --package livekit-plugins-giggy pytest -q livekit-plugins/livekit-plugins-giggy/tests/
```

This plugin contribution is separate from LiveKit Inference onboarding and
does not establish availability in Agent Builder.
