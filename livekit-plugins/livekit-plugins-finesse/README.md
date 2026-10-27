# Finesse plugin for LiveKit Agents

Support for text-to-speech with [Finesse](https://finesse.flamapp.ai) by Flam: realtime speech in
60 languages, including 22 Indian languages, in your own cloned voices or from the voice catalog.

## Installation

```bash
pip install livekit-plugins-finesse
```

## Pre-requisites

You'll need an API key from the [API keys page](https://finesse.flamapp.ai/dashboard/settings?tab=api-keys) of the Finesse dashboard.
Set it as an environment variable: `FINESSE_API_KEY`.

## Usage

```python
from livekit.agents import AgentSession
from livekit.plugins import finesse

session = AgentSession(
    tts=finesse.TTS(
        voice="<your voice id>",
        language="hi",
    ),
    # ... llm, stt, etc.
)
```

`voice` is the id of a ready voice: copy it from **Voices** in the dashboard, or list your voices
with the [Finesse API](https://docs.finesse.flamapp.ai/voice-api/list-voices).

| Parameter | Default | Notes |
| --- | --- | --- |
| `voice` | required | id of a ready Finesse voice, including cloned voices |
| `language` | `"en"` | language code or name, for example `"hi"`, `"ta-IN"` or `"tamil"` |
| `sample_rate` | `24000` | 8000 (telephony), or 24000 (native) to 48000 Hz |
| `num_step` | server default (32) | 1-32, quality vs latency |
| `guidance_scale` | server default (2.0) | 0-10 |
| `length_scale` | server default (1.0) | speaking rate, greater than 0 up to 5; higher is slower |
| `idle_flush_ms` | server default (800) | 200-5000 ms; speak a pending partial sentence after this long without new text |

When the user interrupts, the plugin cancels the turn on the Finesse server, so it stops
synthesizing right away. Each open connection holds one of your plan's concurrent streams.
