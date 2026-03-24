# Google AI plugin for LiveKit Agents

Support for Gemini, Gemini Live, Cloud Speech-to-Text, and Cloud Text-to-Speech.

See [https://docs.livekit.io/agents/integrations/google/](https://docs.livekit.io/agents/integrations/google/) for more information.

## Installation

```bash
pip install livekit-plugins-google
```

## Pre-requisites

For credentials, you'll need a Google Cloud account and obtain the correct credentials. Credentials can be passed directly or via Application Default Credentials as specified in [How Application Default Credentials works](https://cloud.google.com/docs/authentication/application-default-credentials).

To use the STT and TTS API, you'll need to enable the respective services for your Google Cloud project.

- Cloud Speech-to-Text API
- Cloud Text-to-Speech API

## Google STT — Voice Activity Timeouts

`speech_start_timeout` and `speech_end_timeout` control Google's server-side stream lifecycle, **not** VAD or endpointing. When a timeout fires, Google closes the gRPC stream; the plugin automatically reconnects if the session is still active.

| Parameter | What it does |
|---|---|
| `speech_start_timeout` | Google closes the stream if no speech begins within this many seconds |
| `speech_end_timeout` | Google closes the stream if silence lasts this many seconds after speech ends |

Because reconnecting adds a small overhead, set `speech_end_timeout` to the minimum silence you're willing to accept before the stream resets (e.g. `0.5`–`1.0` seconds). This can reduce perceived latency for short utterances like "hi" with `chirp_3`, at the cost of a reconnect between turns.

```python
stt = google.STT(
    model="chirp_3",
    speech_start_timeout=10.0,  # close stream if user doesn't speak within 10s
    speech_end_timeout=0.8,     # close stream 800ms after speech ends, then reconnect
)
```

> **Note:** These parameters only work with V2 API models (e.g. `chirp_3`). They are silently ignored for V1 models.

## Live API model support

LiveKit supports both Gemini Live API on both Gemini Developer API as well as Vertex AI. However, be aware they have slightly different behavior and use different model names.

The following models are supported by Gemini Developer API:

- gemini-2.5-flash-native-audio-preview-09-2025
- gemini-2.5-flash-native-audio-preview-12-2025

And these on Vertex AI:

- gemini-live-2.5-flash-native-audio

References:

- [Gemini API Models](https://ai.google.dev/gemini-api/docs/models)
- [Vertex Live API](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/live-api)
