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

## Live API model support

LiveKit supports both Gemini Live API on both Gemini Developer API as well as Vertex AI. However, be aware they have slightly different behavior and use different model names.

The following models are supported by Gemini Developer API:

- gemini-3.1-flash-live-preview
- gemini-2.5-flash-native-audio-preview-09-2025
- gemini-2.5-flash-native-audio-preview-12-2025

And these on Vertex AI:

- gemini-live-2.5-flash-native-audio

References:

- [Gemini API Models](https://ai.google.dev/gemini-api/docs/models)
- [Vertex Live API](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/live-api)

## Realtime translation

`RealtimeTranslationModel` wraps the Gemini Live translate model
(`gemini-3.5-live-translate-preview`) for live, low-latency speech-to-speech
translation that transcribes, translates, and synthesizes in one step — no STT/TTS/VAD
plugins are needed. Each translated utterance is a turn delimited by the server's
`turn_complete`, so it plugs straight into `AgentSession`:

```python
from livekit.agents import Agent, AgentSession
from livekit.plugins import google

session = AgentSession(
    llm=google.realtime.RealtimeTranslationModel(target_language="es"),
)
await session.start(room=ctx.room, agent=Agent(instructions=""))
```

The source-language transcript is emitted as the user transcript, and the translated
audio + target-language transcript flow through the agent's output.

Notes:

- One model instance translates into a single `target_language` (one-way). To translate
  a room into several languages, run one session per direction — see
  `examples/other/translation/realtime-multi-user-translator.py` (set
  `TRANSLATION_PROVIDER=google`).
- Requires access to the translate-preview model; override `model=` if Google publishes
  a different id. A 1008/policy error usually means the model isn't enabled for your account.
