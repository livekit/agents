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

LiveKit supports the Gemini Live API through both the Gemini Developer API and Vertex AI. Model availability and behavior differ between APIs. Some models, such as `gemini-3.8-live`, are available on both.

For model/API pairs that appear incompatible, the plugin logs a warning and lets the API decide whether the model is available.

The following models are supported by Gemini Developer API:

- gemini-3.8-live
- gemini-3.8-live-extended-thinking
- gemini-3.1-flash-live-preview
- gemini-2.5-flash-native-audio-preview-09-2025
- gemini-2.5-flash-native-audio-preview-12-2025

On the Gemini Developer API, `gemini-3.8-live` does not support `thinking_config.thinking_level`. Use `gemini-3.8-live-extended-thinking` for configurable thinking.

And these on Vertex AI:

- gemini-3.8-live
- gemini-live-2.5-flash-native-audio

References:

- [Gemini API Models](https://ai.google.dev/gemini-api/docs/models)
- [Gemini Live API thinking](https://ai.google.dev/gemini-api/docs/live-api/thinking)
- [Vertex Live API](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/live-api)

## Live Avatar

On Vertex AI, `gemini-3.8-live` can render a lip-synced talking-head video with its speech. Set `avatar_config` on the `RealtimeModel` and publish the video with `AvatarSession`:

```python
from google.genai import types
from livekit.plugins import google

model = google.realtime.RealtimeModel(
    model="gemini-3.8-live",
    vertexai=True,
    avatar_config=types.AvatarConfig(avatar_name="Kira"),
)
session = AgentSession(llm=model)
avatar = google.realtime.AvatarSession(model)
await avatar.start(session, room=ctx.room)
await session.start(agent=MyAgent(), room=ctx.room)
```

The agent participant publishes the avatar as a camera track and a microphone track. One avatar is supported per `RealtimeModel` session.

- [Configure Live Avatars](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/live-api/configure-live-avatars)
