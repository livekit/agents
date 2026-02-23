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

- gemini-2.5-flash-native-audio-preview-09-2025
- gemini-2.5-flash-native-audio-preview-12-2025

And these on Vertex AI:

- gemini-live-2.5-flash-native-audio

References:

- [Gemini API Models](https://ai.google.dev/gemini-api/docs/models)
- [Vertex Live API](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/live-api)
