# LiveKit Plugins Dialogflow

Agent Framework plugin for [Google Dialogflow CX](https://cloud.google.com/dialogflow/cx/docs). This plugin integrates Dialogflow CX as an LLM provider in a LiveKit Agents STT → LLM → TTS pipeline.

## What is Dialogflow CX?

Dialogflow CX is an intent-based conversational AI engine with its own session management, flows, pages, and fulfillment logic. Unlike standard LLMs, all conversation design happens in the [Dialogflow CX Console](https://dialogflow.cloud.google.com/cx/) — the `instructions` field on the LiveKit `Agent` class has no effect.

## Installation

```bash
pip install livekit-plugins-dialogflow
```

## Authentication

This plugin uses [Google Application Default Credentials (ADC)](https://cloud.google.com/docs/authentication/application-default-credentials). Set up authentication using one of:

- **Service account key**: Set `GOOGLE_APPLICATION_CREDENTIALS` to the path of your JSON key file
- **GCP environment**: Running on GCE, GKE, Cloud Run, etc. with appropriate IAM roles
- **gcloud CLI**: Run `gcloud auth application-default login` for local development

## Required Environment Variables

| Variable | Description | Required |
|---|---|---|
| `GOOGLE_APPLICATION_CREDENTIALS` | Path to service account JSON key | Yes (unless using ADC) |
| `GOOGLE_CLOUD_PROJECT` | GCP project ID (alternative to `project_id` param) | No |
| `GOOGLE_CLOUD_LOCATION` | Dialogflow CX location (alternative to `location` param) | No |

## Usage

```python
from livekit.agents import Agent, AgentSession, RtcSession
from livekit.plugins import dialogflow, deepgram, google as google_tts


class MyAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="",  # Not used — Dialogflow manages conversation logic
            llm=dialogflow.LLM(
                project_id="my-gcp-project",
                agent_id="my-dialogflow-agent-id",
                location="us-central1",
            ),
            stt=deepgram.STT(),
            tts=google_tts.TTS(),
        )

    async def on_enter(self):
        await self.session.generate_reply(
            instructions="Greet the user",
        )
```

## Constructor Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `project_id` | `str \| None` | `None` | GCP project ID. Falls back to `GOOGLE_CLOUD_PROJECT` env var. |
| `location` | `str \| None` | `None` | Dialogflow CX location. Falls back to `GOOGLE_CLOUD_LOCATION`, then `"global"`. |
| `agent_id` | `str` | — | The Dialogflow CX agent ID (required). |
| `language_code` | `str` | `"en"` | Language for detect intent requests. |
| `environment_id` | `str \| None` | `None` | Optional environment ID for versioned agents. Uses draft if unset. |
| `session_ttl` | `int` | `3600` | How long to keep Dialogflow sessions alive (seconds). |

## Important Notes

- **No streaming**: Dialogflow CX returns the full response in one shot. This means higher perceived latency compared to streaming LLMs. The entire response is emitted as a single chunk.
- **No token counts**: Dialogflow does not provide token usage metrics. Usage fields are set to 0.
- **Session management**: Currently each request creates a new Dialogflow session. For multi-turn conversations, pass a `session_id` via `extra_kwargs` in the `chat()` call.
- **Custom payloads**: Dialogflow custom payload responses (cards, suggestions, etc.) are available in the `extra` field of `ChoiceDelta`.
