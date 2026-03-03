# LiveKit Plugins Lex

Agent Framework plugin for [Amazon Lex V2](https://docs.aws.amazon.com/lex/). This plugin integrates Amazon Lex V2 as an LLM provider in a LiveKit Agents STT → LLM → TTS pipeline.

## What is Amazon Lex V2?

Amazon Lex V2 is an intent-based conversational AI engine with its own session management, intents, slots, and fulfillment logic. Unlike standard LLMs, all conversation design happens in the [AWS Lex V2 Console](https://console.aws.amazon.com/lexv2/) — the `instructions` field on the LiveKit `Agent` class has no effect.

## Installation

```bash
pip install livekit-plugins-lex
```

## Authentication

This plugin uses standard AWS credential resolution via [aiobotocore/botocore](https://docs.aws.amazon.com/sdkref/latest/guide/setting-global-aws_access_key_id.html). Set up authentication using one of:

- **Environment variables**: Set `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`
- **IAM instance profile**: Running on EC2, ECS, Lambda, etc. with appropriate IAM roles
- **Credentials file**: Configure `~/.aws/credentials` via `aws configure`
- **SSO/Assume role**: Any method supported by the standard boto credential chain

## Required Environment Variables

| Variable | Description | Required |
|---|---|---|
| `AWS_ACCESS_KEY_ID` | AWS access key ID | Yes (unless using IAM role) |
| `AWS_SECRET_ACCESS_KEY` | AWS secret access key | Yes (unless using IAM role) |
| `AWS_REGION` | AWS region (alternative to `region` param) | No |
| `AWS_DEFAULT_REGION` | Fallback AWS region | No |

## Usage

```python
from livekit.agents import Agent, AgentSession, RtcSession
from livekit.plugins import lex, deepgram
from livekit.plugins import google as google_tts


class MyAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="",  # Not used — Lex manages conversation logic
            llm=lex.LLM(
                bot_id="my-lex-bot-id",
                bot_alias_id="my-bot-alias-id",
                locale_id="en_US",
                region="us-east-1",
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
| `bot_id` | `str` | — | The Lex bot ID (required). |
| `bot_alias_id` | `str` | — | The Lex bot alias ID (required). |
| `locale_id` | `str` | `"en_US"` | Locale for the bot (e.g. `"en_US"`, `"es_US"`). |
| `region` | `str \| None` | `None` | AWS region. Falls back to `AWS_REGION` then `AWS_DEFAULT_REGION`. |
| `session_ttl` | `int` | `3600` | How long to keep Lex sessions alive (seconds). |

## Important Notes

- **No streaming**: Amazon Lex V2 returns the full response in one shot. This means higher perceived latency compared to streaming LLMs. The entire response is emitted as a single chunk.
- **No token counts**: Lex does not provide token usage metrics. Usage fields are set to 0.
- **`instructions` field not used**: All conversation logic (intents, slots, fulfillment) is configured in the AWS Lex V2 console. The `instructions` field on the `Agent` class has no effect.
- **Session management**: Currently each request creates a new Lex session. For multi-turn conversations, pass a `session_id` via `extra_kwargs` in the `chat()` call.
- **Lex metadata**: Intent name, session state, interpretation confidence, and slots are available in the `extra` field of `ChoiceDelta`.
