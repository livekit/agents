# OpenAI plugin for LiveKit Agents

Support for OpenAI Realtime API, Responses API, LLM, TTS, and STT APIs.

Also includes support for a large number of OpenAI-compatible APIs including [Azure OpenAI](https://docs.livekit.io/agents/integrations/llm/azure-openai/), [Cerebras](https://docs.livekit.io/agents/integrations/cerebras/), [Fireworks](https://docs.livekit.io/agents/integrations/llm/fireworks/), [Perplexity](https://docs.livekit.io/agents/integrations/llm/perplexity/), [Telnyx](https://docs.livekit.io/agents/integrations/llm/telnyx/), [xAI](https://docs.livekit.io/agents/integrations/llm/xai/), [Ollama](https://docs.livekit.io/agents/integrations/llm/ollama/), [DeepSeek](https://docs.livekit.io/agents/integrations/llm/deepseek/), and [SambaNova](https://docs.livekit.io/agents/integrations/llm/sambanova/).

See [https://docs.livekit.io/agents/integrations/openai/](https://docs.livekit.io/agents/integrations/openai/) for more information.

## Installation

```bash
pip install livekit-plugins-openai
```

## Pre-requisites

You'll need an API key from OpenAI. It can be set as an environment variable: `OPENAI_API_KEY`

## GPT-Live tracing

GPT-Live adds two complementary trace layers to the normal agent/tool/playback spans:

- `llm_request` spans cover managed backend Responses calls, including response and
  delegation IDs, duration, usage, completed public text/refusals and accepted tool calls.
- `gpt_live.protocol` spans record delegation creation, response lifecycle metadata,
  submitted tool results, continuation requests, context updates and acknowledgments,
  input/output transcript timing, usage/context-window metadata, and errors. Correlate
  them using session, delegation, response, tool-call and client-event IDs.

Protocol `direction` distinguishes `queued`, `sent`, `received`, and `send_failed`.
`sent` means the local WebSocket send completed, not that OpenAI accepted the command.
Tool-result submission has no separate success acknowledgment. Context `*.appended`
receipts indicate estimated context injection, not that the caller heard speech.
`connection.closed` with `adapter_cleanup` marks local connection cleanup rather than
an invented provider close receipt.

`gpt_live.audio_activity` summarizes continuous input/output transport in one-second
windows with frame counts and first/last arrival or send times. Final partial windows
flush on connection cleanup. Audio may contain silence: compare output transcripts
with the existing `agent_speaking` spans for local playback evidence. Neither proves
what reached the caller's phone. Raw audio is never included in traces.

Content follows `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` and LiveKit's
PII filtering. Content attributes use the existing GenAI names or `lk.pii.*`; ordinary
metadata excludes text and audio. Token deltas and private reasoning are not captured.
The managed backend's exact input and the voice model's internal consumption of
results are unavailable and are not reconstructed from the speaker transcript.
These traces improve diagnosis; they do not introduce retries or silence recovery.
