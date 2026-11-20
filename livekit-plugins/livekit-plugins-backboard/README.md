# LiveKit Plugins Backboard

Agent Framework plugin for [Backboard.io](https://backboard.io) — persistent memory, RAG document retrieval, and 1,800+ LLM backends for LiveKit Agents.

## What is Backboard?

Backboard is an AI memory and conversation platform that adds persistent memory, document RAG, and stateful threads on top of any LLM. Unlike direct LLM integrations, Backboard maintains context across sessions — your agent remembers users, retrieves relevant documents, and builds on past conversations.

## Installation

```bash
pip install livekit-plugins-backboard
```

## Prerequisites

- A [Backboard.io](https://backboard.io) account and API key
- A Backboard assistant (created via SDK or dashboard)

Set your API key:

```bash
export BACKBOARD_API_KEY=your_api_key
```

## Usage

### Basic Voice Agent

```python
from livekit.agents import AgentSession, Agent
from livekit.plugins import backboard, deepgram, elevenlabs, silero

class MyAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="You are a helpful voice assistant.",
        )

    async def on_enter(self):
        self.session.generate_reply()

session = AgentSession(
    llm=backboard.LLM(
        assistant_id="your-assistant-id",
        llm_provider="openai",
        model_name="gpt-4o",
    ),
    stt=deepgram.STT(),
    tts=elevenlabs.TTS(),
    vad=silero.VAD.load(),
)
```

### With Persistent Memory

```python
# Memory is enabled by default (memory="auto")
# Backboard automatically extracts and retrieves relevant context

llm = backboard.LLM(
    assistant_id="your-assistant-id",
    llm_provider="anthropic",
    model_name="claude-sonnet-4-5-20250929",
    memory="auto",  # "auto" (read+write), "readonly", or None
)
```

### Per-User Assistants

```python
# Set user identity for isolated memory and threads
llm = backboard.LLM(
    assistant_id="user-specific-assistant-id",
    user_id=participant.identity,
    llm_provider="xai",
    model_name="grok-4-1-fast-non-reasoning",
)
```

### Switch Models Mid-Conversation

Backboard threads are model-agnostic — switch providers without losing context:

```python
# Start with GPT-4o
llm = backboard.LLM(
    assistant_id="your-assistant-id",
    llm_provider="openai",
    model_name="gpt-4o",
)

# Later, switch to Claude (same thread, same memory)
llm._llm_provider = "anthropic"
llm._model_name = "claude-sonnet-4-5-20250929"
```

### Custom Thread Management

```python
from livekit.plugins.backboard import SessionStore

# Pre-set a thread ID (e.g. from your database)
store = SessionStore(
    api_key="your-key",
    base_url="https://app.backboard.io/api",
    assistant_id="your-assistant-id",
)
store.set_thread("user-123", "existing-thread-id")

llm = backboard.LLM(
    assistant_id="your-assistant-id",
    session_store=store,
)
```

## Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key` | str | `BACKBOARD_API_KEY` env var | Backboard API key |
| `base_url` | str | `https://app.backboard.io/api` | API base URL |
| `assistant_id` | str | Required | Backboard assistant ID |
| `user_id` | str | `"default"` | User identity for thread isolation |
| `llm_provider` | str | `"openai"` | LLM provider (`openai`, `anthropic`, `xai`, `google`, etc.) |
| `model_name` | str | `"gpt-4o"` | Model name |
| `memory` | str/None | `"auto"` | Memory mode: `"auto"`, `"readonly"`, or `None` |
| `session_store` | SessionStore | Auto-created | Custom thread management |

## How It Works

1. **User speaks** → STT converts to text
2. **Plugin extracts** the latest user message from LiveKit's `ChatContext`
3. **SessionStore** resolves a Backboard thread ID for this user (creates one if needed)
4. **Backboard API** receives the message, retrieves relevant memories and documents, routes to the configured LLM, and streams the response
5. **Plugin emits** `ChatChunk` objects into the LiveKit pipeline
6. **TTS converts** the response to speech

Memory and document context are handled entirely by Backboard — no additional configuration needed in the agent.

## Supported LLM Providers

Backboard supports 1,800+ models. Common providers:

| Provider | Example Models |
|----------|---------------|
| OpenAI | `gpt-4o`, `gpt-4o-mini`, `o1` |
| Anthropic | `claude-sonnet-4-5-20250929`, `claude-3-haiku` |
| xAI | `grok-4-1-fast-non-reasoning` |
| Google | `gemini-2.5-pro`, `gemini-2.5-flash` |
| Mistral | `mistral-large` |
| Meta | `llama-3.1-405b` |

## License

Apache 2.0
