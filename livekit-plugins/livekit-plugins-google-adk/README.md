# LiveKit Google ADK Plugin

LiveKit Agents plugin for [Google ADK (Agent Development Kit)](https://github.com/google/adk-python) integration.

## Overview

This plugin enables LiveKit voice agents to leverage Google ADK's powerful orchestration capabilities:

- **Multi-agent coordination** - Complex agent workflows
- **MCP tool integration** - Access to Model Context Protocol tools
- **Session management** - Stateful conversations with context
- **Native telemetry** - Built-in observability and tracing
- **Streaming support** - Real-time response streaming

## Architecture

```
LiveKit Voice Agent
    ├─ STT: Speech-to-Text (Deepgram, Google, etc.)
    ├─ LLM: Google ADK ⭐ (this plugin)
    ├─ TTS: Text-to-Speech (ElevenLabs, Google, etc.)
    └─ VAD: Voice Activity Detection

Google ADK Server
    ├─ Agent orchestration
    ├─ Tool calling & MCP integration
    ├─ Context & memory management
    └─ Telemetry & observability
```

## Installation

### Using uv (Recommended) ⭐

```bash
# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install the plugin
cd services/livekit-plugins-google-adk
uv sync
uv pip install -e ".[dev]"
```

### Using pip

```bash
pip install livekit-plugins-google-adk
```

Or install from source:

```bash
cd services/livekit-plugins-google-adk
pip install -e .
```

## Quick Start

### 1. Start your Google ADK server

```bash
cd services/ai-engine
python -m genkit start --app orchestartor
```

### 2. Create a LiveKit voice agent with ADK

```python
from livekit.agents import JobContext, WorkerOptions, cli
from livekit.agents.voice import Agent, AgentSession
from livekit.plugins import google, silero
from livekit.plugins.google_adk import LLM as GoogleADK

async def entrypoint(ctx: JobContext):
    # Create agent with Google ADK as LLM
    agent = Agent(
        instructions="You are a helpful assistant.",
        stt=google.STT(),
        llm=GoogleADK(
            api_base_url="http://localhost:8000",
            app_name="orchestartor",
            user_id="user_123",
        ),
        tts=google.TTS(),
        vad=silero.VAD.load(),
    )

    # Start the session
    session = AgentSession()
    await session.start(agent=agent, room=ctx.room)

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
```

### 3. Run the agent

```bash
python agent.py dev
```

## Configuration Options

```python
GoogleADK(
    api_base_url="http://localhost:8000",              # ADK server URL
    app_name="my-agent",                               # ADK app name
    user_id="user_123",                                # User identifier
    model="google-adk",                                # Model identifier (for tracking)
    session_id=None,                                   # Optional: explicit session ID
    use_room_name_as_session=False,                    # Use LiveKit room name as session ID
    use_participant_identity_as_session=False,         # Use participant identity as session ID
    auto_create_session=True,                          # Auto-create session if needed
    request_timeout=30.0,                              # Request timeout in seconds
)
```

## Features

### Session Management ⭐

The plugin provides flexible session management with four strategies:

#### 1. Auto-Generated Sessions (Default)

Each conversation gets a unique session:

```python
llm = GoogleADK(
    api_base_url="http://localhost:8000",
    app_name="my-agent",
    user_id="user_123",
    # auto_create_session=True by default
)
# Session ID: session-1234567890 (auto-generated)
```

#### 2. Room-Based Sessions ⭐ NEW

All participants in the same LiveKit room share one ADK session:

```python
llm = GoogleADK(
    api_base_url="http://localhost:8000",
    app_name="my-agent",
    user_id="user_123",
    use_room_name_as_session=True,  # Use LiveKit room name
)
# Session ID: room-my-meeting-room
# Use case: Multi-participant conversations, shared context
```

#### 3. Participant-Based Sessions ⭐ NEW

Each participant gets their own isolated ADK session:

```python
llm = GoogleADK(
    api_base_url="http://localhost:8000",
    app_name="my-agent",
    user_id="user_123",
    use_participant_identity_as_session=True,  # Use participant identity
)
# Session ID: participant-user-abc123
# Use case: Isolated conversations, personal context
```

#### 4. Explicit Session ID

Manually specify the session ID:

```python
llm = GoogleADK(
    api_base_url="http://localhost:8000",
    app_name="my-agent",
    user_id="user_123",
    session_id="my-custom-session-id",  # Explicit ID
)
# Session ID: my-custom-session-id
# Use case: Resume existing session, custom session management
```

#### Manual Context Setting

You can also manually set LiveKit context:

```python
llm = GoogleADK(
    api_base_url="http://localhost:8000",
    app_name="my-agent",
    user_id="user_123",
    use_room_name_as_session=True,
)

# Manually set context (useful for testing)
llm.set_livekit_context(
    room_name="my-room",
    participant_identity="user-123",
)
```

**Which strategy to use?**

| Strategy | Use Case | Session Scope |
|----------|----------|---------------|
| **Auto-generated** | Simple chatbots, stateless conversations | Per conversation |
| **Room-based** | Multi-participant meetings, shared memory | Per LiveKit room |
| **Participant-based** | Personalized agents, isolated state | Per participant |
| **Explicit** | Resume sessions, custom logic | Custom |

### Streaming Responses

The plugin streams responses in real-time from ADK:

```python
# Streaming is automatic with LiveKit voice agents
async for chunk in llm.chat(chat_ctx=context):
    # Each chunk is yielded as it arrives from ADK
    print(chunk.choices[0].delta.content)
```

### Tool Calling (Coming Soon)

```python
from livekit.agents import llm

# Define tools
tools = [
    llm.FunctionInfo(
        name="get_weather",
        description="Get weather for a location",
        parameters={...},
    )
]

# Tools are automatically forwarded to ADK
agent = Agent(
    llm=GoogleADK(...),
    tools=tools,
)
```

## Examples

See the [examples](./examples) directory for complete examples:

- **[voice_agent.py](./examples/voice_agent.py)** - Basic voice agent with ADK
- **[multi_agent.py](./examples/multi_agent.py)** - Multi-agent orchestration (coming soon)
- **[tool_calling.py](./examples/tool_calling.py)** - Agent with custom tools (coming soon)

## Development

### Setup with uv (Recommended)

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest

# Run integration tests
uv run python test_integration.py

# Build package
uv build
```

### Using pip/traditional tools

```bash
# Install dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code (if installed)
black .

# Lint (if installed)
ruff check .
```

### Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=livekit.plugins.google_adk

# Run specific test
pytest tests/test_llm.py::test_chat_streaming
```

## Architecture Details

### ADK Integration

This plugin integrates with Google ADK via Server-Sent Events (SSE):

1. **Session Creation**: Creates ADK session on first message
2. **Message Streaming**: Sends user message via `/run_sse` endpoint
3. **Chunk Processing**: Parses SSE events and yields chat chunks
4. **Error Handling**: Properly handles connection errors and timeouts

### LiveKit Integration

The plugin implements LiveKit's `LLM` base class:

- `model` property - Returns model identifier
- `provider` property - Returns "google-adk"
- `chat()` method - Streams chat completions
- `aclose()` method - Cleanup resources

## Comparison with Direct Integration

### Before (Direct ADK Integration)

```python
class ADKVoiceAgent(Agent):
    async def llm_node(self, chat_ctx, tools, model_settings=None):
        # Custom SSE parsing and session management
        # 80+ lines of boilerplate code
        ...
```

### After (With Plugin)

```python
agent = Agent(
    llm=GoogleADK(
        api_base_url="http://localhost:8000",
        app_name="orchestartor",
        user_id="user_123",
    ),
    # That's it! 4 lines instead of 80+
)
```

## Contributing

This plugin is designed to be contributed back to the LiveKit community. To contribute:

1. **Fork** the [LiveKit Agents repo](https://github.com/livekit/agents)
2. **Create** `livekit-plugins/livekit-plugins-google-adk/` directory
3. **Copy** this plugin code
4. **Add tests** following LiveKit's testing conventions
5. **Submit PR** with:
   - Plugin code
   - Tests
   - Documentation
   - Example usage

See [CONTRIBUTING.md](https://github.com/livekit/agents/blob/main/CONTRIBUTING.md) for details.

## License

Apache 2.0

## Resources

- [Google ADK Documentation](https://github.com/google/adk-python)
- [LiveKit Agents Framework](https://docs.livekit.io/agents/)
- [LiveKit Python SDK](https://github.com/livekit/python-sdks)

## Support

For issues and questions:
- Google ADK issues: https://github.com/google/adk-python/issues
- LiveKit issues: https://github.com/livekit/agents/issues
