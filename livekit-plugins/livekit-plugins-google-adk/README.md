# LiveKit Google ADK Plugin

LiveKit Agents plugin for [Google ADK (Agent Development Kit)](https://github.com/google/adk-python) integration.

## Overview

This plugin enables LiveKit voice agents to leverage Google ADK's powerful orchestration capabilities:

- **Text-based LLM** - Text-only streaming responses (no audio I/O)
- **Session management** - Stateful conversations with context
- **Streaming support** - Real-time response streaming via SSE
- **Multi-agent coordination** - Complex agent workflows (ADK feature)
- **MCP tool integration** - Access to Model Context Protocol tools (ADK feature)

### ⚠️ Current Limitations

This is a **text-only LLM plugin**. The following features are currently **not supported**:

- ❌ **Audio input/output** - No direct audio streaming (use with LiveKit STT/TTS)
- ❌ **Gemini Live** - No realtime voice integration
- ❌ **Advanced authentication** - Basic HTTP connection only

### ✅ Tool Calling Support

✅ **Tool calls ARE fully supported** - Tools are registered directly in your ADK application configuration, not in LiveKit. ADK handles all tool registration, orchestration, and execution through its own configuration (including MCP tool integration).

⚠️ **Important**: Do NOT pass `tools` parameter to the LiveKit `Agent()`. The plugin will raise a `ValueError` if you try. Tools must be configured in ADK, not LiveKit.

### 📝 Important: Instructions Parameter

⚠️ **The `instructions` parameter in `Agent()` is NOT passed to ADK**

When you create a LiveKit `Agent` with Google ADK as the LLM, the `instructions` parameter (system prompt) is **not sent to ADK**. This is because:

- **ADK manages its own prompts**: Google ADK agents handle system prompts, orchestration logic, and multi-agent coordination internally through their own configuration
- **LiveKit tracking only**: The `instructions` parameter is only used by LiveKit for logging and tracking purposes
- **Best practice**: Set `instructions=""` when using ADK to make this explicit

```python
# Correct usage with ADK
agent = Agent(
    instructions="",  # Empty - ADK handles prompts internally
    llm=google_adk.LLM(...),
)
```

If you need to configure agent behavior, do so in your **ADK application configuration**, not in the LiveKit Agent instructions.

## Architecture

```text
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

## Prerequisites

### ADK Server Requirements

⚠️ **You must have a running ADK server** at the `api_base_url` you configure. This plugin connects to an external ADK server - it does not run ADK internally.

#### Option 1: Official Google ADK Server (Recommended)
```bash
# Install ADK
pip install google-adk

# Start the ADK API server
adk api_server
# Server runs at http://localhost:8000 by default
```

#### Option 2: Custom ADK-Compatible Server

If you're running a custom server (not the official ADK), it **must implement the standard ADK SSE API**:
- `POST /apps/{app_name}/users/{user_id}/sessions/{session_id}` - Create/verify session
- `POST /run_sse` - Stream chat completions via Server-Sent Events

**Required SSE event format**:
```json
data: {"content": {"parts": [{"text": "..."}]}, "partial": true}
data: {"content": {"parts": [{"text": "..."}]}, "partial": false}
```

The final event must have `"partial": false` to signal completion.

**HTTP Headers**: The plugin sends standard SSE headers. If your custom server requires authentication or specific headers, you'll need to modify the plugin or use a proxy:
- `Content-Type: application/json`
- `Accept: text/event-stream` (implicit in SSE response handling)
- Authentication headers are not currently supported (see Limitations)

## Quick Start

**Architecture Overview:**
- **LiveKit** handles voice session management, STT (speech-to-text), and TTS (text-to-speech)
- **ADK** handles LLM responses, orchestration, tool execution, and multi-agent coordination
- **This plugin** bridges the two: converts LiveKit text → ADK → streams back to LiveKit

### 1. Ensure your ADK server is running

```bash
adk api_server
# ADK server should be accessible at http://localhost:8000
```

### 2. Create a LiveKit voice agent with ADK

See the complete working example: [`examples/basic_voice_agent.py`](examples/basic_voice_agent.py)

**Key code snippet:**
```python
from livekit.plugins import google, google_adk, silero

agent = Agent(
    instructions="",  # Not passed to ADK - ADK manages prompts internally
    llm=google_adk.LLM(
        api_base_url="http://localhost:8000",
        app_name="orchestrator",  # Must match your ADK app name
        user_id="user_123",
    ),
    stt=google.STT(),
    tts=google.TTS(),
    vad=silero.VAD.load(),
)
```

### 3. Run the agent

```bash
python examples/basic_voice_agent.py dev
```

## Configuration Options

```python
from livekit.plugins import google_adk

google_adk.LLM(
    api_base_url="http://localhost:8000",              # ADK server URL (required)
    app_name="<your-adk-app-name>",                    # ADK app name - must match your ADK config (required)
    user_id="<user-identifier>",                       # User identifier for ADK session (required)
    model="google-adk",                                # Model identifier for LiveKit tracking (optional)
    session_id=None,                                   # Optional: explicit session ID to reuse
    auto_create_session=True,                          # Auto-create session if not provided (default: True)
    request_timeout=30.0,                              # HTTP request timeout in seconds (default: 30.0)
)
```

### Required Parameters

- **`api_base_url`**: URL of your running ADK server (e.g., `http://localhost:8000`)
- **`app_name`**: Name of your ADK application - this must exactly match the app name in your ADK configuration
- **`user_id`**: Unique identifier for the user - used for session management in ADK

### Optional Parameters

- **`model`**: Model identifier used for LiveKit's internal tracking and logging (default: `"google-adk"`)
- **`session_id`**: Explicit session ID to reuse an existing ADK session. If not provided, a new session will be created
- **`auto_create_session`**: Whether to automatically create a new session if `session_id` is not provided (default: `True`)
- **`request_timeout`**: Timeout in seconds for HTTP requests to the ADK server (default: `30.0`)

### Important Notes

⚠️ **Instructions Parameter**: When creating the LiveKit `Agent`, the `instructions` parameter is **NOT passed to ADK**. ADK agents handle their own system prompts and orchestration logic internally. The `instructions` parameter is only used by LiveKit for logging/tracking purposes. We recommend setting it to an empty string `""` when using ADK.

## Features

### Session Management ⭐

The plugin provides flexible session management with four strategies:

#### 1. Auto-Generated Sessions (Default)

Each conversation gets a unique session automatically:

```python
from livekit.plugins import google_adk

llm = google_adk.LLM(
    api_base_url="http://localhost:8000",
    app_name="<your-adk-app-name>",
    user_id="<user-identifier>",
    # auto_create_session=True by default
)
# Session ID will be auto-generated: session-1234567890
```

**Use case**: Simple chatbots where each conversation is independent and you want ADK to manage session lifecycle.

#### 2. Explicit Session ID

Manually specify a session ID to reuse an existing ADK session:

```python
from livekit.plugins import google_adk

llm = google_adk.LLM(
    api_base_url="http://localhost:8000",
    app_name="<your-adk-app-name>",
    user_id="<user-identifier>",
    session_id="<existing-adk-session-id>",  # Must be a valid ADK session ID
)
# Session ID: <existing-adk-session-id>
```

**Use case**: Resume an existing conversation, maintain context across multiple LiveKit sessions, or implement custom session management logic outside of LiveKit.

**Which strategy to use?**

| Strategy | Use Case | Session Scope |
|----------|----------|---------------|
| **Auto-generated** | Simple chatbots, unique per conversation | Per conversation |
| **Explicit** | Resume sessions, custom logic | Custom |

### Internal: How Streaming Works

This section describes internal plugin behavior - **you don't need to manage streaming manually**.

The plugin automatically:
1. Receives Server-Sent Events (SSE) from ADK with `partial=true` chunks
2. Emits text deltas to LiveKit in real-time
3. Converts text to speech via your configured TTS
4. Waits for final event with `partial=false` before closing the stream

When using the LiveKit `Agent`, all streaming is handled transparently by the plugin.

### Tool Calling

**Tools are registered in ADK, not LiveKit**. ADK manages all tool registration, orchestration, and execution through its own configuration.

```python
from livekit.plugins import google_adk

# No tools parameter needed in LiveKit
# Tools are configured in your ADK application
agent = Agent(
    instructions="",  # Not passed to ADK
    llm=google_adk.LLM(
        api_base_url="http://localhost:8000",
        app_name="<your-adk-app-name>",  # This ADK app has tools configured
        user_id="<user-identifier>",
    ),
    # NO tools parameter - ADK manages tools internally
)
```

**How it works**:
1. **Register tools in ADK**: Configure tools in your ADK application using MCP (Model Context Protocol) servers or custom tool implementations
2. **LiveKit is unaware**: The LiveKit agent does not need to know about available tools
3. **ADK orchestrates**: ADK decides when to call tools, executes them, and handles the results
4. **Responses stream back**: Tool results are incorporated into ADK's responses, which stream to LiveKit

**Tool registration happens in your ADK application code**, not in configuration files. Refer to the [Google ADK documentation](https://github.com/google/adk-python) for:
- Connecting MCP servers to your ADK agent
- Implementing custom tools
- Tool orchestration and multi-agent coordination

This approach gives you the full power of ADK's tool orchestration without coupling to LiveKit's tool system.

## Examples

See the [examples](./examples) directory for complete working examples:

- **[basic_voice_agent.py](./examples/basic_voice_agent.py)** - Basic voice agent with ADK, STT, and TTS

This example demonstrates:
- Setting up the LiveKit agent with Google ADK
- Connecting to your ADK server
- Handling voice sessions with automatic STT/TTS
- Proper configuration for production use

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

# Format and lint code (using ruff from parent repo)
ruff format livekit-plugins/livekit-plugins-google-adk
ruff check livekit-plugins/livekit-plugins-google-adk
```

### Testing

The plugin includes comprehensive unit tests and integration examples.

**Unit Tests** (in `tests/test_llm.py`):
- ✅ LLM initialization with various configurations
- ✅ Session creation and management
- ✅ Chat streaming with SSE responses
- ✅ Error handling for failed requests
- ✅ Resource cleanup (client session closing)

**Integration Example** (in `tests/integration/local_adk_e2e.py`):
- Full end-to-end example with live ADK server
- See `examples/basic_voice_agent.py` for the recommended reference implementation

```bash
# Run all unit tests
pytest

# Run with coverage
pytest --cov=livekit.plugins.google_adk

# Run specific test
pytest tests/test_llm.py::TestGoogleADKLLM::test_chat_streaming

# Run with verbose output
pytest -v

# Run integration test (requires running ADK server)
python tests/integration/local_adk_e2e.py dev
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
from livekit.plugins import google_adk

agent = Agent(
    instructions="",  # Not passed to ADK
    llm=google_adk.LLM(
        api_base_url="http://localhost:8000",
        app_name="<your-adk-app-name>",  # Must match your ADK config
        user_id="<user-identifier>",
    ),
    # That's it! Clean and simple - ADK handles all the complexity
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
- Google ADK issues: [https://github.com/google/adk-python/issues](https://github.com/google/adk-python/issues)
- LiveKit issues: [https://github.com/livekit/agents/issues](https://github.com/livekit/agents/issues)
