# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an Ultravox-LiveKit integration repository containing:

- **Main agent**: `/agent.py` - The primary Ultravox agent implementation with function tools
- **LiveKit Agents Framework**: `/livekit-agents/` - Complete LiveKit agents framework with core library and plugins
- **Ultravox Plugin**: `/livekit-plugins/livekit-plugins-ultravox/` - Custom Ultravox plugin for LiveKit

The repository uses the LiveKit Agents framework to build realtime, programmable voice agents that can see, hear, and understand using Ultravox models.

## Development Commands

This project uses `uv` as the Python package manager and build tool.

### Core Development Commands

```bash
# Install dependencies (run from root directory)
uv sync --all-extras --dev

# Run the main agent in different modes
python agent.py console    # Terminal mode for local testing
python agent.py dev       # Development mode with hot reload
python agent.py start     # Production mode

# Linting and formatting
uv run ruff check    # Lint code
uv run ruff format   # Format code

# Type checking
uv run mypy .

# Run tests
cd tests && make test      # Run integration tests with Docker
uv run pytest             # Run unit tests directly
```

### Environment Setup

Required environment variables for agent functionality:
- `LIVEKIT_URL` - LiveKit server URL
- `LIVEKIT_API_KEY` - LiveKit API key  
- `LIVEKIT_API_SECRET` - LiveKit API secret
- `ULTRAVOX_API_KEY` - Ultravox API key for realtime model
- Create `.env` file in project root with these variables

## Architecture

### Core Components

1. **Agent Framework** (`/livekit-agents/`)
   - `livekit.agents.voice.Agent` - Main agent class with instructions and tools
   - `livekit.agents.voice.AgentSession` - Manages agent-user interactions
   - `livekit.agents.JobContext` - Entry point for interactive sessions
   - `livekit.agents.Worker` - Main process for job scheduling

2. **Ultravox Integration** (`/livekit-plugins/livekit-plugins-ultravox/`)
   - `RealtimeModel` - Ultravox realtime model integration located at `livekit/plugins/ultravox/realtime/realtime_model.py`
   - Custom events and model handling for Ultravox-specific functionality
   - Direct client tools execution pattern with immediate response

3. **Plugin Ecosystem** (`/livekit-plugins/`)
   - 30+ plugins for STT, LLM, TTS providers (OpenAI, Google, Anthropic, etc.)
   - Each plugin follows consistent patterns for integration

### Agent Implementation Pattern

Agents follow this standard pattern:
```python
from livekit.agents import Agent, AgentSession, JobContext, cli, function_tool
from livekit.plugins.ultravox.realtime import RealtimeModel

@function_tool
async def my_tool(param: str) -> str:
    """Tool description for LLM"""
    return "result"

async def entrypoint(ctx: JobContext) -> None:
    await ctx.connect()
    
    session = AgentSession(
        vad=silero.VAD.load(),
        llm=RealtimeModel(model_id="fixie-ai/ultravox"),
    )
    
    await session.start(
        agent=Agent(
            instructions="Agent instructions",
            tools=[my_tool],
        ),
        room=ctx.room,
    )

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
```

## Development Workflow

### Workspace Structure
The project uses a uv workspace configuration defined in the root `pyproject.toml`:
- `members = ["livekit-plugins/*", "livekit-agents"]`
- Each plugin in `/livekit-plugins/` has its own `pyproject.toml`
- Workspace dependencies are managed through `[tool.uv.sources]`

### Plugin Development
Each plugin in `/livekit-plugins/` follows the workspace pattern and can be developed independently while sharing common dependencies.

### Testing Strategy
- Integration tests use Docker Compose with toxiproxy for network simulation (`tests/Makefile`)
- Unit tests with pytest and async support
- Tests located in `/tests/` directory
- Use `make up`, `make test`, `make down` for integration testing

### Code Quality
- Ruff for linting and formatting (config in root `pyproject.toml`)
- MyPy for strict type checking with `strict = true`
- Line length: 100 characters
- Python 3.9+ target
- Select rules: E, W, F, I, B, C4, UP (with E501 ignored)

## Ultravox Specifics

The main agent (`/agent.py`) demonstrates:
- Function tool implementation with proper typing and error handling
- Event handling for conversation tracking and metrics
- Integration with Ultravox realtime models
- VAD (Voice Activity Detection) with Silero

**Important**: Ultravox is a realtime speech-to-speech model with built-in turn detection. Do NOT use the LiveKit turn detector model (`MultilingualModel`) with Ultravox as this creates conflicts between dual turn detection systems and causes function execution issues.

**Function Call Handling**: The Ultravox plugin implements true client tools pattern:
- `auto_tool_reply_generation=False` means we handle complete tool lifecycle
- Tools are sent to Ultravox in `selectedTools` so it can invoke them when needed  
- When Ultravox invokes a tool, we execute it directly and send results immediately
- We also forward to framework for compatibility with LiveKit agent system
- This follows Ultravox client tools pattern where client executes and responds immediately

The Ultravox plugin provides `RealtimeModel` class at `livekit/plugins/ultravox/realtime/realtime_model.py` that integrates with LiveKit's agent framework for real-time voice interactions.

### Example Tool Types
The main agent includes examples of:
- Basic function tools with type annotations (`get_time`, `say_hello`)
- Raw schema function tools (`get_time_raw`)
- Complex parameter handling with enums (`order_pizza` with `PizzaSize`)
- Flexible parameter parsing (list vs comma-separated strings)
- Error handling patterns that return user-friendly messages

## Key Files and Locations

- Main agent implementation: `agent.py`
- Ultravox realtime model: `livekit-plugins/livekit-plugins-ultravox/livekit/plugins/ultravox/realtime/realtime_model.py`
- Agent session management: `livekit-agents/livekit/agents/voice/agent_session.py`
- Core agent framework: `livekit-agents/livekit/agents/`
- Integration tests: `tests/Makefile`
- Project configuration: `pyproject.toml` (root and per-plugin)