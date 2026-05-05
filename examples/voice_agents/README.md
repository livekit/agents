# Voice Agents Examples

This directory contains examples demonstrating various capabilities and integrations with the LiveKit Agents framework.

## Model Configuration

Most examples use **LiveKit Inference** by default, which provides a unified API for accessing STT, LLM, and TTS models:

```python
from livekit.agents import inference

session = AgentSession(
    stt=inference.STT("deepgram/nova-3"),
    llm=inference.LLM("openai/gpt-4.1-mini"),
    tts=inference.TTS("cartesia/sonic-3"),
)
```

**Note:** Real-time voice-to-voice models (Amazon Nova Sonic, xAI Grok, etc.) are not supported by LiveKit Inference and must use the provider plugin directly.

## Table of Contents

### Getting Started

- [`basic_agent.py`](./basic_agent.py) - A fundamental voice agent with multilingual STT, turn detection, preemptive generation, and metrics collection

### Tool Integration & Function Calling

- [`async_tool_agent.py`](./async_tool_agent.py) - Long-running async tools with mid-flight progress updates using `AsyncToolset` (travel assistant demo)
- [`tool_search_agent.py`](./tool_search_agent.py) - Dynamic tool discovery using `ToolSearchToolset` / `ToolProxyToolset` to keep large tool sets out of the LLM context

### Pipeline Nodes & Hooks

- [`fast-preresponse.py`](./fast-preresponse.py) - Generating quick acknowledgements with a small LLM using the `on_user_turn_completed` hook while the main LLM runs
- [`timed_agent_transcript.py`](./timed_agent_transcript.py) - Reading timestamped transcripts from `transcription_node`

### Agent Configuration

- [`instructions_per_modality.py`](./instructions_per_modality.py) - Providing different instructions for voice vs. text modalities using `Instructions`

### Real-time Models

> **Note:** Real-time models use provider plugins directly. These examples require provider-specific API keys.

- [`realtime_joke_teller.py`](./realtime_joke_teller.py) - Amazon Nova Sonic real-time model with function calls
- [`grok/`](./grok/) - xAI Grok Voice Agents API with built-in X.com and web search

### Multi-agent

- [`restaurant_agent.py`](./restaurant_agent.py) - Multi-agent system for restaurant ordering and reservation management

### MCP & External Integrations

- [`mcp/`](./mcp/) - Model Context Protocol (MCP) integration examples
  - [`mcp-agent.py`](./mcp/mcp-agent.py) - Connecting an agent to an MCP server
  - [`server.py`](./mcp/server.py) - MCP server example

### RAG & Knowledge Management

- [`llamaindex-rag/`](./llamaindex-rag/) - RAG implementation with LlamaIndex
  - [`chat_engine.py`](./llamaindex-rag/chat_engine.py) - Chat engine integration
  - [`query_engine.py`](./llamaindex-rag/query_engine.py) - Query engine used as a function tool
  - [`retrieval.py`](./llamaindex-rag/retrieval.py) - Document retrieval

### Tracing & Error Handling

- [`langfuse_trace.py`](./langfuse_trace.py) - LangFuse integration for conversation tracing
- [`error_callback.py`](./error_callback.py) - Handling session errors and close events

### Specialized Use Cases

- [`push_to_talk.py`](./push_to_talk.py) - Push-to-talk for multi-participant conversations via RPC

## Additional Resources

- [LiveKit Agents Documentation](https://docs.livekit.io/agents/)
- [Agents Starter Example](https://github.com/livekit-examples/agent-starter-python)
- [More Agents Examples](https://github.com/livekit-examples/python-agents-examples)
