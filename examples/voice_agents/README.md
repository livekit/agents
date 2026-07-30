# Voice Agents Examples

This directory contains examples demonstrating various capabilities and integrations with the LiveKit Agents framework.

## Model Configuration

Most examples use **LiveKit Inference** by default, which provides a unified API for accessing STT, LLM, and TTS models:

```python
from livekit.agents import inference

session = AgentSession(
    stt=inference.STT("deepgram/nova-3"),
    llm=inference.LLM("google/gemma-4-31b-it"),  # low-latency gemma, hosted on LiveKit
    tts=inference.TTS("cartesia/sonic-3"),
)
```

**Note:** Real-time voice-to-voice models (Amazon Nova Sonic, xAI Grok, etc.) are not supported by LiveKit Inference and must use the provider plugin directly.

## Table of Contents

### Getting Started

- [`basic_agent.py`](./basic_agent.py) - A fundamental voice agent with multilingual STT, turn detection, preemptive generation, and metrics collection

### Real-time Models

> **Note:** Real-time models use provider plugins directly. These examples require provider-specific API keys.

- [`grok/`](./grok/) - xAI Grok Voice Agents API with built-in X.com and web search

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

- [`otel_trace.py`](./otel_trace.py) - OpenTelemetry (OTLP) integration for conversation tracing

## Additional Resources

- [LiveKit Agents Documentation](https://docs.livekit.io/agents/)
- [Agents Starter Example](https://github.com/livekit-examples/agent-starter-python)
- [More Agents Examples](https://github.com/livekit-examples/python-agents-examples)
