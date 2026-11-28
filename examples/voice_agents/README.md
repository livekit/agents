# Voice Agents Examples

This directory contains examples demonstrating various capabilities and integrations with the LiveKit Agents framework.

## Model Configuration

Most examples use **LiveKit Inference** by default, which provides a unified API for accessing STT, LLM, and TTS models:

```python
from livekit.agents import inference

session = AgentSession(
    stt=inference.STT("assemblyai/assemblyai-stt-3-7-realtime-agent"),
    llm=inference.LLM("google/gemma-4-31b-it"),  # low-latency gemma, hosted on LiveKit
    tts=inference.TTS("fishaudio/s2.1-pro", voice="fa4c9eb3dccc4806b382b40d61c6b10a"),
)
```

**Note:** For a speech-to-speech model, use OpenAI GPT-Live through LiveKit Inference with `llm="openai/gpt-live-1"` (or `inference.DuplexModel(model="openai/gpt-live-1")`). xAI Grok Voice is available through `inference.RealtimeModel`; use provider plugin classes for direct credentials or for other realtime models such as Amazon Nova Sonic.

## Table of Contents

### Getting Started

- [`basic_agent.py`](./basic_agent.py) - The agent from the official [agent-starter-python](https://github.com/livekit-examples/agent-starter-python) template: an STT-LLM-TTS pipeline on LiveKit Inference with the LiveKit turn detector, keyterms, expressive TTS with fallback, noise cancellation, and an end call tool. Shows how to switch to OpenAI GPT-Live. For a full project with tests, simulation scenarios, and a Dockerfile, run `lk agent init my-agent --template agent-starter-python`

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
