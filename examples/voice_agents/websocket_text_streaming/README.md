# WebSocket LLM Integration PoC

This example demonstrates how to connect LiveKit Agents to a custom WebSocket-based LLM backend instead of standard providers like OpenAI.

## Components

| File | Description |
|------|-------------|
| `ws_llm_server.py` | Standalone WebSocket server with healthcare LLM persona |
| `ws_llm_provider.py` | Custom LLM provider for LiveKit that connects via WebSocket |
| `agent_example.py` | Example LiveKit agent using the WebSocket LLM provider |
| `chat_client.html` | Web-based chat UI for testing the WebSocket server |

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    ws_llm_server.py                          │
│              (Healthcare LLM WebSocket Server)               │
│                                                              │
│   WebSocket ──► OpenAI API ──► Streaming Response            │
└──────────────────────────▲───────────────────────────────────┘
                           │ ws://localhost:8765
                           │
┌──────────────────────────┴───────────────────────────────────┐
│                    ws_llm_provider.py                        │
│                                                              │
│   WebSocketLLM ──► WebSocketLLMStream                        │
│   (extends llm.LLM)   (extends llm.LLMStream)                │
└──────────────────────────▲───────────────────────────────────┘
                           │
┌──────────────────────────┴───────────────────────────────────┐
│                    agent_example.py                          │
│                                                              │
│   STT (Deepgram) ──► WebSocketLLM ──► TTS (Cartesia)         │
└──────────────────────────────────────────────────────────────┘
```

## Prerequisites

1. Set environment variables (create `.env` file):
   ```
   OPENAI_API_KEY=your-openai-api-key
   LIVEKIT_URL=your-livekit-url
   LIVEKIT_API_KEY=your-api-key
   LIVEKIT_API_SECRET=your-api-secret
   DEEPGRAM_API_KEY=your-deepgram-key
   CARTESIA_API_KEY=your-cartesia-key
   ```

## Running the Example

### Step 1: Start the WebSocket LLM Server

In one terminal:

```bash
cd examples/voice_agents/websocket_text_streaming
uv run ws_llm_server.py
```

You should see:
```
INFO:ws-llm-server:Starting Healthcare LLM Server on ws://localhost:8765
INFO:ws-llm-server:Server started. Press Ctrl+C to stop.
```

### Step 2: Start the LiveKit Agent

In another terminal:

```bash
cd examples/voice_agents/websocket_text_streaming
uv run agent_example.py dev
```

### Step 3: Test with Web Chat Client (Optional)

Open `chat_client.html` in your browser to test the WebSocket server directly:

```bash
# macOS
open chat_client.html

# Linux
xdg-open chat_client.html

# Windows
start chat_client.html
```

The chat client provides a ChatGPT-like UI with:
- Real-time streaming responses
- Connection status indicator
- Suggestion buttons for quick prompts

### Step 4: Connect to LiveKit Room

Use the LiveKit playground or your own client to connect to a room. The agent will:
1. Greet you as a healthcare assistant
2. Listen to your voice input
3. Send text to the WebSocket LLM server
4. Speak the streaming response back to you

## WebSocket Protocol

### Client → Server

```json
{
    "type": "chat",
    "messages": [
        {"role": "user", "content": "What are symptoms of a cold?"}
    ]
}
```

### Server → Client

**Streaming delta** (sent multiple times):
```json
{
    "type": "delta",
    "content": "Common cold"
}
```

**Completion** (sent once at the end):
```json
{
    "type": "complete",
    "content": "Common cold symptoms include runny nose, sore throat..."
}
```

**Error** (if something goes wrong):
```json
{
    "type": "error",
    "message": "Error description"
}
```

## Customization

### Using a Different LLM Backend

The `ws_llm_server.py` uses OpenAI, but you can modify it to use any LLM:

1. Replace the `openai` calls in `_generate_streaming_response()`
2. Keep the same WebSocket protocol (delta/complete messages)

### Changing the System Prompt

Edit `HEALTHCARE_SYSTEM_PROMPT` in `ws_llm_server.py` to change the assistant's personality.

### Adding Tool/Function Support

This PoC doesn't support function calling. To add it:

1. Extend the WebSocket protocol to include tool calls
2. Update `WebSocketLLMStream._run()` to handle function call responses
3. Emit `FunctionToolCall` in `ChatChunk.delta.tool_calls`

## Limitations

- **No tool calling**: This PoC focuses on text streaming only
- **Single connection per request**: A production version would use connection pooling
- **No authentication**: The WebSocket server has no auth mechanism
- **Local only**: Server binds to localhost by default

