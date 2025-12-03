# Streaming Text Output Implementation Guide

This document provides a detailed explanation of the `StreamingTextOutput` solution for real-time text streaming from LiveKit Agents over WebSocket connections.

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Solution Architecture](#solution-architecture)
3. [StreamingTextOutput Class](#streamingtextoutput-class)
4. [Event Types](#event-types)
5. [Integration with AgentSession](#integration-with-agentsession)
6. [WebSocket Server Implementation](#websocket-server-implementation)
7. [Client Implementation](#client-implementation)
8. [Complete Flow Diagram](#complete-flow-diagram)
9. [API Reference](#api-reference)
10. [Best Practices](#best-practices)

---

## Problem Statement

The LiveKit Agents framework processes LLM responses internally with streaming capabilities. However, the public `RunResult` API only exposes **complete events** after they finish processing:

```python
result = await sess.run(user_input="Hello")
# result.events contains only completed ChatMessageEvent, FunctionCallEvent, etc.
# No access to real-time text deltas during generation
```

For WebSocket APIs or real-time UIs, we need access to text deltas **as they are generated**, not after the entire response is complete.

### Framework Internals

Looking at the framework's `generation.py`, text streaming happens internally:

```python
# Inside _llm_inference_task
async for chunk in llm_node:
    if chunk.delta.content:
        data.generated_text += chunk.delta.content
        text_ch.send_nowait(chunk.delta.content)  # Streaming happens here!
```

The text is streamed to a `TextOutput` via `capture_text()`:

```python
# Inside _text_forwarding_task
async for delta in source:
    if text_output is not None:
        await text_output.capture_text(delta)  # Each delta is captured here
```

**Solution**: Create a custom `TextOutput` implementation that exposes these deltas via an async iterator.

---

## Solution Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        AgentSession                                  │
│  ┌─────────┐    ┌─────────┐    ┌──────────────────────────────────┐ │
│  │   LLM   │───▶│  TTS    │───▶│  Output (audio, text, video)     │ │
│  └─────────┘    └─────────┘    │                                  │ │
│       │                        │  sess.output.transcription ──────┼─┼──┐
│       │                        └──────────────────────────────────┘ │  │
│       ▼                                                             │  │
│  Text Deltas                                                        │  │
│  ("Hello", " world", "!")                                          │  │
└─────────────────────────────────────────────────────────────────────┘  │
                                                                         │
┌────────────────────────────────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────────────┐
│                    StreamingTextOutput                               │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  capture_text(delta) ──▶ Queue ──▶ stream() async iterator   │   │
│  │                           │                                   │   │
│  │  flush() ──────────────────┘                                  │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      WebSocket Server                                │
│  async for event in streaming_output:                                │
│      await websocket.send(json.dumps({                              │
│          "type": event.type,                                        │
│          "delta": event.delta  # or event.text                      │
│      }))                                                            │
└─────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         Client                                       │
│  Real-time text display with streaming cursor                       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## StreamingTextOutput Class

### Core Implementation

```python
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Literal

from livekit.agents.voice.io import TextOutput


@dataclass
class TextDeltaEvent:
    """Event emitted for each text chunk from the LLM."""
    delta: str
    """The new text chunk."""
    accumulated: str
    """All text accumulated so far."""
    type: Literal["text_delta"] = "text_delta"


@dataclass
class TextCompleteEvent:
    """Event emitted when the LLM generation is complete."""
    text: str
    """The complete generated text."""
    type: Literal["text_complete"] = "text_complete"


StreamEvent = TextDeltaEvent | TextCompleteEvent


class StreamingTextOutput(TextOutput):
    """
    A TextOutput implementation that streams text deltas in real-time.
    
    This class captures LLM output as it's generated and makes it available
    via an async iterator.
    """

    def __init__(self, *, next_in_chain: TextOutput | None = None) -> None:
        super().__init__(label="streaming_text", next_in_chain=next_in_chain)
        self._queue: asyncio.Queue[StreamEvent | None] = asyncio.Queue()
        self._accumulated_text: str = ""
        self._closed: bool = False

    @property
    def accumulated_text(self) -> str:
        """Returns all text accumulated so far."""
        return self._accumulated_text

    async def capture_text(self, text: str) -> None:
        """
        Called by the agent framework for each text chunk.
        Emits a TextDeltaEvent to the stream.
        """
        self._accumulated_text += text
        
        event = TextDeltaEvent(delta=text, accumulated=self._accumulated_text)
        await self._queue.put(event)

        # Forward to next in chain if exists
        if self.next_in_chain is not None:
            await self.next_in_chain.capture_text(text)

    def flush(self) -> None:
        """
        Called when the LLM generation is complete.
        Emits a TextCompleteEvent to the stream.
        """
        event = TextCompleteEvent(text=self._accumulated_text)
        self._queue.put_nowait(event)
        
        # Signal end of this generation
        self._queue.put_nowait(None)
        
        # Reset for next generation
        self._accumulated_text = ""

        # Forward to next in chain if exists
        if self.next_in_chain is not None:
            self.next_in_chain.flush()

    def close(self) -> None:
        """Close the stream. Call this when done consuming."""
        self._closed = True
        self._queue.put_nowait(None)

    async def stream(self) -> AsyncIterator[StreamEvent]:
        """
        Async iterator that yields streaming events.
        
        Yields TextDeltaEvent for each chunk and TextCompleteEvent when done.
        The iterator completes when flush() is called.
        """
        while not self._closed:
            event = await self._queue.get()
            if event is None:
                break
            yield event

    def __aiter__(self) -> AsyncIterator[StreamEvent]:
        return self.stream()
```

### Key Design Decisions

1. **Queue-based streaming**: Uses `asyncio.Queue` for thread-safe event passing between the framework's internal tasks and the consumer.

2. **Accumulated text tracking**: Each delta event includes both the new chunk and all accumulated text, allowing clients to either build up text incrementally or replace entirely.

3. **Sentinel value for completion**: `None` is pushed to the queue to signal stream completion, allowing the async iterator to terminate cleanly.

4. **Chain support**: The `next_in_chain` parameter allows composing multiple `TextOutput` implementations (e.g., for logging + streaming).

5. **Reset on flush**: The accumulated text resets after each generation, supporting multiple sequential queries.

---

## Event Types

### TextDeltaEvent

Emitted for each text chunk received from the LLM.

| Field | Type | Description |
|-------|------|-------------|
| `type` | `Literal["text_delta"]` | Event type identifier |
| `delta` | `str` | The new text chunk (e.g., `"Hello"`, `" world"`) |
| `accumulated` | `str` | All text accumulated so far |

**Example sequence**:
```
TextDeltaEvent(delta="Hello", accumulated="Hello")
TextDeltaEvent(delta=" world", accumulated="Hello world")
TextDeltaEvent(delta="!", accumulated="Hello world!")
```

### TextCompleteEvent

Emitted when the LLM generation is complete (triggered by `flush()`).

| Field | Type | Description |
|-------|------|-------------|
| `type` | `Literal["text_complete"]` | Event type identifier |
| `text` | `str` | The complete generated text |

**Example**:
```
TextCompleteEvent(text="Hello world!")
```

---

## Integration with AgentSession

### Attaching to Session

The `StreamingTextOutput` is attached to the session's transcription output:

```python
from livekit.agents import AgentSession
from livekit.plugins import openai

async with openai.LLM(model="gpt-4o-mini") as llm:
    async with AgentSession(llm=llm) as sess:
        # Create and attach the streaming output
        streaming_output = StreamingTextOutput()
        sess.output.transcription = streaming_output
        
        # Start the agent
        await sess.start(MyAgent())
        
        # Now streaming_output will receive text deltas
```

### How It Works Internally

1. When `sess.run(user_input="...")` is called, the framework:
   - Sends the input to the LLM
   - Receives streaming chunks from the LLM
   - Calls `text_output.capture_text(chunk)` for each chunk
   - Calls `text_output.flush()` when generation is complete

2. Our `StreamingTextOutput`:
   - Receives `capture_text()` calls and pushes `TextDeltaEvent` to queue
   - Receives `flush()` call and pushes `TextCompleteEvent` + `None` sentinel
   - The consumer iterates via `async for event in streaming_output`

### Consuming the Stream

```python
# Option 1: Consume in the same task (blocks until complete)
result = sess.run(user_input="Hello")
async for event in streaming_output:
    print(event)
await result  # Wait for full completion including tool calls

# Option 2: Consume in a separate task (concurrent processing)
async def consume_stream():
    async for event in streaming_output:
        await process_event(event)

consumer_task = asyncio.create_task(consume_stream())
result = await sess.run(user_input="Hello")
await consumer_task
```

---

## WebSocket Server Implementation

### Server Structure

```python
import asyncio
import json
import logging

from livekit.agents import Agent, AgentSession
from livekit.plugins import openai

from websockets.server import serve

from streaming_text_output import StreamingTextOutput

logger = logging.getLogger(__name__)


class MyAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="You are a helpful assistant.")


async def handle_websocket(websocket):
    """Handle a single WebSocket connection."""
    logger.info("New WebSocket connection")
    
    try:
        async with openai.LLM(model="gpt-4o-mini") as llm:
            async with AgentSession(llm=llm) as sess:
                # Create and attach streaming output
                streaming_output = StreamingTextOutput()
                sess.output.transcription = streaming_output
                
                await sess.start(MyAgent())
                
                # Send ready signal
                await websocket.send(json.dumps({
                    "type": "ready",
                    "message": "Agent ready"
                }))
                
                # Handle incoming messages
                async for message in websocket:
                    data = json.loads(message)
                    user_input = data.get("message", "")
                    
                    if not user_input:
                        continue
                    
                    # Start agent run (non-blocking)
                    result = sess.run(user_input=user_input)
                    
                    # Stream responses to client
                    async for event in streaming_output:
                        if event.type == "text_delta":
                            await websocket.send(json.dumps({
                                "type": "delta",
                                "delta": event.delta,
                                "accumulated": event.accumulated
                            }))
                        elif event.type == "text_complete":
                            await websocket.send(json.dumps({
                                "type": "complete",
                                "text": event.text
                            }))
                    
                    # Wait for full result (tool calls, etc.)
                    await result
                    
                    # Optionally send result events
                    await websocket.send(json.dumps({
                        "type": "result",
                        "events": [format_event(e) for e in result.events]
                    }))
                    
    except Exception as e:
        logger.exception(f"WebSocket error: {e}")


async def main():
    async with serve(handle_websocket, "localhost", 8765):
        await asyncio.Future()  # Run forever


if __name__ == "__main__":
    asyncio.run(main())
```

### Message Protocol

#### Client → Server

```json
{
    "message": "What is the weather in San Francisco?"
}
```

#### Server → Client

**Ready signal**:
```json
{
    "type": "ready",
    "message": "Agent ready"
}
```

**Text delta** (emitted multiple times during streaming):
```json
{
    "type": "delta",
    "delta": "The weather",
    "accumulated": "The weather"
}
```

**Text complete** (emitted once when generation finishes):
```json
{
    "type": "complete",
    "text": "The weather in San Francisco is sunny with a temperature of 68°F."
}
```

**Result** (optional, includes tool calls and messages):
```json
{
    "type": "result",
    "events": [
        {"type": "function_call", "name": "lookup_weather", "arguments": "{\"location\": \"San Francisco\"}"},
        {"type": "function_call_output", "output": "sunny, 68°F"},
        {"type": "message", "role": "assistant", "content": "..."}
    ]
}
```

---

## Client Implementation

### JavaScript WebSocket Client

```javascript
class StreamingAgentClient {
    constructor(url) {
        this.url = url;
        this.ws = null;
        this.onDelta = null;
        this.onComplete = null;
        this.onResult = null;
        this.onError = null;
    }
    
    connect() {
        return new Promise((resolve, reject) => {
            this.ws = new WebSocket(this.url);
            
            this.ws.onopen = () => resolve();
            this.ws.onerror = (e) => reject(e);
            
            this.ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                this.handleMessage(data);
            };
            
            this.ws.onclose = () => {
                console.log('Disconnected');
            };
        });
    }
    
    handleMessage(data) {
        switch (data.type) {
            case 'delta':
                if (this.onDelta) {
                    this.onDelta(data.delta, data.accumulated);
                }
                break;
                
            case 'complete':
                if (this.onComplete) {
                    this.onComplete(data.text);
                }
                break;
                
            case 'result':
                if (this.onResult) {
                    this.onResult(data.events);
                }
                break;
                
            case 'error':
                if (this.onError) {
                    this.onError(data.message);
                }
                break;
        }
    }
    
    send(message) {
        this.ws.send(JSON.stringify({ message }));
    }
    
    close() {
        this.ws.close();
    }
}

// Usage
const client = new StreamingAgentClient('ws://localhost:8765');

client.onDelta = (delta, accumulated) => {
    // Update UI with streaming text
    document.getElementById('response').textContent = accumulated;
};

client.onComplete = (text) => {
    // Handle completion
    console.log('Complete:', text);
};

await client.connect();
client.send('What is the weather in San Francisco?');
```

### React Hook Example

```typescript
import { useState, useEffect, useCallback } from 'react';

interface StreamingState {
    isStreaming: boolean;
    currentText: string;
    isComplete: boolean;
}

function useStreamingAgent(wsUrl: string) {
    const [ws, setWs] = useState<WebSocket | null>(null);
    const [state, setState] = useState<StreamingState>({
        isStreaming: false,
        currentText: '',
        isComplete: false,
    });
    
    useEffect(() => {
        const socket = new WebSocket(wsUrl);
        
        socket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            
            if (data.type === 'delta') {
                setState(prev => ({
                    ...prev,
                    isStreaming: true,
                    currentText: data.accumulated,
                }));
            } else if (data.type === 'complete') {
                setState(prev => ({
                    ...prev,
                    isStreaming: false,
                    isComplete: true,
                    currentText: data.text,
                }));
            }
        };
        
        setWs(socket);
        
        return () => socket.close();
    }, [wsUrl]);
    
    const sendMessage = useCallback((message: string) => {
        setState({ isStreaming: false, currentText: '', isComplete: false });
        ws?.send(JSON.stringify({ message }));
    }, [ws]);
    
    return { ...state, sendMessage };
}

// Usage
function ChatComponent() {
    const { isStreaming, currentText, sendMessage } = useStreamingAgent('ws://localhost:8765');
    
    return (
        <div>
            <p>{currentText}{isStreaming && <span className="cursor">|</span>}</p>
            <button onClick={() => sendMessage('Hello!')}>Send</button>
        </div>
    );
}
```

---

## Complete Flow Diagram

```
User Input                  WebSocket Server                Framework                 LLM
    │                            │                             │                       │
    │──"Hello"──────────────────▶│                             │                       │
    │                            │                             │                       │
    │                            │──sess.run(user_input)──────▶│                       │
    │                            │                             │                       │
    │                            │                             │───prompt─────────────▶│
    │                            │                             │                       │
    │                            │                             │◀──"Hi"────────────────│
    │                            │                             │                       │
    │                            │◀─capture_text("Hi")─────────│                       │
    │                            │                             │                       │
    │◀───{"type":"delta",────────│                             │                       │
    │     "delta":"Hi"}          │                             │                       │
    │                            │                             │◀──" there"────────────│
    │                            │                             │                       │
    │                            │◀─capture_text(" there")─────│                       │
    │                            │                             │                       │
    │◀───{"type":"delta",────────│                             │                       │
    │     "delta":" there"}      │                             │                       │
    │                            │                             │◀──"!"─────────────────│
    │                            │                             │                       │
    │                            │◀─capture_text("!")──────────│                       │
    │                            │                             │                       │
    │◀───{"type":"delta",────────│                             │◀──[done]──────────────│
    │     "delta":"!"}           │                             │                       │
    │                            │                             │                       │
    │                            │◀─flush()────────────────────│                       │
    │                            │                             │                       │
    │◀───{"type":"complete",─────│                             │                       │
    │     "text":"Hi there!"}    │                             │                       │
    │                            │                             │                       │
```

---

## API Reference

### StreamingTextOutput

#### Constructor

```python
StreamingTextOutput(*, next_in_chain: TextOutput | None = None)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `next_in_chain` | `TextOutput \| None` | Optional next output in chain for composition |

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `accumulated_text` | `str` | All text accumulated in current generation |
| `label` | `str` | Label for this output (always `"streaming_text"`) |

#### Methods

| Method | Description |
|--------|-------------|
| `async capture_text(text: str)` | Called by framework for each text chunk |
| `flush()` | Called by framework when generation completes |
| `close()` | Stop the stream (call when done consuming) |
| `async stream() -> AsyncIterator[StreamEvent]` | Returns async iterator of events |
| `__aiter__()` | Allows `async for event in output` syntax |

### TextDeltaEvent

| Field | Type | Description |
|-------|------|-------------|
| `type` | `Literal["text_delta"]` | Event type |
| `delta` | `str` | New text chunk |
| `accumulated` | `str` | All text so far |

### TextCompleteEvent

| Field | Type | Description |
|-------|------|-------------|
| `type` | `Literal["text_complete"]` | Event type |
| `text` | `str` | Complete generated text |

---

## Best Practices

### 1. Always Consume the Stream

The stream must be consumed or the queue will grow unbounded:

```python
# Good: Always iterate the stream
async for event in streaming_output:
    await process(event)

# Bad: Starting a run without consuming
result = await sess.run(...)  # Queue fills up with no consumer!
```

### 2. Handle Multiple Generations

The stream resets after each `flush()`. For multiple queries:

```python
for query in queries:
    result = sess.run(user_input=query)
    async for event in streaming_output:
        # Each query gets its own stream
        await process(event)
    await result
```

### 3. Use Concurrent Tasks for Complex Flows

When you need to process events while also waiting for tools:

```python
async def stream_to_websocket():
    async for event in streaming_output:
        await websocket.send(format_event(event))

stream_task = asyncio.create_task(stream_to_websocket())
result = await sess.run(user_input=query)
await stream_task  # Ensure stream is fully consumed
```

### 4. Chain with Other Outputs

For logging while streaming:

```python
class LoggingTextOutput(TextOutput):
    async def capture_text(self, text: str) -> None:
        logger.debug(f"Text chunk: {text}")
        if self.next_in_chain:
            await self.next_in_chain.capture_text(text)
    
    def flush(self) -> None:
        logger.info("Generation complete")
        if self.next_in_chain:
            self.next_in_chain.flush()

# Chain: Logging → Streaming
streaming = StreamingTextOutput()
logging_output = LoggingTextOutput(next_in_chain=streaming)
sess.output.transcription = logging_output
```

### 5. Error Handling

Always handle disconnections gracefully:

```python
try:
    async for event in streaming_output:
        await websocket.send(format_event(event))
except websockets.ConnectionClosed:
    logger.info("Client disconnected during streaming")
    streaming_output.close()  # Stop the stream cleanly
```

---

## Limitations

1. **Text output only**: This captures the transcription output, not raw LLM chunks. If TTS is enabled, the text may be modified by transcription processing.

2. **Single consumer**: The async iterator is designed for a single consumer. Multiple consumers would receive partial events.

3. **No backpressure**: If the consumer is slow, events queue up in memory. For high-throughput scenarios, consider adding queue size limits.

4. **Session-scoped**: Each `StreamingTextOutput` instance should be used with a single `AgentSession`.

---

## Summary

The `StreamingTextOutput` class provides a clean way to tap into LiveKit Agents' internal text streaming without modifying the framework. By implementing the `TextOutput` interface and using an async queue, it bridges the gap between the framework's callback-based internal API and the async iterator pattern preferred for WebSocket streaming.

Key points:
- Attach to `sess.output.transcription` before starting the agent
- Consume the stream with `async for event in streaming_output`
- Events include both deltas and accumulated text for flexibility
- The stream resets after each generation, supporting multiple queries
- Works seamlessly with the existing `RunResult` API for tool calls and assertions



