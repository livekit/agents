# Grok Voice Agent

You can build voice AI agents with xAI's Grok Voice Agent API and LiveKit Agents in a few lines of code.

This example includes:

- **Grok Realtime API** — Voice-to-voice with xAI's realtime model
- **Built-in web search** — Grok can search X and the web using the `XSearch()` and `WebSearch()` tools

## Quick Start

### 1. Set up your environment variables

```bash
export LIVEKIT_API_KEY=<your-livekit-api-key>
export LIVEKIT_API_SECRET=<your-livekit-api-secret>
export LIVEKIT_URL=<your-livekit-url>
export XAI_API_KEY=<your-xai-api-key>
```

### 2. Install dependencies

```bash
pip install "livekit-agents[xai,silero,turn-detector]" livekit-plugins-noise-cancellation
```

### 3. Run the agent

**Option A: Console mode** — Talk to Grok directly in your terminal:

```bash
python realtime_grok_agents_api.py console
```

**Option B: Web interface** — Run the agent in development mode and connect via a frontend:

```bash
python realtime_grok_agents_api.py dev
```

Then use the [agent-starter-react](https://github.com/livekit-examples/agent-starter-react) template to interact with your agent in a browser. Also available for [Android](https://github.com/livekit-examples/agent-starter-android), [Swift](https://github.com/livekit-examples/agent-starter-swift), [Flutter](https://github.com/livekit-examples/agent-starter-flutter), [React Native](https://github.com/livekit-examples/agent-starter-react-native), [ESP32](https://github.com/livekit/client-sdk-esp32), or [embed](https://github.com/livekit-examples/agent-starter-embed) it directly into your existing website.

## Try it out

Ask Grok something that requires real time information:

> "What is Elon Musk's most recent X post?"

Grok will use its built-in search tools to fetch and summarize the latest content!
