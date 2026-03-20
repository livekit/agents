# Grok Voice Agent API Example

This example demonstrates how to integrate xAI's Grok Voice Agents API with LiveKit Agents. It includes:

- Basic setup with xAI's voice-to-voice model
- Integration with built-in search tools so Grok can search x.com and the web

## Quickstart

### 1. Set up environment variables
> [!NOTE]
> All keys prefixed with `LIVEKIT_` can be obtained from a project created on [LiveKit Cloud](https://cloud.livekit.io) or from a self-hosted [LiveKit server](https://github.com/livekit/livekit) instance.

```bash
export LIVEKIT_API_KEY=<your-livekit-api-key>
export LIVEKIT_API_SECRET=<your-livekit-api-secret>
export LIVEKIT_URL=<your-livekit-url>
export XAI_API_KEY=<your-xai-api-key>
```

### 2. Install dependencies

```bash
uv add "livekit-agents[xai,silero,turn-detector]" livekit-plugins-noise-cancellation
```

### 3. Run the agent

#### Option 1: Console mode

You can talk to Grok directly in your terminal:

```bash
uv run grok_voice_agent_api.py console
```

<img width="2012" height="626" alt="livekit-grok-voice-agents-api-console" src="https://github.com/user-attachments/assets/44f48804-a8c7-4c83-ade9-6ce9bec502ac" />

#### Option 2: Agents playground

LiveKit hosts a playground environment where you can connect to the agent running on your machine. First run the agent:

```bash
uv run grok_voice_agent_api.py dev
```

Then: 

1. Open the playground environment in your browser: [https://agents-playground.livekit.io/](https://agents-playground.livekit.io/)
2. Select the LiveKit Cloud project that's linked to your agent and click "Connect to <project name>"

#### Option 3: Run your own custom frontend

LiveKit offers a bunch of [agent frontend](https://docs.livekit.io/frontends/) starter templates across languages. Like Option 2, the first step is to run the agent on your machine:

```bash
uv run grok_voice_agent_api.py dev
```

Then:

> [!NOTE]
> Whichever client platform you choose, to ensure your agent frontend and backend can connect to one another, use the same values for `LIVEKIT_` environment variables with both your frontend and agent.

Clone and run [agent-starter-react](https://github.com/livekit-examples/agent-starter-react) to interact with your agent in a browser 

Or alternatively: 
- [Android](https://github.com/livekit-examples/agent-starter-android)
- [Swift](https://github.com/livekit-examples/agent-starter-swift)
- [Flutter](https://github.com/livekit-examples/agent-starter-flutter)
- [React Native](https://github.com/livekit-examples/agent-starter-react-native)
- [ESP32](https://github.com/livekit/client-sdk-esp32)
- [Embed](https://github.com/livekit-examples/agent-starter-embed) your agent directly into your existing website

### 4. Try it out

Ask Grok something that requires real time information:

> "What is Elon Musk's most recent X post?"


