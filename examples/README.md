# LiveKit Agents Examples

This directory contains various examples demonstrating different capabilities and use cases for LiveKit agents. Each example showcases specific features, integrations, or workflows that can be built with the LiveKit Agents framework.

## 📁 Example Categories

### 🎙️ [Voice Agents](./voice_agents/)

A comprehensive collection of voice-based agent examples, including basic voice interactions, tool integrations, RAG implementations, and advanced features like multi-agent workflows and push-to-talk agents.

### 🖼️ [Avatar Agents](./avatar_agents/)

Examples showing how to integrate visual avatars with voice agents, including integrations with various avatar providers like Anam, Bey, BitHuman, Hedra, Simli, and Tavus.

### 🔄 [Warm Transfer](./warm-transfer/)

Demonstrates supervisor escalation workflows for call centers, showing how to implement warm transfers where agents can brief supervisors before connecting them to customers.

### 🚗 [Drive-Thru](./drive-thru/)

A complete drive-thru ordering system example that showcases interactive voice agents for food ordering with database integration and order management.

### 🏢 [Front Desk](./frontdesk/)

A front desk agent example demonstrating how to build customer service agents with calendar integration and appointment management capabilities.

### 🔧 [Primitives](./primitives/)

Basic building blocks and fundamental examples showing core LiveKit concepts like room connections, participant management, and basic audio/video handling.

### 🛠️ [Other](./other/)

Additional examples including text-only agents, various TTS providers, transcription services, and translation utilities.

## Running Examples

To run the examples, you'll need:

- A [LiveKit Cloud](https://cloud.livekit.io) account or a local [LiveKit server](https://github.com/livekit/livekit)
- API keys for the model providers you want to use in a `.env` file
- Python 3.9 or higher
- [uv](https://docs.astral.sh/uv/)

### Environment file

Create a `.env` file in the `examples` directory and add your API keys (see `examples/.env.example`):

```bash
LIVEKIT_URL="wss://your-project.livekit.cloud"
LIVEKIT_API_KEY="your_api_key"
LIVEKIT_API_SECRET="your_api_secret"
OPENAI_API_KEY="sk-xxx" # or any other model provider API key
# ... other model provider API keys as needed
```

### Install dependencies

From the repository root, run the following command:

```bash
uv sync --all-extras --dev
```

### Running an individual example

Run an example agent:

```bash
uv run examples/voice_agents/basic_agent.py console
```

Your agent is now running in the console.

For frontend support, use the [Agents playground](https://agents-playground.livekit.io) or the [starter apps](https://docs.livekit.io/agents/start/frontend/#starter-apps).

## 📖 Additional Resources

- [LiveKit Documentation](https://docs.livekit.io/)
- [LiveKit Agents Documentation](https://docs.livekit.io/agents/)
