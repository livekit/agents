# Backend: Real-Time Vision + Avatar Agent

Python backend for real-time AI agent with video vision capabilities and animated avatar responses.

## Features

- **Multimodal LLM**: GPT-4o-vision for seeing and understanding user video
- **Animated Avatars**: Anam avatar with lip-synced ElevenLabs TTS
- **Voice I/O**: Speech-to-text (Deepgram) and text-to-speech with VAD and turn detection
- **Self-Hosted**: Runs on local LiveKit server (no external media servers)
- **Hot Reload**: Development mode with automatic restart on code changes

## Quick Start

### Prerequisites

- Python 3.9+
- LiveKit server running: `livekit-server --dev`

### Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install package with dev dependencies
pip install -e ".[dev]"
```

### Configuration

Create `.env.local` in the project root:

```bash
# LiveKit
LIVEKIT_URL=ws://localhost:7880
LIVEKIT_API_KEY=devkey
LIVEKIT_API_SECRET=secret

# Language Models
OPENAI_API_KEY=sk-...

# Speech Services
DEEPGRAM_API_KEY=...
ELEVENLABS_API_KEY=...

# Avatar
ANAM_API_KEY=...
ANAM_AVATAR_ID=...
```

### Running the Agent

**Terminal mode** (no LiveKit server needed, local audio):
```bash
python src/agent.py console
```

**Development mode** (with hot reload):
```bash
python src/agent.py dev
```

**Production mode**:
```bash
python src/agent.py start
```

## Architecture

### Agent Pipeline

```
User Video + Voice
    ↓
VAD (Silero) + Turn Detection
    ↓
STT (Deepgram) → Text
    ↓
Vision (Capture frames from video)
    ↓
LLM (GPT-4o-vision) with Vision Context
    ↓
Generate Response
    ↓
TTS (ElevenLabs, 16kHz) → Audio
    ↓
Avatar (Anam) → Lip-synced Video
    ↓
Publish to Room
```

### Key Components

- **AgentServer**: Main process coordinating job scheduling
- **AgentSession**: Container managing agent-user interactions
- **RealtimeAgent**: Custom agent class with instructions
- **AvatarSession**: Anam avatar rendering with lip-sync
- **Voice Pipeline**: Configurable STT/LLM/TTS chain

## Configuration Options

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LIVEKIT_URL` | ws://localhost:7880 | LiveKit server WebSocket URL |
| `LIVEKIT_API_KEY` | devkey | API key for authentication |
| `LIVEKIT_API_SECRET` | secret | API secret for token generation |
| `USE_OPENAI_REALTIME` | False | Toggle to OpenAI Realtime API (experimental) |

### Code Configuration

**File**: `src/agent.py`

```python
# Toggle between APIs
USE_OPENAI_REALTIME = False  # Set True for Realtime API

# Adjust instructions in RealtimeAgent class
instructions = "You are a helpful AI assistant..."
```

### Fine-tuning

- **STT Model**: Change `model="deepgram/nova-3"` in inference.STT()
- **LLM Model**: Change `model="openai/gpt-4o-mini"` in inference.LLM()
- **TTS Voice**: Change voice ID in ElevenLabs configuration (Jessica: cgSgspJ2msm6clMCkdW9)
- **Sample Rate**: Must be 16kHz for Anam avatar compatibility
- **Preemptive Generation**: Disabled by default, enable for faster responses

## API Reference

### Agent Instructions

Customize agent behavior by modifying the `RealtimeAgent` class:

```python
class RealtimeAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="Your custom instructions here..."
        )
```

### RPC Methods

Backend can call RPC methods on frontend:

```python
# From within agent session
await session.rpc_call("methodName", data)
```

## Debugging

### Enable Debug Logging

```python
logger.setLevel(logging.DEBUG)
```

### Check LiveKit Connection

```bash
# Verify server is running
curl http://localhost:7880/json
```

### Test Audio/Video Streams

Use LiveKit CLI or web client to verify streams are publishing correctly.

## Performance Tuning

- **CPU**: Preemptive generation reduces perceived latency
- **Memory**: VAD model loading happens at startup (prewarm)
- **Bandwidth**: Video frame sampling (1fps speaking, 0.1fps silent)
- **Latency**: OpenAI Realtime API for <500ms end-to-end (experimental)

## Troubleshooting

### Import Errors
```
ModuleNotFoundError: No module named 'livekit'
```
→ Run `pip install -e ".[dev]"` or `pip install livekit-agents`

### Connection Refused
```
Connection refused - Trying to connect to ws://localhost:7880
```
→ Start LiveKit server: `livekit-server --dev`

### Avatar Not Initializing
```
ANAM_API_KEY is not set
```
→ Add `ANAM_API_KEY` and `ANAM_AVATAR_ID` to `.env.local`

### TTS Audio Issues
```
TTS sample rate mismatch
```
→ Ensure ElevenLabs TTS uses exactly 16kHz (hard requirement for Anam)

## Development

### Run Tests

```bash
pytest tests/ -v
```

### Type Checking

```bash
mypy src/agent.py --strict
```

### Code Formatting

```bash
ruff format src/
```

### Linting

```bash
ruff check src/ --fix
```

## Dependencies

- `livekit-agents[openai,silero,turn-detector]`: Core agent framework
- `livekit-plugins-anam`: Avatar rendering
- `livekit-plugins-noise-cancellation`: Audio preprocessing
- `livekit-plugins-deepgram`: STT provider
- `livekit-plugins-elevenlabs`: TTS provider
- `livekit-plugins-openai`: LLM provider
- `python-dotenv`: Environment variable loading

## Next Steps

1. Customize agent instructions for your use case
2. Test with frontend in `../frontend/`
3. Deploy with production LiveKit server
4. Monitor metrics and optimize latency

## References

- [LiveKit Agents Documentation](https://docs.livekit.io/agents/)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
- [Anam Avatar Platform](https://anam.ai/)
