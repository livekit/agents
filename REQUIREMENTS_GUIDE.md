# LiveKit Voice Agents - Dependency Installation Guide

## Overview

This `requirements.txt` file contains all dependencies needed to run a complete LiveKit voice agent system with:
- **STT (Speech-to-Text)**: Deepgram Nova-3
- **VAD (Voice Activity Detection)**: Silero
- **TTS (Text-to-Speech)**: Cartesia Sonic-2
- **LLM Integration**: OpenAI, Google, Anthropic, Azure, AWS, Groq, Mistral
- **WebRTC/Audio**: LiveKit WebRTC, Audio processing libraries
- **Observability**: OpenTelemetry, Prometheus metrics

## Installation

### Quick Start
```bash
# Install all dependencies from requirements.txt
pip install -r requirements.txt

# Or upgrade existing installation
pip install -r requirements.txt --upgrade
```

### Minimal Installation (Core Only)
If you only need the core agent framework without optional plugins:
```bash
# Create a minimal requirements file with just core packages
pip install livekit-agents>=1.3.6 \
    livekit-plugins-deepgram>=1.3.6 \
    livekit-plugins-silero>=1.3.6 \
    livekit-plugins-cartesia>=1.3.6 \
    python-dotenv>=1.0.0
```

## Dependencies Breakdown

### Core Framework
- **livekit-agents**: Main agent framework
- **livekit**, **livekit-api**, **livekit-protocol**: WebRTC and platform APIs

### Required Plugins (for the current setup)
- **livekit-plugins-deepgram**: STT provider (Deepgram Nova-3)
- **livekit-plugins-silero**: VAD provider (Silero)
- **livekit-plugins-cartesia**: TTS provider (Cartesia Sonic-2)

### Audio Processing
- **numpy>=1.26.0**: Numerical array operations
- **onnxruntime**: ML runtime for Silero VAD model
- **sounddevice**: Audio device handling
- **av>=14.0.0**: Audio/video codec support

### Web & Communication
- **aiohttp**: Async HTTP client/server
- **websockets**: WebSocket support
- **protobuf**: Protocol buffers for message serialization

### Configuration & Logging
- **python-dotenv**: Load environment variables from .env files
- **colorama**: Colored terminal output
- **prometheus-client**: Metrics collection

### Observability
- **opentelemetry-api**: Tracing API
- **opentelemetry-sdk**: Tracing implementation
- **opentelemetry-exporter-otlp**: OTLP exporter for traces

## Environment Configuration

### Required Environment Variables
Create a `.env` file in your project root:

```bash
# LiveKit Server Configuration
LIVEKIT_URL=ws://localhost:7880
LIVEKIT_API_KEY=your_api_key
LIVEKIT_API_SECRET=your_api_secret

# STT Configuration (Deepgram)
DEEPGRAM_API_KEY=your_deepgram_key

# TTS Configuration (Cartesia)
CARTESIA_API_KEY=your_cartesia_key

# LLM Configuration (OpenAI example)
OPENAI_API_KEY=your_openai_key

# Optional: Soft-ack Configuration
LIVEKIT_SOFT_ACKS=okay,yeah,uhhuh,ok,hmm,right,good
```

### Environment Loading
The framework automatically loads from `.env` file using the enhanced search mechanism:
1. Current working directory
2. Parent directories (up to 10 levels)
3. `examples/voice_agents/.env` (for development)
4. Fallback to `.env.example`

## Troubleshooting

### Import Errors
If you encounter import errors, ensure all dependencies are installed:
```bash
pip install -r requirements.txt --force-reinstall
```

### Audio Issues
For audio device problems:
```bash
python -c "import sounddevice; print(sounddevice.query_devices())"
```

### ONNX Runtime Issues
If Silero VAD doesn't load, check ONNX runtime version:
```bash
python -c "import onnxruntime; print(onnxruntime.__version__)"
```

Ensure version is between 1.18 and 1.23.1 (as specified in requirements.txt).

### Plugin Version Conflicts
All plugins should be >=1.3.6. If you have version conflicts:
```bash
pip list | grep livekit
# Update any outdated packages
pip install --upgrade livekit-agents livekit-plugins-*
```

## Version Information

- **Python Version**: 3.9+ (tested with 3.13)
- **livekit-agents**: 1.3.6+
- **All Plugins**: 1.3.6+ (compatible versions)
- **onnxruntime**: 1.18 - 1.23.1 (Silero requirement)
- **numpy**: 1.26.0+ (for audio processing)

## Development Setup

For development with the framework:

```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest pytest-asyncio black flake8 mypy

# Install the local agents package for development
cd livekit-agents
pip install -e .

# Install specific plugin for development
cd livekit-plugins/livekit-plugins-deepgram
pip install -e .
```

## Optional Dependencies

Uncomment in `requirements.txt` if needed:

```bash
# For MCP (Model Context Protocol) support (Python 3.10+)
mcp>=1.10.0,<2

# For image processing capabilities
pillow>=10.3.0

# For advanced audio processing
scipy>=1.10.0
```

## Adding Custom Plugins

To add other LiveKit plugins, append to requirements.txt:

```bash
# Example: Adding ElevenLabs TTS
echo "livekit-plugins-elevenlabs>=1.3.6" >> requirements.txt
pip install livekit-plugins-elevenlabs>=1.3.6
```

## Verification

Verify installation with a simple test:

```python
# test_installation.py
import livekit
from livekit.plugins import deepgram, silero, cartesia

print(f"✓ LiveKit: {livekit.__version__}")
print("✓ Deepgram plugin available")
print("✓ Silero plugin available")
print("✓ Cartesia plugin available")
```

Run it:
```bash
python test_installation.py
```

## Support

For dependency-related issues:
- Check LiveKit documentation: https://docs.livekit.io
- Report issues: https://github.com/livekit/agents
- Discord Community: https://livekit.io/join-discord

---

**Last Updated**: December 2025
**Compatible with**: livekit-agents v1.3.6+
