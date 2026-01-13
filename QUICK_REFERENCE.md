# LiveKit Voice Agent - Quick Reference

## Installation & Setup (3 steps)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Verify Installation
```bash
python verify_dependencies.py
```

### Step 3: Configure Environment
Create `.env` file in `examples/voice_agents/` or project root:
```bash
LIVEKIT_URL=ws://localhost:7880
LIVEKIT_API_KEY=your_api_key
LIVEKIT_API_SECRET=your_api_secret

LIVEKIT_SOFT_ACKS=okay,yeah,uhhuh,ok,hmm,right,good
```

---

## Package Versions at a Glance

| Package | Version | Purpose |
|---------|---------|---------|
| livekit-agents | 1.3.6+ | Core framework |
| livekit-plugins-deepgram | 1.3.6+ | Speech-to-Text |
| livekit-plugins-silero | 1.3.6+ | Voice Activity Detection |
| livekit-plugins-cartesia | 1.3.6+ | Text-to-Speech |
| numpy | 1.26.0+ | Numeric arrays |
| onnxruntime | 1.18-1.23.1 | ML inference (Silero) |
| pydantic | 2.0+ | Data validation |
| python-dotenv | 1.0+ | Env config loading |

---

## Common Issues & Solutions

### Issue: ModuleNotFoundError: No module named 'livekit'
**Solution:**
```bash
pip install -r requirements.txt --force-reinstall
```

### Issue: "No audio devices found"
**Solution:**
```bash
python -c "import sounddevice as sd; print(sd.query_devices())"
# Check if you have microphone/speaker connected
```

### Issue: Silero VAD not loading (ONNX error)
**Solution:**
```bash
pip install "onnxruntime>=1.18,<=1.23.1" --force-reinstall
```

### Issue: "DEEPGRAM_API_KEY not found"
**Solution:**
```bash
# Make sure .env is in correct location:
# - agents/examples/voice_agents/.env  OR
# - agents/.env

# Or set directly:
export DEEPGRAM_API_KEY=your_key_here
```

### Issue: "Soft-acks being processed as interrupts"
**Solution:**
```bash
# Check .env has LIVEKIT_SOFT_ACKS configured:
LIVEKIT_SOFT_ACKS=okay,yeah,uhhuh,ok,hmm,right,good

# Or add your custom soft-acks:
LIVEKIT_SOFT_ACKS=okay,yeah,right,sure,good
```

---

## Running Your Agent

```bash
# Navigate to examples directory
cd examples/voice_agents/

# Run your agent script
python minimal_worker.py
```

---

## Dependency Tree Summary

```
livekit-agents (core)
├── livekit (WebRTC)
├── livekit-api (API client)
├── livekit-protocol (Protobuf)
├── livekit-blingfire (Text processing)
├── numpy (Math)
├── pydantic (Validation)
├── aiohttp (HTTP)
└── opentelemetry-* (Tracing)

plugins/
├── livekit-plugins-deepgram (STT)
│   └── livekit-agents
├── livekit-plugins-silero (VAD)
│   ├── livekit-agents
│   ├── onnxruntime
│   └── numpy
└── livekit-plugins-cartesia (TTS)
    └── livekit-agents
```

---

## Environment Variables Reference

| Variable | Required | Example | Purpose |
|----------|----------|---------|---------|
| LIVEKIT_URL | Yes | ws://localhost:7880 | LiveKit server URL |
| LIVEKIT_API_KEY | Yes | dev_key | API authentication |
| LIVEKIT_API_SECRET | Yes | dev_secret | API authentication |
| DEEPGRAM_API_KEY | Yes | xxxxx | STT provider key |
| CARTESIA_API_KEY | Yes | xxxxx | TTS provider key |
| OPENAI_API_KEY | Yes | sk-xxxxx | LLM provider key |
| LIVEKIT_SOFT_ACKS | No | okay,yeah,right | Soft acknowledgments |

---

## Useful Commands

```bash
# List installed packages and versions
pip list | grep livekit

# Check specific package version
pip show livekit-agents

# Upgrade all packages
pip install -r requirements.txt --upgrade

# Check Python version
python --version

# Test audio devices
python -c "import sounddevice; print(sounddevice.query_devices())"

# Verify all dependencies
python verify_dependencies.py

# Run specific example
python examples/voice_agents/minimal_worker.py
```

---

## Key Features Enabled

✓ **Speech-to-Text**: Deepgram Nova-3  
✓ **Voice Activity Detection**: Silero  
✓ **Text-to-Speech**: Cartesia Sonic-2  
✓ **LLM Integration**: OpenAI, Google, Anthropic, Azure, AWS, Groq, Mistral  
✓ **Soft-Ack Filtering**: Blocks acknowledgments during agent speech  
✓ **WebRTC**: Real-time audio/video  
✓ **Observability**: OpenTelemetry + Prometheus  
✓ **Audio Processing**: NumPy + ONNX Runtime  

---

## Next Steps

1. **Install**: `pip install -r requirements.txt`
2. **Verify**: `python verify_dependencies.py`
3. **Configure**: Create `.env` with API keys
4. **Run**: `python examples/voice_agents/minimal_worker.py`
5. **Debug**: Check logs for `[SOFTACK_CONFIG]`, `[VAD_]`, `[INTERRUPT_]` messages

---

**Documentation**: https://docs.livekit.io  
**GitHub**: https://github.com/livekit/agents  
**Discord**: https://livekit.io/join-discord
