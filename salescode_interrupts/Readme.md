```markdown
#  Filler-Aware Real-Time Voice Agent

This project implements a **real-time conversational voice agent** that intelligently handles filler words and interruptions.

##  Features

- **Filler Suppression**: Ignores "um", "uh", "like", etc.
- **Instant Interruption Detection**: Stops speaking immediately when interrupted
- **Multi-language Support**: Extensible filler word lists
- **Real-Time Pipeline**: STT → LLM → TTS with LiveKit

##  Quick Start

### Installation
```bash
pip install "livekit-agents[openai,silero,deepgram,cartesia]~=1.0"
```

### Environment Setup
Create `.env` file:
```env
DEEPGRAM_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
CARTESIA_API_KEY=your_key_here
```

### Run Agent
```bash
python your_agent_file.py console
```

### Project Structure

```
agents/
│
├── salescode_interrupts/
│   ├── __init__.py
│   ├── interrupt_filter.py
│   └── commands.py
│
└── tests/
    └── test_filter_agent.py

```

##  Configuration

### Filler Words
```python
filler_words = {
    'english': ['um', 'uh', 'like', 'you know', 'actually'],
    'spanish': ['este', 'o sea', 'bueno']
}
```

### Interruption Phrases
```python
interruption_phrases = [
    "stop", "wait", "hold on", "interrupt"
]
```

## Requirements
- Python 3.8+
- LiveKit CLI
- API keys for Deepgram, OpenAI, Cartesia

## Pipeline
1. Deepgram STT → 2. Filler Filter → 3. OpenAI LLM → 4. Cartesia TTS

---

**Note**: Replace API keys with your actual keys.
```