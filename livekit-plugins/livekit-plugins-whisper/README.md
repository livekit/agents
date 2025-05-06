# LiveKit Plugins Whisper

Agent Framework plugin for Whisper (TalTechNLP/whisper-large-et).

## Installation

```bash
pip install livekit-plugins-whisper
```

## Usage

```python
from livekit.agents import AgentSession
from livekit.plugins.whisper import stt

agent = AgentSession(
    stt=stt.main,
    ...
)
```

## Pre-requisites

Requires PyTorch, transformers, pyaudio, numpy, and loguru. See stt.py for more details.