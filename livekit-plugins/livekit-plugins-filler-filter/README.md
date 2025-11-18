# LiveKit Filler Word Filter - Intelligent Interruption Handling

## 🎯 What This Does

This plugin filters out filler word interruptions (like "uh", "um", "hmm") when the LiveKit agent is speaking, while immediately allowing genuine interruptions (like "wait", "stop").

## ✅ What Works

- ✅ Filters filler-only utterances while agent speaks
- ✅ Passes through commands and real interruptions immediately
- ✅ Works with any STT provider (Deepgram, OpenAI, etc.)
- ✅ Runtime configurable filler word lists
- ✅ Confidence-based filtering
- ✅ Multi-language support
- ✅ Built-in statistics and debugging

## ⚠️ Known Issues

1. In very rapid turn-taking (<100ms), there may be a small delay in state updates
2. Very uncommon multi-word fillers may need manual configuration

## 📋 Files Created

- `livekit_plugins/filler_filter/config.py` - Configuration management
- `livekit_plugins/filler_filter/state_tracker.py` - Agent state tracking
- `livekit_plugins/filler_filter/utils.py` - Text processing utilities
- `livekit_plugins/filler_filter/filler_filter.py` - Main wrapper implementation
- `tests/test_filler_filter.py` - Unit tests
- `README.md` - This file

## 🚀 Usage Example

```python
from livekit.plugins import deepgram, openai
from livekit_plugins.filler_filter import FillerFilterWrapper, AgentStateTracker

# Create state tracker and wrap STT
state_tracker = AgentStateTracker()
base_stt = deepgram.STT(model="nova-3")
filtered_stt = FillerFilterWrapper(stt=base_stt, state_tracker=state_tracker)

# Use in session
session = AgentSession(stt=filtered_stt, ...)
state_tracker.attach_to_session(session)
```
