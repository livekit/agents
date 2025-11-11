# Interruption Filtering Extension (Examples)

This repository includes an optional extension that reduces false voice interruptions by ignoring filler-only or low-confidence user speech while the agent is speaking.

- While the agent is speaking, fillers like "uh", "umm", "hmm", "haan" are ignored.
- Real commands like "stop", "wait" interrupt immediately.
- When the agent is quiet, fillers are registered normally.

## Configure

Environment variables (comma-separated values for lists):

- `LIVEKIT_IGNORED_WORDS` (default: `uh,umm,hmm,haan`)
- `LIVEKIT_INTERRUPTION_KEYWORDS` (default: `stop,wait,hold on,one second,no,not that`)
- `LIVEKIT_MIN_ASR_CONFIDENCE` (default: `0.5`)

Example:

```bash
export LIVEKIT_IGNORED_WORDS="uh,umm,hmm,haan"
export LIVEKIT_INTERRUPTION_KEYWORDS="stop,wait,hold on,one second,no,not that"
export LIVEKIT_MIN_ASR_CONFIDENCE=0.5
```

## Test Manually

1. Start a voice agent (e.g., `examples/voice_agents/basic_agent.py`).
2. While the agent speaks, say only fillers (e.g., "umm"). The TTS should continue.
3. Say a command (e.g., "stop"). The agent should interrupt immediately.
4. When the agent is quiet, fillers (e.g., "umm") should be registered normally.
