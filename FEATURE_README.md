# Filler Word Filtering Implementation

## What Changed

Added intelligent interruption handling to distinguish filler words from real interruptions.

**New Files:**
- `livekit-agents/livekit/agents/voice/filler_filter.py` - Core filtering implementation
- `examples/voice_agents/filler_aware_agent.py` - Working example

**Modified:**
- `livekit-agents/livekit/agents/voice/__init__.py` - Added exports
- `examples/voice_agents/requirements.txt` - Added AssemblyAI plugin

## What Works

| User Input | Agent Speaking | Result |
|------------|----------------|--------|
| "uh", "hmm", "umm" | Yes | Continues speaking |
| "wait one second" | Yes | Stops immediately |
| "umm okay stop" | Yes | Stops (has non-filler) |
| "umm" | No | Registers as speech |

Features:
- Configurable filler words via environment variables
- Confidence threshold filtering
- Case-insensitive with punctuation handling
- Thread-safe, no SDK modifications

## Known Issues

- Multi-word fillers need each word in the list separately
- Default list is English (customize via `FILLER_WORDS` env var)
- Only filters interim transcripts (final transcripts always processed by design)

## Steps to Test

### 1. Install Dependencies

```bash
cd examples/voice_agents
pip install -r requirements.txt
```

### 2. Configure Environment

Create `.env` in `examples/voice_agents/`:

```bash
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=your_api_key
LIVEKIT_API_SECRET=your_api_secret
OPENAI_API_KEY=your_openai_key
ASSEMBLYAI_API_KEY=your_assemblyai_key
CARTESIA_API_KEY=your_cartesia_key

# Optional
FILLER_WORDS=uh,umm,hmm,haan,huh
FILLER_CONFIDENCE_THRESHOLD=0.5
```

### 3. Run Agent

```bash
python3 filler_aware_agent.py dev
```

### 4. Connect & Test

Generate token via LiveKit Dashboard (https://cloud.livekit.io) or CLI:
```bash
lk token create --join --room test-room --identity user
```

Connect at https://agents-playground.livekit.io with your token.

**Test scenarios:**
1. Say "uh" while agent speaks → agent continues
2. Say "stop" while agent speaks → agent stops
3. Say "umm" when agent quiet → agent responds

Watch agent logs to see filtering in action.

## Environment Details

- Python 3.9+
- LiveKit Agents SDK 1.2.18+
- Tested: AssemblyAI (STT), OpenAI GPT-4 (LLM), Cartesia (TTS)
- Platform: macOS, Linux, Windows

## How It Works

`FillerFilteredAgentActivity` extends `AgentActivity` and overrides `on_interim_transcript()`:

1. Check if agent is speaking (`_current_speech` state)
2. Normalize and tokenize transcript
3. If all words are fillers → ignore interruption
4. If any non-filler word → process interruption
5. If agent quiet → always process

Custom `FillerAwareAgentSession` injects the filtered activity during `_update_activity()`.
