# Intelligent Filler Detection for LiveKit Agents

## 🎯 Overview

This feature adds intelligent filler word detection to LiveKit Agents, preventing false interruptions from background sounds like "uh", "umm", "hmm" while still allowing genuine user interruptions.

## ✨ What Changed

### New Components

1. **`FillerDetector` class** (`livekit/agents/voice/filler_detector.py`)
   - Detects and filters filler words from transcripts
   - Supports multi-language filler detection
   - Runtime-configurable filler word lists
   - Confidence-based filtering for low-quality audio

2. **AgentSession Integration**
   - New parameters: `filler_words`, `filler_min_confidence`, `filler_languages`, `filler_enable_logging`
   - Automatic integration with AudioRecognition pipeline
   - Zero performance overhead when disabled

3. **AudioRecognition Enhancement**
   - Filler detection applied to FINAL, INTERIM, and PREFLIGHT transcripts
   - Preserves existing VAD and turn detection behavior
   - Thread-safe async implementation

## 🎬 How It Works

### Decision Logic

1. **Speech Activity Detection (VAD)**: Continues to detect speech activity as before.
2. **Filler Detection**: During speech, applies filler detection logic:
   - If speech contains filler words (e.g., "uh", "umm") and is below the `filler_min_confidence` threshold, it is ignored.
   - If speech contains valid commands (e.g., "stop", "wait"), it is processed immediately.
   - Filler words are filtered out from the transcript before processing.
3. **Transcription and Processing**: The filtered transcript is then processed for agent actions.

### Configuration Parameters

- `filler_words`: List of words to treat as fillers (default: common English fillers)
- `filler_min_confidence`: Minimum confidence (0.0-1.0) to consider speech valid
- `filler_languages`: Languages for automatic filler word loading (e.g., `['en', 'hi']`)
- `filler_enable_logging`: Toggle debug logging for filler detection events

## 🚀 Getting Started

### Setup

1. **Install Dependencies**: Ensure you have the latest LiveKit Agents package.
2. **Configure Agent**: Update your agent configuration to include filler detection parameters.

### Running the Agent

Start your agent as usual. Filler detection is integrated into the AudioRecognition pipeline and requires no additional steps to activate.

## ✅ What Works

- **Filler Detection During Agent Speech**: Ignores "uh", "umm", "hmm" when agent is speaking
- **Valid Speech When Agent Quiet**: Registers all speech (including fillers) when agent is not speaking
- **Real Interruption Detection**: Immediately stops agent on meaningful commands like "wait", "stop"
- **Mixed Content Handling**: Detects meaningful words within filler-heavy speech ("umm okay stop" → interrupts)
- **Low Confidence Filtering**: Treats low-confidence transcripts as potential fillers during agent speech
- **Multi-Language Support**: Pre-configured filler words for English, Hindi, Spanish, French
- **Dynamic Updates**: Runtime modification of filler word lists via `update_filler_words()`
- **Statistics Tracking**: Comprehensive metrics via `get_stats()` method

### Tested Scenarios

| Scenario | User Input | Agent Speaking | Confidence | Expected Behavior | Status |
|----------|-----------|----------------|------------|-------------------|--------|
| Filler during speech | "uh", "hmm" | Yes | High | Ignored, agent continues | ✅ |
| Real interruption | "wait", "stop" | Yes | High | Agent stops immediately | ✅ |
| Filler when quiet | "umm" | No | High | Registered as valid speech | ✅ |
| Mixed content | "umm okay stop" | Yes | High | Agent stops (contains command) | ✅ |
| Low confidence murmur | "hmm yeah" | Yes | Low (<0.3) | Ignored as filler | ✅ |
| Empty transcript | "" | Yes/No | Any | Ignored, no action | ✅ |

## 🐛 Known Issues

1. **STT Latency**: In rare cases, very fast filler detection may race with VAD events
2. **Language Detection**: Mixed-language speech may require manual filler word configuration
3. **Edge Case**: Single-word meaningful interruptions that match filler patterns (e.g., "um" as a person's name)

**Workarounds**:
- Adjust `min_confidence_threshold` to reduce false positives
- Use `update_filler_words()` to remove specific words from filler list
- Monitor `get_stats()` output to tune detection sensitivity

## 🧪 Steps to Test

### 1. Setup Environment

```bash
# Navigate to project directory
cd c:\Users\HP\OneDrive\Desktop\tech\agents

# Install dependencies (if not already done)
uv sync --extra openai --extra silero --dev

# Set environment variables
# Create .env file with:
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=your_api_key
LIVEKIT_API_SECRET=your_api_secret
OPENAI_API_KEY=your_openai_key
```

### 2. Run the Agent

```bash
# Start agent in development mode
uv run python examples/voice_agents/simple_agent.py dev
```

Expected output:

```
... agent logs ...
Filler detection initialized with languages: ['en', 'hi']
Agent is ready and listening for commands.
```

### Test 2: Live Agent Testing

You can test with either example:

#### Option A: Simple Agent (Minimal Example)
```bash
uv run python examples/voice_agents/simple_agent.py dev
```

#### Option B: Basic Agent (Full-Featured Example)
```bash
uv run python examples/voice_agents/basic_agent.py dev
```

**Both agents include:**
- ✅ Filler detection with English + Hindi support
- ✅ Confidence threshold filtering (0.3)
- ✅ Debug logging enabled

