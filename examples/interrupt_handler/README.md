# Filler-Aware Interruption Handling

This example demonstrates intelligent interruption handling that distinguishes between meaningful user interruptions and filler words/phrases (e.g., "uh", "umm", "hmm", "haan").

## What Changed

### New Modules

- **`filler_interrupt_filter.py`**: Core filter logic that:
  - Tracks agent speaking state
  - Processes user transcripts to distinguish fillers from real interruptions
  - Uses configurable thresholds for confidence and word counts
  - Provides structured logging for debugging

- **`filler_interrupt_agent.py`**: Example agent that:
  - Integrates the filter with LiveKit AgentSession
  - Uses Deepgram STT and TTS (no LLM required)
  - Speaks a hardcoded sentence for testing interruption handling
  - Listens to agent state changes and user transcriptions
  - Automatically interrupts TTS when valid interruptions are detected

### Key Features

1. **Filler Detection**: Ignores filler words ("uh", "umm", "hmm", "haan") when agent is speaking
2. **Interrupt Keywords**: Always interrupts on keywords like "stop", "wait", "no", "cancel"
3. **Confidence Thresholds**: Uses ASR confidence scores to filter low-confidence murmurs
4. **Meaningful Speech Detection**: Interrupts on meaningful speech based on token count and confidence
5. **State-Aware**: Only filters when agent is speaking; passes through all speech when agent is quiet

## What Works

- ✅ Filler words are ignored when agent is speaking (if confidence is low)
- ✅ Real interruptions (keywords or meaningful speech) stop the agent immediately
- ✅ Fillers are registered as normal speech when agent is quiet
- ✅ Mixed filler + command correctly triggers interruption
- ✅ Low-confidence background murmurs are ignored
- ✅ High-confidence meaningful speech triggers interruption
- ✅ Dynamic updates to ignored words list (runtime configuration)

## Known Issues

- **Confidence Scores**: `UserInputTranscribedEvent` doesn't include confidence scores from STT.
  The filter uses a default confidence of 0.7. For more accurate filtering, you would need to
  subscribe to STT events directly (requires more complex integration).
- Multi-language filler detection requires language-specific word lists
- Very fast speech may cause timing issues with state tracking

## Steps to Test

### Prerequisites

1. Install dependencies:
```bash
# Install from the repository root with all required plugins
pip install -e .[deepgram,silero]
```

Or if installing from PyPI:
```bash
pip install "livekit-agents[deepgram,silero]~=1.0"
```

**Required plugins:**
- `deepgram` - for STT (Speech-to-Text) and TTS (Text-to-Speech)
- `silero` - for VAD (Voice Activity Detection)

**Optional (for advanced turn detection):**
- `turn-detector` - for transformer-based turn detection (MultilingualModel)
  ```bash
  pip install livekit-plugins-turn-detector
  ```
  If installed, you can change `turn_detection="vad"` to `turn_detection=MultilingualModel()` in the code.

2. Set up environment variables in `.env` file (no OpenAI key needed):
```env
LIVEKIT_URL=wss://your-livekit-server.livekit.cloud
LIVEKIT_API_KEY=your_api_key
LIVEKIT_API_SECRET=your_api_secret
DEEPGRAM_API_KEY=your_deepgram_key
```

**Note**: This example does NOT require an OpenAI API key or any LLM. The agent speaks a hardcoded sentence for testing purposes.

### Running the Agent

1. **Console Mode** (for local testing):
```bash
python examples/interrupt_handler/filler_interrupt_agent.py console
```

2. **Dev Mode** (with hot reload):
```bash
python examples/interrupt_handler/filler_interrupt_agent.py dev
```

3. **Production Mode**:
```bash
python examples/interrupt_handler/filler_interrupt_agent.py start
```

### Testing Scenarios

1. **Filler while agent speaking**:
   - Wait for agent to start speaking
   - Say "uh" or "umm"
   - Expected: Agent continues speaking (ignored)

2. **Real interruption**:
   - Wait for agent to start speaking
   - Say "stop" or "wait"
   - Expected: Agent stops immediately

3. **Filler while agent quiet**:
   - Wait for agent to finish speaking
   - Say "umm"
   - Expected: Speech is registered (pass through)

4. **Mixed filler + command**:
   - Wait for agent to start speaking
   - Say "umm okay stop"
   - Expected: Agent stops (contains "stop" keyword)

5. **Meaningful speech**:
   - Wait for agent to start speaking
   - Say "I have a question" (high confidence)
   - Expected: Agent stops (meaningful speech)

### Configuration

You can configure the filter behavior via environment variables:

```env
# Filler words to ignore (comma-separated)
IGNORED_WORDS=uh,umm,hmm,haan

# Keywords that always trigger interruption (comma-separated)
INTERRUPT_KEYWORDS=stop,wait,hold on,no,cancel,pause

# Minimum ASR confidence for filler-only speech (0.0-1.0)
MIN_ASR_CONFIDENCE=0.55

# Minimum confidence for non-keyword interruptions (0.0-1.0)
INTERRUPT_CONFIDENCE=0.70

# Minimum number of meaningful tokens for interruption
MIN_MEANINGFUL_TOKENS=2
```

### Running Tests

```bash
pytest tests/test_filler_interrupt_filter.py -v
```

## Environment Details

- **Python Version**: 3.8+
- **Dependencies**:
  - `livekit-agents` (core framework)
  - `livekit-plugins-deepgram` (STT and TTS)
  - `livekit-plugins-silero` (VAD)
  - `livekit-plugins-turn-detector` (turn detection)
  - `python-dotenv` (for .env file loading)
- **No LLM Required**: This example uses hardcoded text, so no OpenAI or other LLM API keys are needed

## Architecture

```
User Speech → Deepgram STT → Transcript Event
                              ↓
                    FillerAwareInterruptFilter
                              ↓
                    Decision: ignore/interrupt/pass_through
                              ↓
                    If interrupt → session.interrupt()
                              ↓
                    Agent stops TTS immediately
```

## Logging

The filter logs decisions with structured information:
- `ignored_filler_low_conf`: Filler ignored due to low confidence
- `valid_interrupt_keyword`: Interruption triggered by keyword
- `valid_interrupt_confidence`: Interruption triggered by high confidence
- `valid_interrupt_tokens`: Interruption triggered by meaningful tokens
- `pass_through_quiet`: Speech passed through when agent not speaking

Check logs for:
```
INFO: Transcript decision: interrupt | Reason: valid_interrupt_keyword | Text: 'stop' | Confidence: 0.85
```

## Future Enhancements

- Multi-language filler detection
- Dynamic confidence threshold adjustment
- Machine learning-based filler classification
- Integration with more STT/TTS providers

