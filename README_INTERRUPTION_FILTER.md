# LiveKit Voice Interruption Filter

## Overview

This feature intelligently filters voice interruptions in LiveKit VoiceAssistant to prevent filler words (like "uh", "umm", "hmm", "haan") from prematurely stopping agent responses while still allowing genuine user interruptions.

## Problem Solved

The default LiveKit behavior treats all user speech during agent responses as interruptions. This causes:

- Agents stop mid-sentence when users say casual fillers
- Unnatural conversation flow
- Poor user experience, especially in multilingual contexts

## Solution

A filtering layer that:

1. **Tracks agent state** - Knows when agent is speaking
2. **Analyzes transcriptions** - Checks against ignored words list
3. **Considers confidence** - Uses ASR confidence scores for decisions
4. **Passes genuine interruptions** - Allows "stop", "wait", etc. through immediately

## Usage

### Basic Integration

```python
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli, llm
from livekit.plugins import openai, silero
from src.interruption_filter import InterruptionAwareVoiceAssistant

async def entrypoint(ctx: JobContext):
    # Create assistant with interruption filtering
    assistant = InterruptionAwareVoiceAssistant(
        vad=silero.VAD.load(),
        stt=openai.STT(),
        llm=openai.LLM(),
        tts=openai.TTS(),
    )

    await assistant.start(ctx.room)
```

### Configuration

Set environment variables:

```bash
# Comma-separated or JSON list
export IGNORED_WORDS='["uh","umm","hmm","haan","er"]'

# Confidence threshold (0-1)
export CONFIDENCE_THRESHOLD=0.6

# Enable runtime updates
export ENABLE_DYNAMIC_UPDATES=true
```

### Dynamic Updates (Optional)

```python
# Access config manager
config = assistant.config_manager

# Add new filler word
config.add_ignored_word("like")

# Remove a word
config.remove_ignored_word("hmm")

# Get current list
words = config.get_ignored_words()
```

## Architecture

```
User Speech → STT → Transcription Event
                         ↓
              [InterruptionFilter]
                         ↓
        ┌────────────────┴────────────────┐
        ↓                                 ↓
   IGNORE (filler)              ALLOW (genuine)
        ↓                                 ↓
   Continue Agent              Stop Agent & Respond
```

### Components

1. **ConfigManager** - Manages ignored words and thresholds
2. **InterruptionFilter** - Core filtering logic
3. **InterruptionAwareVoiceAssistant** - Wrapper with event interception

## Decision Logic

```
IF agent NOT speaking:
    → PASS_THROUGH (allow all)

IF transcription is empty:
    → IGNORE

IF transcription matches ignored word:
    IF confidence < threshold OR NOT final:
        → IGNORE
    ELSE:
        → ALLOW (might be intentional repetition)

ELSE:
    → ALLOW (genuine interruption)
```

## Testing

Run tests:

```bash
python -m pytest tests/test_interruption_handling.py -v
```

### Test Scenarios Covered

- ✅ Filler words during agent speech are ignored
- ✅ Low confidence fillers are filtered
- ✅ High confidence fillers can interrupt (user emphasis)
- ✅ Genuine commands pass through immediately
- ✅ All speech passes when agent is quiet
- ✅ Empty transcriptions are ignored
- ✅ Dynamic configuration updates work

## Performance

- **No added latency** - Filtering happens in-line with transcription
- **Thread-safe** - Config updates use locks
- **Memory efficient** - Set-based lookup O(1)
- **VAD unaffected** - No changes to voice activity detection

## Supported Languages

Default fillers include:

- **English**: uh, umm, hmm, er, like, you know
- **Hindi**: haan, haan-haan

Easily extensible via `IGNORED_WORDS` configuration.

## Evaluation Metrics

To evaluate effectiveness:

1. **Interruption precision**: % of ignored events that were fillers
2. **Interruption recall**: % of genuine interruptions that passed
3. **Conversation flow**: User feedback on naturalness
4. **Response latency**: No degradation vs baseline

## Troubleshooting

### Agent still stops on fillers

- Check `CONFIDENCE_THRESHOLD` - lower it (e.g., 0.4)
- Add more filler variants to `IGNORED_WORDS`
- Check logs for decision reasoning

### Agent doesn't respond to commands

- Ensure commands aren't in ignored words list
- Check if confidence scores are too low
- Verify agent speaking state is tracked correctly

### Logs

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Future Enhancements

- ML-based filler detection
- Context-aware filtering (user speech patterns)
- Language-specific models
- A/B testing framework

## License

Same as LiveKit agents project.

## Author

Developed for LiveKit Bounty Challenge 2024
