# LiveKit Agents for Python

Realtime framework for production-grade multimodal and voice AI agents.

See [https://docs.livekit.io/agents/](https://docs.livekit.io/agents/) for quickstarts, documentation, and examples.

## Filler Word Detection

Enhanced voice agent with filler word detection and filtering capabilities.

### What Changed

- Added `FillerWordDetector` class for detecting filler words using regex and phonetic matching
- Added `PhoneticMatcher` class with support for multiple phonetic algorithms (Soundex, Metaphone)
- Added `VADWrapper` to filter out filler words when the agent is speaking
- Integrated filler word detection into `AgentSession`

### New Parameters

#### FillerWordConfig
- `filler_words`: List of filler words to detect (default: common fillers like 'um', 'uh', 'like', 'you know')
- `case_sensitive`: Whether matching should be case sensitive (default: False)
- `word_boundary`: Whether to match whole words only (default: True)
- `phonetic_config`: Configuration for phonetic matching (see below)

#### PhoneticConfig
- `enabled`: Enable/disable phonetic matching (default: True)
- `algorithm`: Phonetic algorithm to use ('soundex', 'metaphone') (default: 'metaphone')
- `min_word_length`: Minimum word length for phonetic matching (default: 2)
- `custom_mappings`: Custom phonetic mappings for specific words

### What Works

- Detects common filler words and their variations (e.g., 'um', 'uh', 'umm', 'uhh')
- Handles multi-word fillers (e.g., 'you know', 'I mean')
- Supports custom phonetic mappings for specific words
- Filters out filler words when the agent is speaking
- Case-insensitive matching by default
- Handles repeated characters in words (e.g., 'sooo' -> 'so')

### Known Issues

- May have false positives with short words that sound similar to fillers
- Performance impact may be noticeable with very large custom word lists
- Phonetic matching may not work well with all languages

### Steps to Test

1. Start the agent with filler word detection enabled:

```python
from livekit.agents.voice import FillerWordConfig, PhoneticConfig

# Configure filler word detection
filler_config = FillerWordConfig(
    filler_words=["um", "uh", "like", "you know", "I mean"],
    phonetic_config=PhoneticConfig(
        enabled=True,
        algorithm="metaphone",
        min_word_length=2,
        custom_mappings={
            "hmm": ["hmmm", "hm", "hmmmm"]
        }
    )
)

# Create agent session with filler word detection
session = AgentSession(
    llm=openai.realtime.RealtimeModel(voice="coral"),
    filler_config=filler_config
)
```

2. Test with different inputs:
   - Single filler words ("um", "uh", "like")
   - Variations of fillers ("ummm", "uhhh", "sooo")
   - Multi-word fillers ("you know", "I mean")
   - Sentences with and without fillers

3. Verify that:
   - Filler words are correctly identified
   - The agent filters out fillers when speaking
   - Custom phonetic mappings work as expected

### Environment Details

- Python 3.8+
- Dependencies:
  - `livekit`
  - `jellyfish>=1.0.0` (for phonetic algorithms)
  - `pytest` (for running tests)

### Running Tests

```bash
# Run all tests
pytest tests/

# Run only filler detector tests
pytest tests/voice/test_filler_detector.py -v
```

```python
from dotenv import load_dotenv

from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions
from livekit.plugins import openai

load_dotenv()

async def entrypoint(ctx: agents.JobContext):
    await ctx.connect()

    session = AgentSession(
        llm=openai.realtime.RealtimeModel(
            voice="coral"
        )
    )

    await session.start(
        room=ctx.room,
        agent=Agent(instructions="You are a helpful voice AI assistant.")
    )

    await session.generate_reply(
        instructions="Greet the user and offer your assistance."
    )


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
```
