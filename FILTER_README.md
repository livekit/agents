# Interruption Filter for LiveKit Agents

## Problem

When the agent is speaking, it stops as soon as the user says words like “yeah”, “ok”, or “hmm”.
This feels unnatural because users often say these words just to show they are listening, not to interrupt.

## Solution

We added an interruption filter.

The filter checks:

What the user said
Whether the agent is currently speaking
If the agent is speaking and the user only says listening words (like “yeah” or “ok”), the agent keeps talking instead of stopping.

## How to Use

### Basic Usage

```python
from livekit.agents import AgentSession

session = AgentSession(
    stt="deepgram/nova-3",
    llm="openai/gpt-4.1-mini",
    tts="cartesia/sonic-2",
    vad=silero.VAD.load(),
    interruption_filter_enabled=True,  # enabled by default
)
```

### Custom Ignore Words

```python
session = AgentSession(
    # ... other params ...
    interruption_ignore_words=['yeah', 'ok', 'sure', 'gotcha'],
)
```

### Environment Variable

```bash
export LIVEKIT_INTERRUPTION_IGNORE_WORDS="yeah,ok,hmm,right"
```

### Disable the Filter

```python
session = AgentSession(
    # ... other params ...
    interruption_filter_enabled=False,
)
```

## Default Ignore Words

yeah, ok, okay, hmm, mhm, mm-hmm, uh-huh, right, aha, ah, oh, sure, yep, yup, gotcha, got it, alright, cool

## How It Works

1. The agent is speaking
2. The user talks
3. VAD detects the user’s voice
4. Speech is converted to text (STT)
5. The filter checks:
- Is the agent speaking?
- Are all the spoken words in the ignore list?
6. Decision:
- Yes → Ignore it, agent continues speaking
- No → Agent stops speaking


## Examples

**Agent speaking, user says "yeah":**
- Filter ignores it, agent continues

**Agent speaking, user says "wait":**
- Filter allows it, agent stops

**Agent silent, user says "yeah":**
- Filter allows it, agent responds

**Agent speaking, user says "yeah wait":**
- Filter allows it (contains "wait"), agent stops

## Testing

Run the test:
```bash
python test.py
```

All 4 test scenarios should pass.

## Implementation Details

The filter is in `filter.py`. It's integrated into `agent_activity.py` in the `_interrupt_by_audio_activity()` method.

When an interruption is detected, it checks the transcript and agent state before deciding whether to actually interrupt.

## Configuration

You can customize the ignore words list or disable the filter entirely. The filter is enabled by default because it improves the conversation flow.

## Performance

The filter adds less than 1ms of latency. It just does simple string matching on the transcript.
