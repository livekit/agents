# How to Add Filler Handling to Your Existing Agent

This guide shows how to integrate the filler interruption handler into any existing LiveKit agent with minimal changes.

---

## 🔧 3-Step Integration

### Step 1: Import the Handler

Add to your agent file:
```python
from filler_interrupt_handler import FillerInterruptionHandler
```

### Step 2: Initialize Handler

In your `entrypoint()` function, before starting the session:
```python
# Create filler handler
filler_handler = FillerInterruptionHandler(
    ignored_words=['uh', 'um', 'umm', 'hmm', 'haan']
)
```

### Step 3: Add Event Hooks

After creating your `AgentSession`, add these hooks:
```python
@session.on("agent_state_changed")
def on_agent_state_changed(event):
    filler_handler.update_agent_state(event.new_state)

@session.on("user_input_transcribed")
def on_user_transcript(event):
    decision = filler_handler.analyze_transcript(
        transcript=event.transcript,
        is_final=event.is_final
    )
    
    if not decision.should_interrupt:
        logger.info(f"Ignoring filler: {event.transcript}")
    else:
        logger.info(f"Valid speech: {event.transcript}")
```

**That's it!** Your agent now intelligently filters fillers.

---

## 📝 Complete Example: Before & After

### BEFORE (basic_agent.py)

```python
import logging
from dotenv import load_dotenv
from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, cli
from livekit.plugins import silero, openai, cartesia

logger = logging.getLogger("my-agent")
load_dotenv()

async def entrypoint(ctx: JobContext):
    session = AgentSession(
        vad=silero.VAD.load(),
        llm=openai.LLM(model="gpt-4o-mini"),
        stt="assemblyai/universal-streaming:en",
        tts=cartesia.TTS(),
    )
    
    await session.start(
        agent=Agent(instructions="You are a helpful assistant."),
        room=ctx.room
    )

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
```

### AFTER (with filler handling)

```python
import logging
from dotenv import load_dotenv
from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, cli
from livekit.plugins import silero, openai, cartesia

# ✅ Step 1: Import handler
from filler_interrupt_handler import FillerInterruptionHandler
from livekit.agents.voice import AgentStateChangedEvent, UserInputTranscribedEvent

logger = logging.getLogger("my-agent")
load_dotenv()

async def entrypoint(ctx: JobContext):
    # ✅ Step 2: Initialize handler
    filler_handler = FillerInterruptionHandler(
        ignored_words=['uh', 'um', 'umm', 'hmm', 'haan']
    )
    
    session = AgentSession(
        vad=silero.VAD.load(),
        llm=openai.LLM(model="gpt-4o-mini"),
        stt="assemblyai/universal-streaming:en",
        tts=cartesia.TTS(),
        # Enable false interruption handling (recommended)
        resume_false_interruption=True,
        false_interruption_timeout=1.0,
    )
    
    # ✅ Step 3: Add event hooks
    @session.on("agent_state_changed")
    def on_agent_state_changed(event: AgentStateChangedEvent):
        filler_handler.update_agent_state(event.new_state)
        logger.info(f"Agent: {event.old_state} -> {event.new_state}")
    
    @session.on("user_input_transcribed")
    def on_user_transcript(event: UserInputTranscribedEvent):
        decision = filler_handler.analyze_transcript(
            transcript=event.transcript,
            is_final=event.is_final
        )
        
        if not decision.should_interrupt:
            logger.info(f"🚫 Ignoring filler: '{event.transcript}'")
        else:
            logger.info(f"✅ Valid speech: '{event.transcript}'")
    
    await session.start(
        agent=Agent(instructions="You are a helpful assistant."),
        room=ctx.room
    )

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
```

---

## 🎨 Customization Options

### Option 1: Environment-Based Configuration

```python
import os

# Load filler words from .env
filler_words_env = os.getenv('IGNORED_FILLER_WORDS', '')
if filler_words_env:
    filler_words = [w.strip() for w in filler_words_env.split(',')]
else:
    filler_words = None  # Use defaults

filler_handler = FillerInterruptionHandler(ignored_words=filler_words)
```

In `.env`:
```bash
IGNORED_FILLER_WORDS=uh,um,umm,hmm,haan,yeah
```

### Option 2: Language-Specific Lists

```python
# English + Hindi
filler_handler = FillerInterruptionHandler(
    ignored_words=['uh', 'um', 'umm', 'hmm', 'haan', 'achha', 'theek']
)

# English + Spanish
filler_handler = FillerInterruptionHandler(
    ignored_words=['uh', 'um', 'umm', 'hmm', 'este', 'bueno', 'pues']
)
```

### Option 3: Runtime Updates

```python
# Add words during runtime
filler_handler.add_ignored_words(['okay', 'alright'])

# Remove words
filler_handler.remove_ignored_words(['yeah'])
```

### Option 4: Statistics Logging

```python
# Add shutdown callback to log stats
async def log_stats():
    filler_handler.log_statistics()

ctx.add_shutdown_callback(log_stats)
```

---

## 🔍 Integration Patterns

### Pattern 1: Minimal (Just Logging)

```python
@session.on("user_input_transcribed")
def on_transcript(event):
    decision = filler_handler.analyze_transcript(event.transcript, event.is_final)
    logger.info(f"Decision: {decision.reason}")
```

### Pattern 2: With Statistics

```python
@session.on("user_input_transcribed")
def on_transcript(event):
    decision = filler_handler.analyze_transcript(event.transcript, event.is_final)
    
    if not decision.should_interrupt:
        logger.info(f"Filler ignored: {event.transcript}")
        # Could track custom metrics here
    else:
        logger.info(f"Valid input: {event.transcript}")
```

### Pattern 3: With Custom Actions

```python
@session.on("user_input_transcribed")
def on_transcript(event):
    decision = filler_handler.analyze_transcript(event.transcript, event.is_final)
    
    if not decision.should_interrupt:
        # Custom action for fillers
        logger.info(f"Filler detected: {event.transcript}")
        # e.g., send telemetry, update metrics, etc.
    else:
        # Custom action for valid speech
        logger.info(f"Valid speech: {event.transcript}")
        # e.g., trigger specific handling
```

---

## 📊 Verification

After integration, verify it works:

### 1. Check Logs

Look for these patterns:
```
INFO | Agent: idle -> speaking
INFO | 🚫 Ignoring filler: 'umm'
INFO | Agent: speaking -> listening
INFO | ✅ Valid speech: 'hello'
```

### 2. Test Scenarios

Run through:
- [ ] Say "umm" while agent speaking → Should see "Ignoring filler"
- [ ] Say "wait" while agent speaking → Should see "Valid speech"
- [ ] Say "umm" while agent quiet → Should see "Valid speech"

### 3. Review Statistics

At session end, check:
```
=== Filler Handler Statistics ===
Total transcripts: 25
Ignored fillers: 8
Valid interruptions: 12
Processed while idle: 5
```

---

## 🐛 Troubleshooting

### Issue: Handler not filtering

**Check:**
1. Are event hooks registered before `session.start()`?
2. Is `resume_false_interruption=True` set?
3. Are logs showing state changes?

**Fix:**
```python
# Ensure hooks are before session.start()
@session.on("user_input_transcribed")
def on_transcript(event):
    # ... handler code ...

# Then start
await session.start(...)
```

### Issue: All speech being ignored

**Check:**
1. Is agent state being updated correctly?
2. Are filler words too broad?

**Fix:**
```python
# Add debug logging
@session.on("agent_state_changed")
def on_state(event):
    filler_handler.update_agent_state(event.new_state)
    logger.debug(f"State updated to: {event.new_state}")  # Check this
```

### Issue: Fillers not being ignored

**Check:**
1. Is transcript text matching your word list?
2. Is punctuation affecting matching?

**Fix:**
```python
# Log what's being analyzed
decision = filler_handler.analyze_transcript(event.transcript, event.is_final)
logger.debug(f"Analyzed: '{event.transcript}' -> {decision.should_interrupt}")
```

---

## 📚 More Examples

### Example: Weather Agent with Filler Handling

See `filler_aware_agent.py` for a complete example with:
- Function tools
- Custom agent class
- Full event handling
- Statistics logging
- Environment configuration

### Example: Multi-Agent with Filler Handling

For multi-agent setups:
```python
# Create one handler per session
async def entrypoint(ctx: JobContext):
    handler1 = FillerInterruptionHandler(ignored_words=['uh', 'um'])
    session1 = AgentSession(...)
    
    handler2 = FillerInterruptionHandler(ignored_words=['hmm', 'yeah'])
    session2 = AgentSession(...)
    
    # Hook each handler to its session
    # ...
```

---

## ✅ Integration Checklist

- [ ] Import `FillerInterruptionHandler`
- [ ] Import event types (`AgentStateChangedEvent`, `UserInputTranscribedEvent`)
- [ ] Initialize handler before session
- [ ] Add `agent_state_changed` hook
- [ ] Add `user_input_transcribed` hook
- [ ] Set `resume_false_interruption=True` (recommended)
- [ ] Test with fillers while agent speaking
- [ ] Test with real speech while agent speaking
- [ ] Test with speech while agent quiet
- [ ] Verify logs show correct behavior

---

## 🎯 Next Steps

1. **Test thoroughly** with your specific use case
2. **Tune filler word list** for your domain
3. **Adjust timeout** based on your needs
4. **Monitor statistics** to optimize
5. **Add custom logic** as needed

---

## 📖 Further Reading

- `README_FILLER_HANDLER.md` - Full technical documentation
- `filler_aware_agent.py` - Complete example implementation
- `test_filler_handler.py` - Test suite with examples
- `QUICKSTART.md` - Setup and testing guide

---

**Questions?** Check the full documentation or examine `filler_aware_agent.py` for a working example!
