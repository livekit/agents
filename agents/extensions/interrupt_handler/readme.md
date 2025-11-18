# Smart Voice Interrupt Handler (LiveKit Agent Extension)

This module adds real-time interruption detection into LiveKit voice agents.
It filters filler speech (e.g., “umm”, “haan”) and only interrupts the agent
when the user provides meaningful speech commands like “stop”, “wait”, “no”.

✔ No modification to LiveKit VAD or Core SDK  
✔ Runs inside transcription callback event loop  
✔ Fully async + thread-safe  
✔ Configurable ignored and command keywords  

---

## Key Features

| Scenario | Behavior |
|---------|----------|
| Filler while agent speaking | Ignored |
| Command while agent speaking | Interrupt agent immediately |
| Filler while silent | Counted as user speech |
| Confidence below threshold | Ignored |

---

# Integration (LiveKit transcription callback)

```python
async def on_transcription(text: str, confidence):
    await interrupt_handler.handle_asr(text, confidence)

session.stt.on("transcription", on_transcription)

The extension does not modify TTS / STT engines — only filters logical behavior.

# Configuration

| Variable             | Purpose                        | Default            |
| -------------------- | ------------------------------ | ------------------ |
| `IGNORED_WORDS`      | Words to ignore while speaking | uh, umm, hmm, haan |
| `COMMAND_WORDS`      | Words for interruption         | stop, wait, no     |
| `ASR_CONF_THRESHOLD` | Min confidence for validation  | 0.6                |

#Set via environment:

export IGNORED_WORDS="uh,umm,hmm,haan"
export COMMAND_WORDS="stop,wait,no"
export ASR_CONF_THRESHOLD=0.6

## Functional Test Example

#Run:

    python examples/test_interrupt_handler.py


#Expected:

        User says: umm
        IGNORED_FILLER

        User says: wait stop
        VALID_INTERRUPT
        >>> Agent interrupted and stopped speaking

        User says: umm
        USER_SPEECH

#File Structure

        agents/extensions/interrupt_handler/
        │
        ├── handler.py     # core logic
        ├── config.py      # configurable ignore + command lists
        └── README.md      # this file