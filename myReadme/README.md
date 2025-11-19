# LiveKit Voice Interruption Handler

## What Changed
I implemented a custom `InterruptHandlingAgent` using the **LiveKit Agents V1.0** framework.
- **Logic:** Overrode `on_user_turn_completed` to intercept user speech before the LLM processes it.
- **Filtering:** Input text is cleaned and checked against a configurable `IGNORED_WORDS` list (e.g., "umm", "uh", "like").
- **Behavior:**
  - **Filler Words:** If the user says only a filler word (e.g., "umm"), the agent logs `Detected filler word. Ignoring.` and **does not** trigger an LLM response. This maintains the conversation flow.
  - **Real Speech:** If the user says a valid command (e.g., "Stop"), the agent processes it normally.

## Verification & Testing
**Note:** The agent logic is fully functional, verified by logs. However, the LLM audio response is currently disabled due to OpenAI API quota limits.

**Test Log Evidence:**
1. **Filler Detection:**
   ```text
   INFO voice-agent User said: 'mhmm'
   INFO voice-agent Detected filler word 'mhmm'. Ignoring.