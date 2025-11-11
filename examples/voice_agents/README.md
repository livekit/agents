# LiveKit Interrupt Handler Challenge

## Overview

This project implements a `FillerInterruptFilter` extension for a LiveKit voice agent.
The filter distinguishes between filler sounds (such as “uh”, “umm”, “hmm”) and real user interruptions (such as “wait”, “stop”, “no”) during a conversation.

It improves the conversational flow by allowing the agent to ignore meaningless filler sounds and continue speaking naturally, while still reacting immediately to meaningful user inputs.

The complete implementation is in `examples/voice_agents/solved.py`.

---

## What Changed

A new class, `FillerInterruptFilter`, was added to the agent.
This class hooks into the `AgentSession`’s transcription events (`user_input_transcribed` or `transcription`) and processes user speech in real time.

### Core Logic

1. Checks if the agent is currently speaking.
2. Tokenizes and normalizes the user transcript (lowercasing, removing punctuation, collapsing repeated characters).
3. Calculates a filler ratio to measure how much of the input consists of filler words.
4. If the ratio exceeds a configurable threshold (default 0.8), the input is ignored and TTS playback continues.
5. If the input includes meaningful words (for example, “stop” or “wait”), it is treated as a real interruption.
6. When the agent is not speaking, all user speech is treated as valid.

---

## Key Features

* **Real-time filler detection:** Ignores filler interjections while the agent is speaking.
* **Confidence-based filtering:** Uses ASR confidence scores to reject uncertain speech.
* **Dynamic configuration:** Allows updating the list of ignored filler words at runtime.
* **Non-invasive integration:** Hooks into LiveKit’s event system without modifying the SDK.
* **Logging and metrics:** Emits `filler_ignored`, `filler_valid`, and `filler_resume_suggested` events.
* **Asynchronous and thread-safe:** Uses `asyncio.Lock()` to handle parallel events safely.

---

## Example Behavior

| User Transcript | Agent Speaking? | Filler Ratio | Behavior                           |
| --------------- | --------------- | ------------ | ---------------------------------- |
| uh umm          | Yes             | 1.0          | Ignored (agent continues speaking) |
| uh hello        | Yes             | 0.5          | Treated as valid interruption      |
| hmm             | Yes             | 1.0          | Ignored                            |
| I want pizza    | Yes             | 0.0          | Valid interruption                 |
| umm             | No              | 1.0          | Processed as normal input          |

---

## Configuration Options

| Parameter                           | Description                                                 | Default                                         |
| ----------------------------------- | ----------------------------------------------------------- | ----------------------------------------------- |
| `ignored_words`                     | List of filler words to ignore                              | `{"uh", "umm", "hmm", "haan", "achha", "arre"}` |
| `min_confidence`                    | Minimum ASR confidence to trust the input                   | `0.45`                                          |
| `filler_ratio_threshold`            | Ratio threshold to decide if input is filler                | `0.8`                                           |
| `enable_confidence_filter`          | Whether to use ASR confidence                               | `True`                                          |
| `min_non_filler_for_low_confidence` | Minimum count of non-filler tokens for low-confidence input | `2`                                             |

---

## Steps to Test

### 1. Local Plugin Test (No LiveKit Required)

1. Set the environment variable:

   **Bash:**

   ```bash
   export LOCAL_PLUGIN_TEST=1
   ```

   **Windows CMD:**

   ```bash
   set LOCAL_PLUGIN_TEST=1
   ```

2. Run the script:

   ```bash
   python examples/voice_agents/solved.py
   ```

3. Observe the console output. Example:

   ```
   >>> Running local filler-interrupt-filter test (no LiveKit required)
   ('emit:filler_ignored', {'transcript': 'uh umm', 'reason': 'filler_ratio(1.00)'})
   ('emit:filler_valid', {'transcript': 'uh hello', 'confidence': 0.9})
   >>> Local test finished.
   ```

This verifies that filler inputs are ignored and meaningful ones are accepted.

---

### 2. Full Agent Test (LiveKit Session)

1. Create a `.env` file with LiveKit credentials:

   ```
   LIVEKIT_URL=wss://<your-server>.livekit.cloud
   LIVEKIT_API_KEY=<your_api_key>
   LIVEKIT_API_SECRET=<your_api_secret>
   ```

2. Start the agent worker in development mode:

   ```bash
   python examples/voice_agents/solved.py dev
   ```

3. Open a console session:

   ```bash
   python examples/voice_agents/solved.py console
   ```

4. Speak sample phrases to verify:

   * Filler words → ignored
   * Real words → agent responds or stops as expected

---

## Design Summary

* **Encapsulation:** Implemented as a standalone class without changing core SDK.
* **Event-driven:** Reacts to live transcription updates.
* **Context awareness:** Works only when the agent is speaking.
* **Extensible:** Supports adding or replacing filler word lists dynamically.

---

## File Structure

```
examples/
└── voice_agents/
    ├── solved.py          # Main file with agent and filler filter
    ├── ...
```

---

## Example Logs

```
[filler_filter|ignored] uh umm (reason=filler_ratio(1.00))
[filler_filter|valid] uh hello (confidence=0.90)
[filler_filter|ignored] hmm (reason=filler_ratio(1.00))
[filler_filter|valid] I want pizza (confidence=0.95)
```

---

## Technical Details

* **Language:** Python 3.10
* **Framework:** LiveKit Agents SDK
* **Libraries:** `asyncio`, `re`, `dotenv`, `logging`
* **Models Used:**

  * STT: `assemblyai/universal-streaming:en`
  * LLM: `openai/gpt-4.1-mini`
  * TTS: `cartesia/sonic-2`
  * Optional VAD and turn detection: `silero`, `MultilingualModel`

---

## Expected Behavior

The agent:

* Ignores filler-only utterances while speaking.
* Immediately reacts to meaningful speech interruptions.
* Logs both ignored and valid speech events clearly.
* Provides a configurable and modular interruption management system.

This meets the goal of building a real-time speech interruption handler that improves agent interactivity.

---

## Author Notes

* Main file: `examples/voice_agents/solved.py`
* Includes both the agent implementation and the `FillerInterruptFilter` plugin.
* Compatible with LiveKit `dev`, `console`, and `start` modes.
* Verified locally through the built-in test harness.

---

## Summary

This project introduces a simple but effective mechanism to handle conversational interruptions in real time.
The `FillerInterruptFilter` allows the agent to distinguish between genuine user intent and meaningless filler speech, improving overall dialogue quality.


