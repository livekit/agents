# Intelligent Voice Interruption Handler

**Branch:** `feature/livekit-interrupt-handler-Aayush_Rawat`

This project implements an **AI Interruption Handler** for the LiveKit Agents framework to solve the problem of filler words (like "uh" and "umm") causing false interruptions during active agent speech. The goal is to create a more natural, seamless dialogue experience by ensuring the agent only pauses gracefully for genuine user input or explicit commands, while ignoring minor hesitations.

***

## What Changed: Solution Overview 💻

The core solution introduces an **Extension Layer** that processes ASR transcription events before they trigger the default VAD interruption logic. This maintains the integrity of the base LiveKit VAD algorithm, as required by the assessment.

### New Modules Added:

1.  **`config.py`**: A dedicated configuration module exposing lists of words, ensuring the system is scalable and language-agnostic:
    * `IGNORED_WORDS`: Filler words (e.g., "uh," "hmm") to be suppressed when the agent is actively speaking.
    * `INTERRUPTION_COMMANDS`: Keywords (e.g., "stop," "wait") that must **always** cause an immediate interruption.
2.  **`AI-InterruptHandler.py`**: A new class that holds the core decision logic:
    * It checks the agent's real-time speaking status (`session.tts_stream.is_active`).
    * It analyzes incoming ASR transcription text and its confidence score against the configurable lists to decide whether to stop the agent's streams.
3.  **`my-ai-agent.py`**: A sample agent that integrates the handler by subscribing to the `user_speech_transcribed` event. When a valid, non-filler interruption is detected, the script manually calls `session.tts_stream.stop()`, `session.llm_session.stop()`, and `session.queue_user_turn()` to gracefully execute the pause and transfer control back to the user.

***

## What Works: Verified Scenarios ✅

The following scenarios were successfully validated in `console` mode, confirming the interruption handler logic:

| Scenario | User Input | Agent Speaking? | Handler Decision | Observed Result |
| :--- | :--- | :--- | :--- | :--- |
| **Filler Ignored** | "Umm, I think..." | **Yes** | **Ignore.** (Filler-only text) | Agent continues speaking without interruption. |
| **Command Interruption** | "Stop, wait a second." | **Yes** | **Interrupt.** (Contains `INTERRUPTION_COMMAND`) | Agent immediately stops TTS and processes the command. |
| **Meaningful Speech** | "Wait, what about the price?" | **Yes** | **Interrupt.** (Contains `INTERRUPTION_COMMAND` and new speech) | Agent immediately stops and processes the new query. |
| **Background Murmur** | Low-volume noise (low ASR confidence) | **Yes** | **Ignore.** (Confidence below `0.6` threshold) | Agent continues speaking, demonstrating noise immunity. |
| **Filler When Quiet** | "Uh, okay then." | **No** | **Register Turn.** | Agent processes the speech as a normal user turn. |

***

## Steps to Test 🧪

Please follow these steps to run and verify the Intelligent Interruption Handler:

1.  **Clone and Checkout:** Ensure you have cloned the repository and are on the `feature/livekit-interrupt-handler-Aayush_Rawat` branch.
2.  **Install Dependencies:** Run the following command in your repository's root directory:
    ```bash
    pip install "livekit-agents[openai,deepgram,elevenlabs,silero]~=1.0"
    ```
3.  **Set Environment Variables:** Set all required environment variables using your own credentials (see **Environment Details** section below).
4.  **Run the Agent:** Execute the custom agent script in console mode:
    ```bash
    python examples/my_intelligent_agent.py console
    ```
5.  **Test Scenarios:** Speak into your microphone and test the scenarios above, monitoring the terminal output for the `**LOG: ...**` messages to confirm the correct decision is made.

***

## Known Issues ⚠️

* **ASR Confidence Reliance:** The detection relies on the confidence score provided by the STT provider (Deepgram). Very short, meaningful utterances delivered with poor voice quality might be mistakenly filtered if their confidence score drops below the `LOW_CONFIDENCE_THRESHOLD` defined in `config.py`.
* **Latency:** The filtering mechanism introduces negligible latency, but overall responsiveness is still constrained by the response times of the external LLM/TTS APIs.

***

## Environment Details 🔑

The following environment variables **must be set** in your terminal session before executing the agent. The system is designed to load these secrets automatically.

| Component | Environment Variable | Purpose |
| :--- | :--- | :--- |
| **LiveKit Server** | `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` | Required for connection to the media server. |
| **LLM (OpenAI)** | `OPENAI_API_KEY` | Provides conversational intelligence. |
| **STT (Deepgram)** | `DEEPGRAM_API_KEY` | Provides real-time transcription with confidence scores. |
| **TTS (Eleven Labs)** | `ELEVEN_API_KEY` | Provides high-quality voice output. |

### Configurable Parameters

The lists of words can be updated in `examples/config.py`:

```python
# Defined in examples/config.py
IGNORED_WORDS = ['uh', 'umm', 'hmm', 'haan', 'like', 'you know']
INTERRUPTION_COMMANDS = ['wait', 'stop', 'hold on', 'excuse me']
