# Intelligent Interruption Handler Submission
## Aditya Singh 23/SE/012
## Overview
This submission implements a **context-aware interruption logic** for LiveKit agents using **Regex-based Normalization**.
The agent ignores backchanneling words (e.g., "Yeah", "Ok", "Hmm", "Exactly") while speaking, but interrupts immediately for meaningful commands (e.g., "Stop").

## How It Works

### 1. Library Modification (`livekit/agents/voice/agent_activity.py`)
*   **VAD Interruption Disabled**: The agent no longer interrupts solely based on audio energy. This prevents false starts.
*   **STT Interruption Guarded**: The agent waits for the transcript and checks it against the `InterruptionPolicy` before stopping.

### 2. Interruption Policy (`livekit/agents/voice/interruption_policy.py`)
The policy uses robust logic to classify transcripts:

*   **Normalization**: Uses Regex to:
    *   Collapse repeated characters (e.g., "yeeaah" -> "yeah", "mmmmm" -> "mm").
    *   Remove punctuation and special characters.
    *   Collapse whitespace.
*   **Tokenization**: Splits the transcript into words.
*   **Classification**:
    *   **Ignore**: If the *entire* transcript consists only of backchannel words (e.g., "yeah ok").
    *   **Interrupt**: If *any* meaningful word is found (e.g., "yeah stop").

### 3. Backchannel Lists
The policy includes a comprehensive list of ignorable words:
*   **Standard**: "yeah", "ok", "hmm", "uh-huh", "right", "sure"
*   **Agreement**: "exactly", "precisely", "indeed", "i agree", "that's true"
*   **Colloquial**: "cool", "nice", "wow", "no way", "you bet"


## Video Link
Check out the [VideoLink](https://drive.google.com/file/d/1ph9dpD9qrptskoNwIzyoJEz4C67szIDz/view?usp=sharing) for more information.
## Setup & Run

1.  **Install Dependencies**:
    ```bash
    pip install -r examples/voice_agents/requirements.txt
    ```
    *(Note: The environment must use the local `livekit-agents` package. If running from source, ensure `PYTHONPATH` is set or install via `pip install -e .`)*

2.  **Run the Agent**:
    ```bash
    python examples/voice_agents/interrupt_handler_agent.py dev
    ```

3.  **Test**:
    *   Ask the agent to "tell a long story".
    *   Say "Yeah exactly" or "Mmhmmm" -> **Agent continues**.
    *   Say "Stop" -> **Agent stops**.
