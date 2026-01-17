# Intelligent Interruption Handling for LiveKit Agent 🎧

This project implements a context-aware interruption handling system for LiveKit Voice Agents. It solves the common problem where agents abruptly stop speaking when a user gives a passive acknowledgment (like "yeah", "mhmm", "okay").

## 🚀 How to Run

1.  **Navigate to the agent directory:**
    ```bash
    cd examples/voice_agents
    ```

2.  **Start the Agent:**
    ```bash
    python basic_agent.py dev
    ```

3.  **Connect a client** (e.g., LiveKit Playground) and start speaking!

## 🧠 How It Works (The Logic)

The solution uses a **Runtime Monkeypatch** to inject intelligent filtering logic directly into the LiveKit framework's event loop. This ensures that checks happen *before* the audio stream is paused.

### 1. The Core Problem
Standard VAD (Voice Activity Detection) is binary: if it hears sound, it pauses the agent immediately. This causes awkward 1-2 second silences even for simple backchannel words like "mhmm".

### 2. The Solution: "Guard at the Gate"
We injected a patch into `AgentActivity` that intercepts the "Pause Audio" signal.
*   **Intercept:** When VAD detects speech, our patch runs first.
*   **Analyze:** It extracts the *new* words spoken (the "Delta").
*   **Decide:**
    *   If the words are **Backchannel** (e.g., "yeah", "ok"): **BLOCK THE PAUSE**. The agent keeps speaking smoothly.
    *   If the words are **Commands** (e.g., "stop", "wait") or **Content**: **ALLOW THE PAUSE**. The agent stops immediately.

### 3. Delta Detection
To avoid confusion with previous sentences, the system calculates the "Delta":
*   **Transcript:** `"Tell me a story... mhmm"`
*   **Last Sent:** `"Tell me a story..."`
*   **Delta:** `"mhmm"` ✅ -> Identified as backchannel.

## ⚙️ Configuration

The agent now uses an external configuration file for easy editing. You do not need to touch the code.

### `filter_config.json`
This file is located in the same directory as the agent. You can modify the lists of words directly.

```json
{
    "backchannel_words": [
        "yeah", "ok", "okay", "mhmm", "uh-huh", "right", "sure"
    ],
    "command_words": [
        "stop", "wait", "hold", "pause", "no", "nope", "don't"
    ]
}
```

*   **`backchannel_words`**: Words that will be **IGNORED** by the interruption filter (agent keeps speaking).
*   **`command_words`**: Words that will **ALWAYS** interrupt the agent.

## 🧪 Testing

1.  **Backchannel Test:** Ask the agent to tell a long story. While it speaks, say **"mhmm"** or **"yeah"**.
    *   *Result:* Agent continues speaking without any pause.
2.  **Interruption Test:** While the agent speaks, say **"Wait, stop."**
    *   *Result:* Agent stops immediately.
3.  **Mixed Input:** Say **"Yeah but wait."**
    *   *Result:* Agent stops (detects "wait" command).
