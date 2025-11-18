# 🗣️ Conversational AI Agent with Static Backchannel Filtering

## Project Overview

This project implements an advanced conversational AI agent using an **Intelligent Interrupt Handler**. The core feature is the use of a predefined, static list of **ignored words (backchannels)** to filter out common conversational fillers (like "yeah," "uh-huh") and prevent the agent from being unnecessarily interrupted. This significantly improves the natural flow of the full-duplex conversation.

## Key Components

### 1\. The Agent (`my_ai_agent.py`)

This is the main entry point, containing the session setup and event handlers for managing the conversation flow.

### 2\. Intelligent Interrupt Handler (`IntelligentInterruptHandler.py`)

This class contains the core logic for distinguishing between a listener backchannel and a genuine interruption command. It checks for:

  * **Static Ignored Words:** Exact matches to `IGNORED_WORDS` are filtered.
  * **Explicit Commands:** Words like "STOP" are immediately processed if confidence is high.
  * **Confidence Threshold:** Filters out low-confidence, noisy transcriptions.

## Feature Branch Details: Static Interruption Filtering

### What Changed: Overview of new modules, params, and logic added.

  * **New Module:** Introduced `IntelligentInterruptHandler.py` to encapsulate and centralize all interruption decision logic, separating it from the main agent loop.
  * **New Parameter:** The `IGNORED_WORDS` list was introduced in `config.py` to hold the static, pre-defined set of backchannels (e.g., `mhm`, `i see`).
  * **Core Logic:** The `user_speech_transcribed` event handler now delegates the interruption decision to an instance of `IntelligentInterruptHandler`, which applies the static filter and confidence checks.

### What Works: Features verified through manual or automated testing.

  * **Backchannel Filtering:** The agent successfully ignores single-word or short phrases (e.g., **"mhm," "right"**) found in the static `IGNORED_WORDS` list, allowing it to complete its current speech segment without disruption.
  * **Explicit Interruption:** Explicit commands like **"STOP"** or **"WAIT"** are immediately recognized and halt the agent's speech, provided the transcription confidence meets the required threshold.
  * **Full-Duplex Flow:** The agent maintains responsiveness, ensuring low latency for STT, LLM, and TTS services.

### Known Issues: Any edge cases or instability observed.

  * **Static Limitation:** The word list is **static** and cannot be updated dynamically during a session. A restart is required to add or remove ignored words.
  * **Partial Match Edge Case:** The current logic only ignores the transcription if it is an *exact* match for a word in `IGNORED_WORDS`. A complex phrase starting with a filler (e.g., "Yeah, can you pause for a second") is currently treated as a full interruption.
  * **Confidence Instability:** Highly mumbled or low-confidence filler words sometimes slip through the confidence threshold, leading to minor, unwarranted interruptions.

### Steps to Test: How to start the agent and verify filler vs. real speech handling.

1.  **Start the agent** using the instructions in the "Running the Agent" section below.
2.  **Test 1 (Ignored Backchannel):** While the agent is speaking a long response, say: **"Mhm"** or **"Got it"**.
      * *Expected Result:* The agent must continue speaking without interruption.
3.  **Test 2 (Valid Interruption):** While the agent is speaking, say: **"Stop talking now"**.
      * *Expected Result:* The agent should immediately halt its speech and acknowledge the interruption.
4.  **Test 3 (Unlisted Filler):** While the agent is speaking, say: **"Like"** (assuming "like" is *not* in the static list).
      * *Expected Result:* The agent should halt, demonstrating the need for new fillers to be added to the static list.

### Environment Details: Python version, dependencies, and config instructions.

  * **Python Version:** Python 3.8+
  * **Dependencies:** All required packages are listed in `requirements.txt`.
  * **Required External APIs:**
      * **Speech-to-Text (STT): Deepgram**
      * **Text-to-Speech (TTS): LiveKit In-Built TTS**
      * **Large Language Model (LLM): Grok API**

-----

## Configuration & Static Filtering

The list of words and phrases the agent will ignore is defined statically in `config.py`.

### `config.py` (Example)

```python
# List of words/phrases the agent will ignore to prevent unnecessary interruption.
IGNORED_WORDS = [
    "yeah",
    "uh-huh",
    "mhm",
    "right",
    "i see",
    "okay",
    "got it",
    "wow",
    "go on",
]