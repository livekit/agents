# 🎯 LiveKit Filler-Aware Interrupt Handler  
**Author:** Khush Gupta  
**Branch:** `feature/livekit-interrupt-handler-khush_gupta`  
**File Modified:**  
`C:\Users\jayag\agents\livekit-agents\livekit\agents\voice\agent_activity.py`

---

## 🧩 1. What Changed

### 🔹 Objective
This implementation enhances LiveKit’s **voice agent interruption logic** to make conversations more natural and human-like.  

It ensures that:
- Filler or hesitation words (e.g., *“uh”*, *“umm”*, *“haan”*) **do not interrupt** the agent’s speech.  
- Only **meaningful user responses** trigger an interruption.  
- Filler detection works for **multi-language speech (English, Hindi, Hinglish)**.  
- Filler lists can be **updated dynamically at runtime**.  

---

### 🔹 New Components

#### 🧠 `FillerManager` (New Class)
Added to manage ignored fillers dynamically.

```python
class FillerManager:
    def get_fillers_for(lang)
    def add_filler(word, lang="default")
    def remove_filler(word, lang="default")
    def get_min_conf()
```

**Features:**
- Loads initial fillers from:
  ```bash
  LIVEKIT_IGNORED_FILLERS="uh,umm,hmm,haan"
  LIVEKIT_FILLER_CONFIDENCE="0.6"
  ```
- Maintains per-language filler sets (`default`, `en`, `hi`, `hinglish`)
- Allows live modification of filler lists through the API

#### 🌍 Global Instance
```python
FILLERS = FillerManager()
```
Shared across the session lifecycle.

---

### 🔹 Modified Function: `on_interim_transcript(...)`

**File:** `livekit/agents/voice/agent_activity.py`  
Implements the filler-aware interruption logic.

#### 🧩 Previous Behavior:
Any interim speech detected from the user interrupted the agent — even meaningless utterances or noise.

#### 🧠 New Behavior:
1. Extracts transcript, confidence, and language from the STT result.  
2. Tokenizes text using regex for normalized word matching.  
3. Fetches all filler tokens for the detected language.  
4. Determines:
   - If text = **filler-only** → ignore  
   - If confidence < threshold → ignore  
   - Else → interrupt TTS immediately  
5. Logs both “ignored” and “valid interruption” events separately.  

#### 🔤 Multi-Language Handling:
Merges filler sets:
```
default ∪ language-specific ∪ hinglish
```
✅ “umm haan okay” → detected as filler mix and ignored.  
✅ “umm okay stop” → triggers valid interruption.  

#### 🔄 Retains Original Logic:
Keeps LiveKit’s `false_interruption_timeout` resume feature intact.

---

### 🔹 Bonus Feature: Runtime Filler Update
You can dynamically update fillers using a `function_tool`:
```python
@function_tool
async def add_filler_word(self, context, word, lang="default"):
    from livekit.agents.voice.agent_activity import FILLERS
    FILLERS.add_filler(word, lang)
    return f"Added filler '{word}' for {lang}"
```
→ The agent learns new fillers during runtime (e.g., add *“arey”* as Hindi filler).

---

## ✅ 2. What Works (Tested Features)

| Scenario | Expected Behavior | Verified |
|-----------|------------------|-----------|
| User says “umm” while agent speaks | Ignored | ✅ |
| User says “uh okay stop” | Agent interrupts immediately | ✅ |
| Background murmur with low confidence | Ignored | ✅ |
| Agent silent + “umm” | Treated as input | ✅ |
| Mixed-language “umm haan okay” | Ignored | ✅ |
| Add filler “arey” at runtime | Ignored dynamically | ✅ |
| Separate logs for filler vs real speech | ✅ | ✅ |

---

## ⚠️ 3. Known Issues

| Issue | Description | Impact | Mitigation |
|--------|--------------|---------|-------------|
| STT language tag missing | Some STT providers omit language | Falls back to default list | Acceptable |
| Tokenization edge case | “um-m” or slurred audio may bypass filler detection | Rare | Regex normalization |
| High-confidence noise | Can occasionally interrupt | Low | Tune `LIVEKIT_FILLER_CONFIDENCE` |

✅ No runtime crashes, timeouts, or deadlocks observed.

---

## 🧪 4. Steps to Test

### 🧰 Setup (WSL)
```bash
git clone <your-fork-url>
cd livekit-agents
git checkout -b feature/livekit-interrupt-handler-khush_gupta
```

Install dependencies:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv sync
```

---

### ⚙️ Environment Variables
```bash
export LIVEKIT_URL="wss://your-livekit-host"
export LIVEKIT_API_KEY="your_api_key"
export LIVEKIT_API_SECRET="your_api_secret"
export OPENAI_API_KEY="sk-..."
export LIVEKIT_IGNORED_FILLERS="uh,umm,hmm,haan,accha"
export LIVEKIT_FILLER_CONFIDENCE="0.6"
```

---

### ▶️ Run Agent
Use the lightweight example (no Silero plugin) for WSL:
```bash
uv run python examples/voice_agents/basic_agent_nosilero.py start
```

---

### 🎤 Test Cases

| Input | Expected Output in Logs |
|--------|--------------------------|
| “umm” | `[ignored non-meaningful speech while agent speaking]` |
| “uh okay stop” | `[valid user interruption]` |
| “haan” | `[ignored non-meaningful speech]` |
| Add filler “arey” | `[added filler 'arey' for lang 'hi']` |

All behaviors validated via terminal logs.

---

## 💻 5. Environment Details

| Component | Version / Info |
|------------|----------------|
| **Python** | 3.10+ (WSL Ubuntu) |
| **LiveKit SDK** | 1.2.18 |
| **Package Manager** | uv 0.9.x |
| **Optional** | torch (CPU build for Silero) |
| **OS** | Windows 11 (WSL2 backend) |

---

**End of File — README.md**
