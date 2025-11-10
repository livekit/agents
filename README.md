# Filler Interruption Handling (Voice)

This branch adds a interruption classification layer that prevents false cuts during agent speech by ignoring filler-only user utterances (e.g., “uh”, “umm”, “hmm”, “haan”), while still reacting immediately to real commands (e.g., “stop”, “wait”). Core VAD/STT behavior is unchanged.

## What Changed
- **New module**
  - [livekit.agents.voice.interrupt_filter.InterruptionClassifier](cci:2://file:///c:/Users/91811/Desktop/agents/livekit-agents/livekit/agents/voice/interrupt_filter.py:36:0-150:86)
    - Classifies interim STT transcripts into:
      - `ignore_filler`: suppress filler-only speech while agent is speaking
      - `real_interrupt`: contentful or stop-keyword speech interrupts immediately
      - `passive`: agent not speaking; pass through
    - Multi-language support with default and per-language sets.
    - Confidence threshold for short murmurs.
    - Runtime hot-reload of fillers/stop keywords/min confidence.
- **Integration**
  - [AgentActivity.on_interim_transcript(...)](cci:1://file:///c:/Users/91811/Desktop/agents/livekit-agents/livekit/agents/voice/agent_activity.py:1208:4-1283:61) uses the classifier to ignore fillers during TTS and to allow real interruptions.
  - [AgentSession.update_interruption_filter(...)](cci:1://file:///c:/Users/91811/Desktop/agents/livekit-agents/livekit/agents/voice/agent_activity.py:219:4-232:76) exposes a safe, public API to update the classifier at runtime (from UI/RPC/tools).
- **Logging & UX**
  - Structured debug logs show:
    - user vs agent, decision kind/reason, ignored filler tokens.
  - Demo prints a console banner with active config at startup.
- **Demo**
  - [examples/voice_agents/filler_interrupt_demo.py](cci:7://file:///c:/Users/91811/Desktop/agents/examples/voice_agents/filler_interrupt_demo.py:0:0-0:0):
    - Loads `.env` and [config](cci:7://file:///c:/Users/91811/Desktop/agents/config:0:0-0:0).
    - Provides [update_filter](cci:1://file:///c:/Users/91811/Desktop/agents/examples/voice_agents/filler_interrupt_demo.py:21:0-37:40) tool for runtime updates.
- **CI/Compatibility**
  - Python 3.9 f-string fixes (precompute `\n` escapes).
  - Modern typing (`list`, `set`, `dict`, `X | None`).
  - [mypy.ini](cci:7://file:///c:/Users/91811/Desktop/agents/mypy.ini:0:0-0:0) excludes duplicate-named example workers to keep CI green.
  - `ruff` lint/format clean.

## What Works
- **Filler suppression** during agent speech, including low-confidence murmurs.
- **Immediate interruption** when a stop keyword or contentful phrase is detected.
- **Passive mode** when the agent is quiet; all speech is registered normally.
- **Multi-language** filler/stop sets via env and hot-reload (e.g., English + Hindi).
- **Clear logs**: decision, reason, and ignored tokens are visible during audio mode.
- **Demo banner** showing the currently effective configuration.

## Known Issues
- **STT confidence** may not always be provided; falls back to conservative defaults.
- **Language codes** must align with the STT output; otherwise defaults are used.
- **Filler lists** may require tuning for specific locales; override via env or runtime tool.

## Steps to Test
1. **Set environment** (PowerShell examples):
   - `setx AGENTS_IGNORED_FILLERS "uh,umm,um,hmm,haan"`
   - `setx AGENTS_STOP_KEYWORDS "stop,wait,hold,hold on,pause"`
   - `setx AGENTS_MIN_CONFIDENCE "0.6"`
   - Optional per-language:
     - `setx AGENTS_IGNORED_FILLERS_HI "haan,acha,umm"`
     - `setx AGENTS_STOP_KEYWORDS_HI "ruk,bas,thambo"`
   - Restart the shell or ensure your IDE loads `.env` and [config](cci:7://file:///c:/Users/91811/Desktop/agents/config:0:0-0:0).
2. **Run the demo**:
   - `python examples/voice_agents/filler_interrupt_demo.py console`
   - Confirm the startup banner prints current filler/stop sets and min confidence.
3. **Interact**:
   - While agent is speaking, say only fillers (“uh”, “hmm”) → agent continues.
   - While agent is speaking, say “wait” or “stop” → agent interrupts immediately.
   - While agent is quiet, say “umm” → transcript registers normally.
4. **Runtime updates**:
   - Use the [update_filter](cci:1://file:///c:/Users/91811/Desktop/agents/examples/voice_agents/filler_interrupt_demo.py:21:0-37:40) tool (exposed in the demo) to add/remove fillers or change `min_confidence` without restarting.

## Environment Details
- **Python**: 3.9+ (CI uses 3.9)
- **Providers** (for demo; alternatives possible):
  - STT: e.g., `assemblyai/universal-streaming`
  - LLM: e.g., `openai/gpt-4.1-mini`
  - TTS: e.g., `cartesia/sonic-3` (or other supported)
- **Config loading**:
  - `.env` and [config](cci:7://file:///c:/Users/91811/Desktop/agents/config:0:0-0:0) files are both loaded in the demo and tests.
- **Key environment variables**:
  - `AGENTS_IGNORED_FILLERS`
  - `AGENTS_STOP_KEYWORDS`
  - `AGENTS_MIN_CONFIDENCE`
  - `AGENTS_IGNORED_FILLERS_<lang>`
  - `AGENTS_STOP_KEYWORDS_<lang>`
  - LiveKit/API keys for your chosen providers (e.g., `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`, `DEEPGRAM_API_KEY`, `OPENAI_API_KEY`, `ELEVEN_API_KEY`, etc.), as applicable to your setup.

## Developer Notes
- **Formatting & Linting**:
  - `ruff format --check .`
  - `ruff check .`
- **Type checking**:
  - `mypy .`
  - A [mypy.ini](cci:7://file:///c:/Users/91811/Desktop/agents/mypy.ini:0:0-0:0) is included to exclude duplicate example workers:
    - `examples/avatar_agents/(audio_wave|anam)/agent_worker.py`
- **Key files**:
  - [livekit-agents/livekit/agents/voice/interrupt_filter.py](cci:7://file:///c:/Users/91811/Desktop/agents/livekit-agents/livekit/agents/voice/interrupt_filter.py:0:0-0:0)
  - [livekit-agents/livekit/agents/voice/agent_activity.py](cci:7://file:///c:/Users/91811/Desktop/agents/livekit-agents/livekit/agents/voice/agent_activity.py:0:0-0:0)
  - `livekit-agents/livekit/agents/voice/agent_session.py`
  - [examples/voice_agents/filler_interrupt_demo.py](cci:7://file:///c:/Users/91811/Desktop/agents/examples/voice_agents/filler_interrupt_demo.py:0:0-0:0)
  - [examples/voice_agents/session_close_callback.py](cci:7://file:///c:/Users/91811/Desktop/agents/examples/voice_agents/session_close_callback.py:0:0-0:0)
  - [tests/conftest.py](cci:7://file:///c:/Users/91811/Desktop/agents/tests/conftest.py:0:0-0:0)
  - [mypy.ini](cci:7://file:///c:/Users/91811/Desktop/agents/mypy.ini:0:0-0:0)
  - [config](cci:7://file:///c:/Users/91811/Desktop/agents/config:0:0-0:0)
