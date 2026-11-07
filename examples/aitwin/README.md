# AiTwin × LiveKit Agents demo

Local harness: **LiveKit** runs STT + LLM + **TTS**. **AiTwin server** aligns PCM → visemes (`/avatar_alignVisemes`). The browser **AiTwin SDK** plays that audio and drives lipsync via `beginSpeech` / `pushSpeechChunk` / `endSpeech`.

```text
Mic / chat (LLM) / Speak text (aitwin.say, no LLM)
  → LiveKit Agents (OpenAI STT + LLM when needed, audio_output=False)
  → OpenAI TTS PCM (or optional HeadTTS)
  → AiTwin POST /avatar_alignVisemes  ← server G2P / alignment
  → aitwin.speech {begin, chunk, end}
  → browser @streamoji/aitwin pushSpeechChunk()
  → AiTwin Web Audio + lipsync
```

Browser never calls `/avatar_ttsWithPoses` — that path is removed from this demo.

No agent audio track — audio bytes travel with server-aligned visemes so clocks stay aligned.

## Prerequisites

- Python 3.10+
- Node.js 18+
- [LiveKit Cloud](https://cloud.livekit.io) credentials
- OpenAI API key (STT + LLM + default TTS)
- AiTwin Client-Id + Client-Secret (dashboard → API Keys) — used by token server **and** agent `alignVisemes`
- Optional: HeadTTS REST server

## Setup

### Backend (this directory)

From the repository root:

```bash
uv sync --all-extras --dev
```

Or from this example alone:

```bash
cd examples/aitwin
cp .env.example .env
# Fill LIVEKIT_*, OPENAI_API_KEY, AITWIN_CLIENT_ID, AITWIN_CLIENT_SECRET
# TOPIC_SPEECH=aitwin.speech is required
uv sync
uv run python agent.py download-files
```

### Frontend

```bash
cd examples/aitwin/frontend
cp .env.example .env
# VITE_TOPIC_SPEECH / VITE_TOPIC_SAY must match backend
npm install
```

## Run (3 terminals)

```bash
# Terminal 1 — token + agent dispatch + AiTwin auth
cd examples/aitwin
uv run python token_server.py
```

```bash
# Terminal 2 — LiveKit Agents worker
cd examples/aitwin
uv run python agent.py dev
```

```bash
# Terminal 3 — AiTwin UI
cd examples/aitwin/frontend
npm run dev
```

Open http://localhost:3000 → **Connect**.

| Mode | Behavior |
|------|----------|
| **Mic** | STT → LLM → TTS → AiTwin alignVisemes → `aitwin.speech` → lipsync |
| **Chat (LLM)** | same as mic, typed via `lk.chat` |
| **Speak text** | `aitwin.say` → agent TTS + alignVisemes → `aitwin.speech` (no LLM, no browser TTS) |

## Env reference

| Where | Vars |
|-------|------|
| `examples/aitwin/.env` | `LIVEKIT_*`, `OPENAI_API_KEY`, `LLM_MODEL`, `STT_MODEL`, `TTS_MODEL`, `TTS_VOICE`, `AITWIN_*`, `AITWIN_ALIGN_LANGUAGE`, `AITWIN_VISEME_ENGINE_ID`, `AGENT_NAME`, `TOPIC_SPEECH`, `TOPIC_SAY`, `TOKEN_SERVER_PORT`, `TOKEN_SERVER_HOST`, `TOKEN_SERVER_SECRET`; optional `HEADTTS_URL` |
| `examples/aitwin/frontend/.env` | `VITE_TOPIC_SPEECH`, `VITE_TOPIC_CHAT`, `VITE_TOPIC_SAY` (must match backend); `VITE_TOKEN_SERVER_SECRET` (must match `TOKEN_SERVER_SECRET`); optional `VITE_AITWIN_*` |

Keep `AITWIN_CLIENT_SECRET` on the backend only. The browser receives a short-lived `client_*` JWT from `POST /token`. The agent mints its own token for `/avatar_alignVisemes`.

**Token endpoint hardening (local demo):** `/token` binds to `127.0.0.1` by default, ignores client-supplied room names (always mints `aitwin-demo-<uuid>`), and when `TOKEN_SERVER_SECRET` is set requires header `X-Token-Server-Secret`. This is not a production auth system — do not expose the token server to the public internet.

## Troubleshooting

| Symptom | Likely cause |
|---------|----------------|
| Token 401 | `TOKEN_SERVER_SECRET` set but frontend missing/mismatched `VITE_TOKEN_SERVER_SECRET` |
| Token 500 | Missing LiveKit env on token server |
| AiTwin auth missing | `AITWIN_CLIENT_ID` / `AITWIN_CLIENT_SECRET` not set |
| `avatar_alignVisemes` fails | Bad auth, wrong `AITWIN_API_BASE`, or API unavailable |
| Agent never joins | Worker not running, or `AGENT_NAME` mismatch |
| Twin silent after chat | Console: `aitwin.speech` / `beginSpeech` / `pushSpeechChunk`; twin ready; topics match |
| Mic unused | Tap **Enable mic** after Connect |

## Decision notes

- **LLM / STT / TTS:** LiveKit agent (OpenAI). No agent audio track.
- **Visemes:** AiTwin server `/avatar_alignVisemes` (not local Python G2P).
- **Transport:** JSON text streams on `aitwin.speech` (`begin` → `chunk` → `end`).
- **Avatar:** `@streamoji/aitwin` in the browser (external speech ingest APIs).
