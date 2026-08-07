# LiveKit xAI Realtime plugin issues — Voice Agent API

Audit of `livekit-plugins-xai` Realtime (`wss://api.x.ai/v1/realtime`) against xAI Voice Agent docs and the Pipecat findings in the local audit playbook.

Scope: Voice Agent path (`xai.realtime.RealtimeModel`), not cascaded STT/TTS.

---

## Summary

| # | Issue | Severity | Status |
| -: | :---- | :---- | :---- |
| 1 | `session.created` unimplemented / wrong bootstrap | Blocker (Pipecat) | **N/A / OK** — unknown events do not abort; `session.update` queued at session init (OpenAI pattern) |
| 2 | Silent audio drop until conversation setup | Blocker (Pipecat) | **N/A / OK** — no conversation-seed gate; `push_audio` appends once WS send loop runs |
| 3 | Interrupt skips `response.cancel` under server VAD | High (Pipecat) | **OK** — base `interrupt()` always cancels when a generation is active; does not clear input buffer |
| 4 | Default model still `grok-voice-think-fast-1.0` | Medium | **Fixed** — default is now `grok-voice-latest` (resolves to think-fast-2.0) |
| 5 | Local PortAudio dependency | Medium (Pipecat) | **N/A** — LiveKit room / frame path is headless |
| 6 | Truncate missing on interrupt | Medium (Pipecat) | **OK** — activity calls `truncate()` after partial playout; base supports `conversation.item.truncate` |
| 7 | Process overhead vs raw WS | Low (Pipecat) | Accepted / out of scope |
| 8 | `PipelineRunner` deprecation | Low (Pipecat) | **N/A** — LiveKit CLI |
| 9 | Event / session field drift | Medium | **Partially fixed** — see below |
| 10 | Empty default input transcription (no `grok-transcribe`) | Medium | **Fixed** — default `AudioTranscription(model="grok-transcribe")` |
| 11 | No `force_message` / `say()` | Medium | **Fixed** — `supports_say=True` + `RealtimeSession.say()` via `force_message` |
| 12 | Live captions `…transcription.updated` ignored | Low–Medium | **Fixed** — emitted as non-final `input_audio_transcription_completed` |
| 13 | Tests assumed xAI lacks `conversation.item.delete` | Low | **Fixed** — history-replace test now includes xAI; delete confirmed on prod |
| 14 | Audio harness pushed multi-second silence as one frame | Low (harness) | **Fixed** — stream trailing silence in 20 ms frames for server VAD |

---

## Architecture map

1. `xai.realtime.RealtimeModel` subclasses OpenAI realtime, points at `wss://api.x.ai/v1/realtime`, auth via `XAI_API_KEY`.
2. Session construct queues `session.update` immediately (not gated on `session.created`).
3. `_wrap_session_update` lifts `voice` / `turn_detection` to top-level session fields for xAI.
4. Audio: LiveKit frames → `push_audio` → base64 `input_audio_buffer.append` at 24 kHz.
5. Barge-in: `input_audio_buffer.speech_started` → activity → `interrupt()` → `response.cancel`; truncate via activity after playout.
6. xAI extras: provider tools (`web_search` / `x_search` / `file_search`), transcript hold until agent speaks, abandoned-response discard, `force_message` say().

---

## Event coverage (high level)

| Area | Status |
| ---- | ------ |
| Bootstrap `session.created` / `conversation.created` | Tolerated (no abort); debug log on `session.created` |
| `session.update` / `session.updated` | Supported (xAI field relocation) |
| Audio append / commit / clear | Inherited from OpenAI base |
| `response.create` / `response.cancel` | Supported |
| `conversation.item.create` / `delete` / `truncate` | Supported |
| `force_message` | Supported via `say()` |
| Input transcription completed + **updated** | Supported |
| `input_audio_buffer.timeout_triggered` | Logged; server drives proactive turn |
| `reasoning` / `idle_timeout_ms` (on `ServerVad`) | Pass-through via constructor / turn_detection |
| MCP lifecycle / SIP DTMF / binary transport | Not first-class in OpenAI base either — deferred |

---

## Live verification (2026-08-07)

With `XAI_API_KEY` against prod:

- `grok-voice-latest` → `session.model = grok-voice-think-fast-2.0`
- `session.update` accepts `reasoning.effort=none`, `idle_timeout_ms`, `grok-transcribe`
- `response.cancel` → `response.done` status `cancelled`
- `force_message` produces audio without `response.create`
- `conversation.item.delete` → `conversation.item.deleted`
- `uv run pytest tests/test_realtime/test_realtime.py -k xai` → **16 passed**
- Unit: `tests/test_realtime/test_xai_realtime_model.py` → **20 passed**

---

## Recommended follow-ups

1. First-class MCP tool config in the xAI plugin once the OpenAI realtime base exposes it.
2. Optional binary audio transport if/when LiveKit agents need it.
3. Long-idle soak for `timeout_triggered` proactive turns.
4. `RealtimeReasoning(effort="none")` needs `model_construct` today (OpenAI SDK enum lacks `"none"`).
