# Voice-agent benchmarks

LiveKit Agents as the agent under test in third-party voice-agent benchmarks, so SDK
configurations — and, next, thinker/talker delegation patterns — can be compared on the same
callers, tools and graders.

| Benchmark | Status | What it measures |
|---|---|---|
| [tau2-bench voice](https://github.com/sierra-research/tau2-bench) (τ-Voice, `voice-v1`) | wired up | task success (final DB state + required info said) on airline / retail / telecom, plus turn-taking metrics |
| [EVA-Bench](https://github.com/ServiceNow/eva) | wired up, not yet run (needs caller + judge credentials) | task accuracy (EVA-A) and conversational experience (EVA-X) on airline / ITSM / medical HR |

## How the agent is run

The agent is an ordinary `AgentSession` + `Agent`, run in-process with no room
([`lk_voice_bench/session.py`](lk_voice_bench/session.py)):

- **Microphone** (`CallerAudioInput`): the benchmark's caller audio, pushed in 20 ms frames.
- **Speaker** (`SpeakerOutput`): a virtual loudspeaker that plays the agent's audio out one 20 ms
  frame at a time, on the benchmark's clock or its own real-time one. Playback position, pause
  (false-interruption handling) and `clear_buffer` (barge-in) behave as on a telephony sink, and
  what it played is exactly what the benchmark receives.
- **Transcript**: LiveKit's `TranscriptSynchronizer` sits between the session and the speaker, as
  in `RoomIO`, so the agent's words reach the benchmark as they are spoken, and an interrupted turn
  is cut at the word reached (TTS word timestamps).
- **Tools**: each benchmark tool becomes a raw-schema `function_tool` whose body hands the call to
  the benchmark's environment and waits for its result.

Everything else about the agent comes from a **preset** in
[`lk_voice_bench/config.py`](lk_voice_bench/config.py):

| Preset | STT | LLM | TTS | Notes |
|---|---|---|---|---|
| `baseline-gemma4` | `deepgram/flux-general` | `google/gemma-4-31b-it` | `cartesia/sonic-3` | the repo's starter agent (`examples/voice_agents/basic_agent.py`), all via the LiveKit gateway; deviations listed in `config.py` |

## tau2 voice

tau2 drives an agent through a `DiscreteTimeAdapter`: every 200 ms tick it hands over the
caller's 8 kHz μ-law audio and takes back the agent's audio, transcript and tool calls.
[`lk_voice_bench/tau2_adapter.py`](lk_voice_bench/tau2_adapter.py) implements one on top of the
session above, paced in wall-clock time (as tau2's hosted-agent providers are), and
[`tau2/register-livekit-agents.patch`](tau2/register-livekit-agents.patch) registers it as the
`livekit_agents` provider on a pinned tau2 commit. `--audio-native-model` names the preset.

Setup (once; clones tau2 into `.vendor/` and builds `.venv-tau2/`):

```bash
benchmarks/voice/tau2/setup.sh
```

tau2's simulated caller speaks with ElevenLabs voices you create in your own account (the shipped
IDs are Sierra-internal). Create the two English ones and put the printed IDs in
`benchmarks/voice/.env.tau2.local` (git-ignored):

```bash
cd benchmarks/voice/.vendor/tau2-bench && ELEVENLABS_API_KEY=$ELEVEN_API_KEY ../../.venv-tau2/bin/python -m tau2.voice.scripts.setup_voices --language en
```

Keys: `LIVEKIT_URL` / `LIVEKIT_API_KEY` / `LIVEKIT_API_SECRET` (the agent, via the gateway),
`OPENAI_API_KEY` (the caller and the communicate judge, gpt-5.5), `ELEVEN_API_KEY` or
`ELEVENLABS_API_KEY` (the caller's voice).

Run:

```bash
# a few tasks: a custom run with voice-v1's caller, judge and limits pinned
benchmarks/voice/tau2/run.sh baseline-gemma4 smoke-retail --domain retail --num-tasks 2

# the released voice-v1 frame (airline 50, retail 114, telecom 114), or one domain of it
benchmarks/voice/tau2/run.sh baseline-gemma4 v1-airline --domain airline --max-concurrency 5
```

Results land in `benchmarks/voice/results/tau2/<run-name>/` (`results.json`, per-simulation
JSON) with the agent-side session logs (turns, states, tool calls, per-turn latency metrics) in
`agent_logs/`. View them with tau2's tools from the vendored checkout:

```bash
cd benchmarks/voice/.vendor/tau2-bench && ../../.venv-tau2/bin/tau2 view --file ../../results/tau2/smoke-retail
```

Published `voice-v1` pass^1 for reference (retail / airline / telecom): Sierra's own LiveKit
cascade (nova-3 + gpt-4.1 + aura) 28.9 / 48.0 / 16.7; gpt-realtime-2 47.4 / 58.0 / 21.9;
gpt-live-1 78.9 / 82.0 / 84.2. Sierra's `livekit` provider is a hand-built STT→LLM→TTS loop on
LiveKit plugins, not `AgentSession`, so it does not measure this SDK.

Summarize one or more runs side by side (pass^k, terminations, calls tau2 discarded for caller
hallucinations, and the agent's own latency / interruption / tool stats):

```bash
cd benchmarks/voice && ../../.venv/bin/python -m lk_voice_bench.report results/tau2/v1-airline results/tau2/<other-run>
```

## EVA-Bench

EVA pins litellm to a version that needs `openai<3`, so it cannot share a venv with
livekit-agents. The agent therefore runs as its own websocket service,
[`lk_voice_bench/ws_agent.py`](lk_voice_bench/ws_agent.py) (one `BenchAgent` per call, audio sent as
it plays), and EVA gets a thin `livekit` framework server,
[`eva/livekit_server.py`](eva/livekit_server.py), that relays the caller's Twilio stream, runs
tools against EVA's scenario database and writes EVA's audit and framework logs, like EVA's own
Cartesia bridge to a hosted agent. It is scored as a cascade (the text it reports is what it
spoke).

```bash
benchmarks/voice/eva/setup.sh                                   # .vendor/eva + .venv-eva, pinned
cd benchmarks/voice && ../../.venv/bin/python -m lk_voice_bench.ws_agent --port 8765 --log-dir results/eva/<run>/agent_logs
```

EVA run config: `framework: livekit`, `model: {s2s: livekit, s2s_params: {model: <preset>}}`.
Still needed before a scored run: caller agents (`EVA_EN_USER_F` / `EVA_EN_USER_M`, ElevenLabs
agents per EVA's `docs/experiment_setup.md`, or its OpenAI Realtime caller), and judge access —
text judges can be pointed at OpenAI with `JUDGE_MODEL`, but the audio judge behind
`agent_speech_fidelity` (part of EVA-A pass) needs Gemini on Vertex.

## Live checks

Not unit tests: they call the LiveKit gateway.

```bash
uv run python benchmarks/voice/tests/live_smoke.py                          # BenchAgent alone, with a barge-in
benchmarks/voice/.venv-tau2/bin/python benchmarks/voice/tests/tau2_adapter_smoke.py  # tau2 tick protocol, no paid caller
cd benchmarks/voice && PYTHONPATH=. ../../.venv/bin/python tests/ws_agent_smoke.py  # ws_agent (start it first)
```
