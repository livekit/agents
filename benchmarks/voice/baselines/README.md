# Baselines

Summaries of baseline runs (`python -m lk_voice_bench.report --json`); the raw results, audio and
logs stay out of git in `benchmarks/voice/results/`.

## `baseline-gemma4`, airline — 2026-11-23

Agent: the repo's starter agent through the LiveKit gateway — `deepgram/flux-general`,
`google/gemma-4-31b-it`, `cartesia/sonic-3` (deviations in `lk_voice_bench/config.py`).
One trial per task.

### tau2 voice, `voice-v1` airline (50 tasks) — `tau2-v1-airline.json`

| | |
|---|---|
| pass^1 | **0.56** corrected (0.54 as run) |
| infra errors | 0 |
| calls discarded for caller hallucination and re-run | 26 |
| agent e2e latency p50 / p90 | 2.93 s / 8.75 s |
| end-of-turn delay p50 | 2.24 s |
| gemma-4 TTFT p50 / TTS TTFB p50 | 0.28 s / 0.15 s |
| agent replies interrupted | 36 % |
| tool calls per call | 7.4 |

Correction: 8 tasks (29 30 32 39 43 44 46 48) overlapped the EVA run and hit the gateway's TTS
concurrency limit (`429 MaxConcurrentGatewayTTS`); re-run alone (`tau2-v1-airline-rerun8.json`,
custom run with voice-v1 settings pinned), 7 scored the same and task 48 went 0 -> 1.

Published `voice-v1` airline pass^1 for reference: Sierra's own LiveKit cascade (nova-3, gpt-4.1,
aura) 48.0; gpt-realtime-2 58.0; gemini-3.1-flash-live 64.0; gpt-live-1 82.0; Pine Voice 92.0.

Deviations from Sierra's setup: caller voices created from tau2's design prompts in our own
ElevenLabs account (the official IDs are Sierra-internal); caller-hallucination review by gpt-5.5
(default claude-opus-4-5).

### EVA-Bench airline (50 scenarios, 49 scored) — `eva-airline.json`

| | |
|---|---|
| EVA-A pass, without speech fidelity | **0.47** |
| EVA-X pass | **0.82** |
| task completion / faithfulness (means) | 0.67 / 0.54 |
| turn taking / conciseness / conversation progression (means) | 0.93 / 0.85 / 0.76 |
| STT WER / key-entity transcription accuracy | 0.079 / 0.956 |
| EVA response speed p50 | 1.46 s |
| agent e2e latency p50 / p90 | 1.38 s / 2.42 s |
| end-of-turn delay p50 | 0.50 s |
| agent replies interrupted | 4.8 % |

Record 1.2.2 did not finish in 3 attempts. One session hit the TTS concurrency limit.

Deviations from EVA's published setup (see `eva/run.sh`): OpenAI Realtime caller
(`gpt-realtime-1.5`) instead of ElevenLabs agents; OpenAI judges only, faithfulness by gpt-5.2
instead of Claude Opus 4.6; no Gemini audio judge, so `agent_speech_fidelity` (an EVA-A pass
condition), `tts_fidelity` and the `user_speech_fidelity` validation check are not run. Published
cascades for reference: Nova-3 + GPT-5.4-mini + Aura-2 at EVA-A 0.40 / EVA-X 0.10; Universal 3.5
Pro + GPT-5.4 + Sonic 3.5 at 0.53 / 0.25.

### Observations

- The same agent sees two very different callers: tau2's (noise, disfluencies, backchannels,
  deliberate interruptions) gives a 2.2 s median end-of-turn delay and 36 % interrupted replies;
  EVA's clean OpenAI Realtime caller 0.5 s and 5 %.
- Many tau2 failures start with a misheard entity (`sophia_silva` -> "sophia silver") that the
  agent retries instead of asking the caller to spell, then transfers.
- One trial per task is noisy: a single task flip moves airline pass^1 by 2 points. Compare
  presets with `--num-trials 3` (tau2) / `--num-trials 3` (EVA) and pass^k.
- The LiveKit gateway's TTS concurrency limit for the project (~20 concurrent sessions here) is
  what caps parallel calls, not the machine.
