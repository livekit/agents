# AMD audio scenarios

Fixed outbound call scripts for [PR #6202](https://github.com/livekit/agents/pull/6202).
The call paths come from the [Core Detection Flow Design](https://app.notion.com/p/3d43c4901a4281b7885bdc8925a604f9).
The suite targets the PR's client-side STT/LLM/TTS implementation.

This is a custom room runner. It does not run `lk agent simulate`.

The caller and scripted callee join a new LiveKit room. No SIP participant is
created and no phone number is dialed. The caller uses the appointment-confirmation
instructions and model defaults from `examples/telephony/amd.py`.

The callee plays fixed TTS audio, with separate machine and human voices on the
same audio track. Audio clips are generated before the call and cached. The
callee does not use an LLM to choose words or a path. The scenario labels and
expected stages are never sent to the caller or AMD model.

## Compatibility with `lk agent simulate audio`

Checked on 2026-09-11 with LiveKit CLI 2.18.6. The
[CLI supports audio simulations and scenario YAML files](https://docs.livekit.io/reference/developer-tools/livekit-cli/agent/#simulate).
Its scenario fields provide instructions, reply expectations, tags, and userdata.
They do not define fixed audio clips, per-step voices, or playback triggers.

The simulation service generates the callee's replies with an LLM and selects one
voice for the session. A scripted line in its instructions is a prompt, not a
fixed playback operation. There is no DTMF-receipt trigger to advance a menu or
explicit timeline for pauses and human pickup during voicemail.

A live CLI audio probe also failed to initiate an outbound conversation. Despite
instructions to speak first, the simulator sent no opening line and disconnected
after about 31 seconds. AMD completed as `uncertain` with zero turns and reason
`participant_disconnected`. The probe used a 40-second AMD idle timeout so AMD
would not end the call before the simulator's inactivity deadline.

`on_simulation_end` can add agent-side checks through `SimulationContext.fail()`.
That can preserve an AMD trajectory verdict, but it cannot supply the missing
callee playback controls. Keep this runner until the service can execute those
controls. Converting the scripts to instructions alone would change what the
suite verifies.

## Run

Use the repository's development environment. Configure `LIVEKIT_URL`,
`LIVEKIT_API_KEY`, and `LIVEKIT_API_SECRET` in the environment or `.env`.
The project needs access to LiveKit Inference and permission to create rooms.

```sh
# List scenarios without making network requests.
uv run python -m examples.telephony.amd_simulation --list

# Run selected scenarios, serially.
uv run python -m examples.telephony.amd_simulation \
  --scenario direct-human \
  --scenario screening-human \
  --scenario ivr-human \
  --output /tmp/amd-selected

# Run the complete matrix.
uv run python -m examples.telephony.amd_simulation --all --output /tmp/amd-all

# Check the trajectory grader without providers or room connections.
uv run pytest tests/test_amd_simulation.py --unit -q
```

The output directory must be new. Exit status is zero only when every selected
scenario passes. A run still uses room and model resources, including an LLM
judge. Scenarios run serially. Voicemail idle cases deliberately wait for the
PR's full 60-second deadline.

`--llm`, `--stt`, `--tts`, and `--judge` select models. `--machine-voice` and
`--human-voice` select distinct callee voices. `--caller-voice` selects the caller's
voice. Voice IDs must match the selected
TTS provider. The defaults use Cartesia Sonic 3. `--cache` sets the audio cache
directory; preserve that directory to reuse the same synthesized clips.

## Script contract

`scenarios.py` contains the fixed lines, voice role, pauses, expected stages,
required reply meaning, and DTMF digits. Each required response advances the
script after speech finishes or the callee receives the expected number of DTMF
digits. Grading checks the actual digits. It never substitutes a successful
tool call for receipt at the callee.
DTMF-only steps require no spoken reply.

A step can advance after reply audio starts to schedule a human interruption.
Steps with no expected reply use an explicit observation period. Delayed menus
use an explicit pause. None of these scripts chooses its next line from AMD's
predicted category.

The call has one shared allowance of two extra uncertain turns and five seconds
of uncertainty or explicit wait time. Stage changes do not reset either allowance.
The timer starts when a public prediction is unexpectedly `uncertain` or has
`should_wait=True`. A concrete, non-waiting prediction or AMD completion ends
that interval. The measured intervals accumulate across the whole call.

Scripted audio, scripted pauses, caller playback, and the required machine silence
are excluded from these intervals. A scripted uncertain stage does not consume
the extra allowance. A raw uncertain classification that retains a concrete stage
counts toward the two-turn allowance but does not itself start a waiting interval.

Normal response latency does not start the waiting timer. The report records
`turnaround_seconds` separately: the sum of input-end to expected decision/action
start times, including required machine silence. Normal latency has no five-second
cap. The scenario's fixed overall call deadline still bounds the runner.

Raw `uncertain` classifications count even when AMD retains an established
machine stage. The runner observes the PR's `AMD classification` debug records
for this information, which is not present in the public prediction event.
A missing raw record for a successful prediction fails the evidence check.
An expected uncertain step is not an *extra* uncertain prediction. The
`uncertainty-limit` script explicitly exercises the three-turn SDK limit.

## Coverage

| Group | Scenarios |
| --- | --- |
| Direct answer | Human, unavailable number, full mailbox |
| Screening | Repeated questions; human, voicemail, and rejection exits |
| Voicemail | Idle, repeated prompt without duplicate message, mailbox failure |
| Menus | Direct IVR, nested IVR, human, voicemail, and unavailable exits |
| Recording | Voicemail → IVR → human; voicemail → IVR → voicemail re-recording |
| Audio timing | Human interrupts voicemail; split screening rollover; delayed post-message menu |
| Uncertainty | Ambiguous fragment → human; three ambiguous turns |
| Lifecycle | Silent call; screening and IVR idle; silent and screening disconnect |
| Overall deadline | Screening → voicemail with a fixed 40-second overall deadline |

The coverage test compares the scripts with the PR's allowed-transition table,
including repeated stages. It checks edge coverage rather than claiming every
possible number of loop iterations. Human scenarios include a follow-up turn
to check that normal conversation continues after AMD completes.

The overall-deadline case uses a 40-second overall timeout and a 45-second
non-voicemail idle timeout to distinguish the two exits without waiting two
minutes. Other scenarios retain the PR's timeouts.

## Grading and evidence

The deterministic grader checks the stage trajectory, prediction-before-action
ordering, required machine silence, exact DTMF, reply count, interruption,
completion category and reason, local voicemail playback, and call-wide budgets.
An incorrect concrete stage fails even if the call later reaches the correct
outcome. Extra uncertainty does not excuse a premature reply or wrong action.

A separate LLM judge checks each reply's required facts and meaning. It accepts
paraphrases and rejects invented facts, wrong answers, or missing information.
Its verdict can fail a run; it cannot override a deterministic failure. Judge
errors and missing verdicts fail the run. This semantic check is probabilistic.

Each run saves:

- `manifest.json`: code revision, model and voice choices, budgets, and selected scripts.
- `summary.json`: pass/fail results and reasons.
- `<scenario>/result.json`: script, timestamped inputs, raw classifications,
  AMD events, received DTMF, reply text, audio observations, and verdicts.
- `<scenario>/callee.wav`: published scripted audio, including silence.
- `<scenario>/caller.wav`: audio received by the callee, with network gaps preserved.

WAV time zero corresponds to the trace's monotonic time origin. The caller's
reply text comes from AgentSession; received audio confirms that speech reached
the callee. The suite does not independently transcribe the caller's audio.
Model errors are real failures of the run. The suite does not inject malformed
model responses or model timeouts; existing AMD unit tests cover those paths.

SIP answer gating, carrier early media, realtime models, and the obsolete cloud
AMD gateway are outside this room-based suite.
