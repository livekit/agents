# Experimental multi-turn AMD

AMD classifies the call participant at client-side end of turn (EOT). It can
handle several screening, voicemail, and IVR turns before it completes.
The SDK owns classification, stage changes, reply guards, and deadlines.
AgentSession owns the customer hook, interruption, and playback.

This replaces the one-shot AMD API. `execute()` now returns `AMDCompletedEvent`,
not the first `AMDPredictionEvent`. Use `amd_prediction` to observe each decision.
The application still decides whether to continue or end the call.

## Start AMD

Start AgentSession first. Enter AMD before creating the SIP participant.
Do not generate an `on_enter` greeting before AMD starts.

```python
from livekit.agents import AMD

detector = AMD(
    session,
    participant_identity=callee_identity,
    # llm=classification_llm,  # Optional classification and menu model.
    # stt=fast_stt,            # Optional second STT for AMD only.
)

@detector.on("amd_prediction")
def on_prediction(event):
    print(event.turn_id, event.category, event.reason)

@detector.on("amd_menu_observed")
def on_menu(event):
    record_menu(event)  # Informational. This event does not authorize an action.

async with detector:
    # Create the SIP participant here.
    result = await detector.execute()
    # Choose the next call action from result.category and result.reason.
```

Omit `participant_identity` for console input. By default, AMD and the Agent
discard SIP pre-answer audio. Set `wait_until_answered=False` to use subscribed
early media. This does not cause the SIP provider to supply early media.

## Model selection and transcript race

- `llm` accepts an LLM instance or a model string. If omitted, AMD uses the
  active Agent's pipeline LLM. It does not auto-select a fast model.
- `stt` accepts `str | STT | None | NotGiven`. If omitted or `None`, AMD uses
  only the session transcript. A model instance or string enables a second STT.
- Model strings use LiveKit Inference. Provider plugins can use the caller's
  own provider credentials.
- AMD closes models it creates from strings. It does not close supplied model
  instances. It closes its requests and STT streams in either case.

The first non-empty final transcript wins each AMD turn. Empty and interim
results cannot win. AMD collects further final segments from the winning source
until client-side EOT. The Agent's transcript, history, hooks, and EOT stay
unchanged. A faster AMD transcript does not make the Agent's EOT arrive earlier.

If neither source has text at EOT, AMD waits up to 500 ms within the prediction
deadline. A non-streaming STT receives the turn's audio at EOT. A streaming STT
uses one stream per committed turn; its reader can drain for up to 30 seconds.
This keeps late results attached to the original turn. Both STT paths use the
session's audio-muting guard during AEC warmup and uninterruptible speech.

Late text updates AMD history for the next inference. It does not change an
in-flight request or a reply that already started. Losing transcripts are marked
as another reading of the same audio. History holds the last 20 committed turns.
A new turn can cancel classification work without discarding its transcript.
An empty EOT does not cancel a useful pending classification.

## Stage behavior

| Prediction | Reply behavior | AMD lifecycle |
| --- | --- | --- |
| `uncertain` | Use the current stage; otherwise permit normal reply handling. | Continue within the uncertainty and time limits. |
| `machine-screening` | Answer the screener's latest question briefly. | Continue listening for the next turn. |
| `machine-vm` | Generate one message per voicemail stage. | Keep listening during and after playback. |
| `machine-ivr` | Use the actual prompt to choose DTMF or a spoken response. | Continue listening for the next turn. |
| `human` | Supply a temporary human-state notice; earlier automated prompts no longer apply. | Complete AMD. |
| `machine-unavailable` | Cancel held replies; do not generate a machine reply. | Complete AMD. |

Stage instructions are temporary. They do not enter the Agent's saved history.
Use `screening_instructions`, `voicemail_instructions`, and `ivr_instructions`
to customize them. The customer hook runs before AMD adds its instructions.
`StopResponse` and interruption remain AgentSession responsibilities.

The normal interruption path handles a person who speaks during a message.
Interruption does not itself authorize a reply. AMD still checks the next turn.
An uncertain prediction does not reset an established stage.

Machine predictions wait for 1.5 seconds of continuous participant silence.
This includes screening, voicemail, IVR, and unavailable results. The wait covers
both the prediction event and permission to reply. Silence before EOT and during
classification counts toward the threshold. A ready result does not time out
while it waits for silence.

Human and initial `uncertain` predictions use normal EOT timing. If `uncertain`
keeps an established machine stage, the machine silence rule still applies.
New speech cancels a pending release. The next EOT can replace the prediction,
or rearm it if there is no new text. AMD retains the earlier transcript.
Superseded results do not emit a prediction or authorize an old reply.

## DTMF and menus

The built-in `send_dtmf_events` tool reports successful local sends to AMD.
The next classification includes those digits with the participant history.
A failed or canceled publish is not reported. DTMF alone neither triggers
classification nor proves that a person answered.

If a custom tool sends DTMF, call `detector.notify_dtmf_sent(digit)` after each
successful publish. Report digits separately if a sequence can fail midway.

Menu extraction runs as separate best-effort work. It never holds a reply.
Each `amd_menu_observed` event identifies the turn and contains a menu description
and options with `label`, `dtmf`, and `spoken_response`. AMD does not build a tree
or execute the observed menu. Extraction has a 5-second deadline and at most
20 options. A new classification cancels the previous menu request.

## Deadlines and cleanup

| Control | Default | Behavior |
| --- | --- | --- |
| `inference_timeout` | 1.5 seconds | Use the current stage on timeout; its silence rule still applies. |
| `machine_silence_threshold` | 1.5 seconds | Wait for continuous silence before a machine prediction releases the turn. Set to `0` to disable. |
| `idle_timeout` | 10 seconds | Complete after inactivity outside voicemail. |
| `voicemail_idle_timeout` | 60 seconds | Allow a delayed post-message menu after playback. |
| `timeout` | 120 seconds | Fixed overall limit from the start of listening. |
| `max_uncertain_turns` | 3 | Complete after consecutive uncertain predictions without an established stage. |

A late prediction can update the stage if no newer inference replaced it.
It cannot change a reply that already started. Three consecutive prediction
timeouts also complete AMD after the current stage's silence requirement.
Model errors and reused results use the same stage and silence rule.
New speech cancels the idle timer. Stage changes do not extend the overall limit.
The overall limit can end AMD during a silence wait. Without speech-end timing,
the silence wait starts at EOT.

Completion or context exit closes AMD requests, removes listeners and reply
guards, and releases pending callers. The application receives the final category
and reason. Completion does not decide the next call action.

## Run the example

Configure `.env` for the selected providers, then run:

```sh
uv run python examples/telephony/amd.py console
```

Use `dev` for a LiveKit room. To place an outbound call, also set
`SIP_PHONE_NUMBER`, `SIP_PARTICIPANT_IDENTITY`, and `SIP_OUTBOUND_TRUNK_ID`.
Console mode does not place SIP calls.

## Current limits

- Pipeline STT/LLM/TTS only. Realtime reply control is not implemented.
- Agent handoff during AMD is not supported.
- The customer Agent versus a dedicated AMD Agent remains an open API decision.
- No audio-based hold detection. `should_wait` remains false.
- The remote-session protocol maps screening to `AMD_UNKNOWN`. Full v2
  prediction, menu, and completion fields still need protocol support.
- Classification quality needs repeated model evals. Unit tests prove routing,
  transcript ownership, deadlines, and cleanup, not model accuracy.
