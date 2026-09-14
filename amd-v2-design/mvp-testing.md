# Client-side AMD MVP

AMD now runs in the Python SDK. There is no separate AMD server, WebSocket
protocol, or AMD authentication step.

## Run

Use the repository's `.env`. Keep its credentials private.

```sh
uv run --no-sync python examples/telephony/amd.py console
```

For a room, use `dev`. To place an outbound SIP call, also set
`SIP_PHONE_NUMBER`, `SIP_PARTICIPANT_IDENTITY`, and `SIP_OUTBOUND_TRUNK_ID`.
Console mode does not place SIP calls.

The example uses `google/gemma-4-31b-it`, `cartesia/ink-2`, and
`cartesia/sonic-3` through LiveKit Inference. These are example choices.
AMD can use provider plugins with the customer's own credentials.
Model strings still select the standard LiveKit Inference service; they do not
select a separate AMD service.

## SDK API

```python
detector = AMD(
    session,
    participant_identity=callee_identity,
    # llm=classification_llm,  # Optional. Defaults to the active Agent's LLM.
    # stt=fast_stt,            # Optional. Race this model against session STT.
)

@detector.on("amd_prediction")
def on_prediction(event):
    print(event.category, event.turn_id)

@detector.on("amd_menu_observed")
def on_menu(event):
    record_menu(event)  # Informational. Do not use this to authorize an action.

async with detector:
    # Create the SIP participant here, if needed.
    result = await detector.execute()
    # The application decides the next call action.
```

Start `AgentSession` first. Do not generate an `on_enter` greeting before AMD
starts. For console input, omit `participant_identity`.

The `llm` option accepts an LLM instance or a model string.
The `stt` option accepts `str | STT | None | NotGiven`.
Omit `stt`, or pass `None`, to use only the session transcript.
Pass a model instance or string to enable a second STT path.

AMD closes its model requests and STT streams. It does not close model
instances supplied by the caller. It closes model instances it creates from
strings. No model is selected from the presence of LiveKit credentials.

## Transcript race

The first non-empty final transcript selects the source for each AMD turn.
Interim and empty results cannot win. AMD collects further final segments from
that source until client-side EOT, then freezes the classification input.

The race affects AMD only. The Agent keeps its own transcript, conversation
history, hooks, and EOT path. Thus, a faster AMD transcript does not make the
Agent's own EOT arrive earlier.

If neither source has text at EOT, AMD waits up to 500 ms for the optional
STT result. This wait is part of the prediction deadline, not an extra delay
after it. A non-streaming STT receives the turn's audio at EOT.

The optional streaming STT has a separate stream for each committed turn.
Each stream ends input at EOT and can drain for up to 30 seconds. This keeps
late results attached to the correct turn without provider-specific timestamp
or finalization protocols. Stream setup cost depends on the provider.

Late text updates the original turn's AMD history. It cannot change a request
already in progress or a reply already started. The next inference receives
the updated history and `updated_turn_ids`. A losing transcript is marked as
another reading of the same audio, not another participant turn.

An empty EOT does not cancel useful pending classification. A new turn with
evidence can replace it, but its transcript stays in history. History holds the
last 20 committed turns. Each source transcript is limited to 16,000 characters.

Both STT paths use the session's audio-muting guard during AEC warmup and
uninterruptible speech. VAD and normal interruption handling stay in the session.

## Stages and deadlines

`execute()` returns `AMDCompletedEvent`, not the first machine prediction.
Screening and IVR can span several turns. Voicemail sends one message per stage.
An uncertain model result preserves an established stage.

The prediction deadline defaults to 1.5 seconds per committed turn.
A timeout or model error releases that turn using the current stage.
A late prediction can update the stage if no newer inference replaced it.
Three consecutive prediction timeouts, or three uncertain predictions, end AMD.

Normal idle timeout is 10 seconds. Voicemail idle timeout is 60 seconds.
The voicemail idle period starts after playback ends. New speech cancels it.
A transition to IVR restores the normal idle timeout.

The hard timeout is 120 seconds from the start of listening.
Stage changes and idle resets do not extend it.

When a human takes over, the next reply gets a temporary human-state notice.
Earlier automated prompts no longer apply. This notice does not enter the
Agent's saved history. The session still owns hooks, StopResponse, interruption,
and playback. AMD does not decide whether to end the call.

By default, AMD discards pre-answer audio from both STT paths and the Agent
pipeline. Set `wait_until_answered=False` to use subscribed SIP early media.

## DTMF and menus

The built-in `send_dtmf_events` tool reports each successful local send.
A failed or cancelled publish is not reported. A successful partial sequence
is retained. The next EOT includes the digits in classification context.

For a custom sender, report each digit after its publish succeeds:

```python
await room.local_participant.publish_dtmf(code=1, digit="1")
detector.notify_dtmf_sent("1")
```

DTMF alone does not start inference or prove that a human answered.

Menu extraction is separate best-effort work. It cannot hold a reply.
There is at most one menu request per AMD run at a time. A new classification
cancels the old menu request. Menu requests have a 5-second deadline.
Model responses are limited to 8 KiB; menus have at most 20 options.

## Current limits

- Pipeline STT/LLM/TTS only. Realtime reply control is not implemented.
- Agent handoff during AMD is not supported.
- The customer-Agent versus special AMD-Agent choice remains open.
- No audio-based hold detection. `should_wait` stays false in this MVP.
- The remote-session protocol maps screening to `AMD_UNKNOWN`.
  It does not yet carry the full v2 prediction, menu, or completion fields.
- Classification quality still needs repeated model evals across real screeners,
  languages, and partial transcripts.

## Checks

```sh
uv run --no-sync pytest --unit --no-concurrent \
  tests/test_amd_detector.py tests/test_amd_local_inference.py \
  tests/test_amd_classifier.py tests/test_audio_recognition_push_audio.py \
  tests/test_session_host.py
uv run --no-sync mypy -p livekit.agents.voice.amd
```

These unit tests use controlled providers. They prove routing, history,
deadlines, stage transitions, and cleanup. They do not prove model accuracy
or SIP audio delivery.

### Verification on 2026-09-11

- Focused AMD, audio-input, and session-host tests: 141 passed.
- AMD type checks, changed-file lint, and formatting checks: passed.
- Unit and audio-EOT suite, with the exclusions below: 2,483 passed, 3 skipped.
- Excluded `tests/test_room.py`: `livekit-server` is not installed locally.
- Deselected `test_clear_error_when_project_unresolvable`: local Google default
  credentials violate its no-credentials assumption. The test passed separately
  when `google.auth.default` was patched to raise `DefaultCredentialsError`.
  No Google or room-test code was changed.

The live voice check used Cue CLI at
`c749f5a77488ca9bfb63b142c112c09a44349689` with SDK HEAD
`8e08168c3f8921fa9f5cfa4f61c212dc00267e45` plus this working diff.
Both processes loaded the same repository `.env`. SIP dialing was disabled.

Session `sid_949f370a7d14` supplied a Google-style call-screening prompt.
AMD classified it as `machine-screening` in 314 ms after client-side EOT.
The Agent gave its name and call purpose. Client-side audio crosscheck passed:
one committed reply, one aligned audio window, no unspoken replies, and no lag.

Artifacts are in
`/Users/chenghao/.cue-cli/sessions/sid_949f370a7d14/`:
`events.jsonl`, `recordings/001_run/result.json`, and `recording.wav`.
Recheck the stored audio with:

```sh
uv run --no-sync cue-cli crosscheck --session sid_949f370a7d14
```

Run that command from the Cue CLI checkout. A later connection reset prevents
a multi-turn live verdict. This run does not prove SIP delivery, the optional
STT race, voicemail takeover, or repeated classification accuracy. Those paths
still need live coverage. The earlier setup-only session `sid_5bbe124bd49c`
reached idle timeout before speech and does not count as a behavior check.

See [draft review](draft-pr-6202-review.md) for the source review and migration decisions.
