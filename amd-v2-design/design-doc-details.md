# Superseded gateway draft

As of 2026-09-11, AMD stays in the client SDK. This document records the earlier
gateway proposal. See [the current MVP](mvp-testing.md) for the local API and behavior.

## Summary

AMD v2 replaces the local, one-shot classifier with a cloud detector. It classifies each participant turn until AMD completes.

The gateway detects categories. The SDK controls stage changes, replies, timers, and completion. AgentSession controls interruption and playback.

### Design rules

- Use one inference stream for each AMD run. Process turns in order.
- Let the application enable AMD once. The application does not need to manage stages.
- At each client-side end of turn (EOT), commit the participant turn for classification.
- Include short turns that AgentSession would otherwise suppress.
- Emit a prediction before its stage action. Complete AMD only after the completion action takes effect.

### Categories

| Category | AMD status | New reply |
|---|---|---|
| `uncertain` | Active | Allow the normal reply. |
| `machine-screening` | Active | Reply to each turn with screening instructions. |
| `machine-vm` | Active | Generate one message on stage entry. |
| `machine-ivr` | Active | Use speech or DTMF for each turn. |
| `human` | Complete | Allow the normal reply. |
| `machine-unavailable` | Complete | Cancel the held reply. |

Screening, Voicemail, and IVR can continue across multiple turns. Human and Unavailable complete AMD.

### Reply control

For each committed turn:

1. Send the AMD classification request. Hold permission for a new reply.
2. Call the customer Agent's `on_user_turn_completed` once with the participant message and mutable `turn_ctx`.
3. Wait for the hook and the prediction or fallback. The hook and classification can run in parallel.
4. Remove controls from the previous stage. Add the current stage's controls after the customer's context changes.
5. Check any speculative reply against the final context and stage. If its inputs or reply path changed, cancel it.
6. Apply the current stage's reply rule. Authorize only the selected reply.

Customers add context in `on_user_turn_completed`. AMD adds its controls at the same point. Never pass an AMD control message as `new_message`. Never call the hook again for that control message.

If the hook raises `StopResponse`, suppress the new reply. AMD still classifies the turn, emits events, and removes old controls.

### Interruption

Interruption does not authorize a reply. AgentSession can pause or stop current speech while AMD blocks a new reply.

AgentSession decides whether an interruption is false or confirmed. It also controls playback, audio accounting, and interruption events. AMD does not delay these decisions or their timers.

### Control messages and cleanup

AMD control messages contain stage instructions in the `user` role. Append them at the end of the context. This preserves the stable instruction prefix and its input cache.

- Tag each control message with its AMD run and stage.
- When a stage ends, remove that stage's control messages.
- When AMD completes, remove all controls from that run. This includes timeout, failure, and cancellation.
- Keep participant transcripts, customer context changes, Agent replies, and tool history.
- For realtime models, update the provider context before the next reply.
- Cancel held speculative replies that used removed controls.

Control messages do not emit `user_input_transcribed` or `conversation_item_added`. They do not change the speaking state.

## Model compatibility

| Agent model | V1 support | Reply control |
|---|---|---|
| Pipeline LLM with STT and TTS | Supported | Generation can start early. Playback and tools wait for AMD authorization. |
| Realtime model with client-side turn detection | Supported with the capabilities below | Wait for the hook and AMD decision. Update the context, then request generation. |
| Realtime model with server-side turn detection | Unsupported | The provider can generate output or tool calls before AMD responds. |
| Realtime model without mutable chat context | Unsupported | AMD cannot safely add or replace stage controls. |

A supported realtime model must allow server turn detection to be disabled. Its chat context must also be mutable. Holding playback does not remove output already added to provider history.

During `__aenter__`, check these capabilities before AMD blocks replies or starts listening. For an unsupported realtime configuration, raise a compatibility error with instructions to use client-side turn detection. V1 does not switch turn-detection modes at runtime or fall back to a pipeline model.

A pipeline LLM can prepare a speculative reply while AMD classifies the turn. Before playback or tool execution, check it against the final hook context and AMD decision.

## Public API (TODO)

- [ ] Choose `AMD(session)`, a predefined `AgentTask`, or both later. The core AMD logic, events, and results stay the same.

## Stage behavior

The shared timeout and error rules apply to all stages. Each stage's fine print lists its specific rules.

### Uncertain

{{image:uncertain}}

AMD starts in `uncertain`, not a null state.

- For a pipeline LLM, allow speculative reply generation. Hold playback and tools until AMD authorizes it.
- If the result stays `uncertain`, allow the normal reply without machine-stage controls.
- If the result is Screening, Voicemail, or IVR, cancel the normal speculative reply. Apply the new stage's instructions.
- For Human, allow the normal reply. Complete AMD.
- For Unavailable, cancel the held reply. Complete AMD.

<details color="gray">
<summary>Timeouts and error handling</summary>
	- Start `idle_timeout` when listening starts. Restart it after each reply while AMD remains active.
	- Before any concrete category, three consecutive uncertain turns complete AMD with `category="uncertain"` and `reason="max_uncertain_turns"`.
	- If the prediction deadline expires, allow the normal reply. Emit `reason="inference_timeout"`.
	- A late concrete prediction can replace the fallback for the current turn. A newer turn cancels this opportunity.
	- Completion predictions are an exception. Apply the shared completion rules.
</details>

### Screening

{{image:screening}}

Screening is an exchange with a call-screening system. It starts only from Uncertain.

- When `machine-screening` is confirmed, cancel the normal speculative reply.
- For each committed turn, append a control message from `screening_instructions`.
- Generate one reply for that turn.
- Keep the configured Agent instructions, model, voice, hooks, and tools.

Screening can change to Human, Voicemail, or Unavailable. It cannot change directly to IVR.

Default `screening_instructions`:

> Call state: automated call screening. Speak to the screening assistant so it can connect the call. Answer its latest prompt directly and briefly using what you know about who you are and why you are calling. Then wait for its next prompt.

<details color="gray">
<summary>Timeouts and error handling</summary>
	- Start `idle_timeout` after the reply finishes playing.
	- An uncertain prediction or one inference fallback keeps `machine-screening`. It does not count toward `max_uncertain_turns`.
	- If the idle timer expires, complete with `category="machine-screening"` and `reason="idle_timeout"`.
</details>

### Voicemail

{{image:voicemail}}

Voicemail starts from Uncertain, Screening, or IVR. AMD stays active while the Agent leaves a message.

- On stage entry, append `voicemail_instructions`. Generate one concise, self-contained message.
- Continue detection during generation and playback.
- At client-side EOT, classify the participant's next turn.
- For `machine-vm`, raw `uncertain`, or an inference fallback, suppress a new reply.
- Do not regenerate an interrupted message while the category stays unchanged.
- For Human, IVR, or Unavailable, remove the old controls. Apply the new stage's reply rule.
- After the full message plays, set `voicemail_message_played=True`. Keep this flag for the rest of the AMD run.

Voicemail cannot change to Screening.

A later IVR menu can request submission, review, deletion, recording, or re-recording. `voicemail_message_played` confirms local playback only. It does not confirm that the voicemail system stored the message.

Default `voicemail_instructions`:

> Call state: voicemail. Deliver one concise, self-contained message. State who you are, why you are calling, and the next step you want the recipient to take.

<details color="gray">
<summary>Playback, timeouts, and error handling</summary>
	- For a false interruption, AgentSession can resume the existing message. This requires pause-capable output and false-interruption resume support.
	- A confirmed interruption can stop the message without authorizing another reply.
	- If AgentSession discards the speech handle, AMD cannot resume it. AMD does not add a playback buffer.
	- Partial playback does not set `voicemail_message_played`.
	- Start `idle_timeout` after playback finishes.
	- After client-side EOT, an unchanged category, inference timeout, or inference error can restart the idle timer.
	- Apply all shared idle-timer conditions before this restart. This also applies if AgentSession has already discarded the speech handle.
	- If playback is paused or resumes, wait for AgentSession to finish playback or resolve the interruption.
	- If the category changes, do not restart the old Voicemail timer.
	- A raw uncertain prediction keeps `machine-vm`. It does not count toward `max_uncertain_turns`.
	- For a connection failure, apply the shared failure rules. AgentSession still controls current playback.
</details>

### IVR

{{image:ivr}}

IVR is a menu that can continue across multiple turns. It starts from Uncertain or Voicemail.

- For each committed turn, append a control message from `ivr_instructions`.
- Let the Agent reply with speech or the DTMF tool.
- Keep AMD active while the menu continues.
- The existing IVR helper still supports DTMF, repeated prompts, and silence. It does not classify AMD stages.
- After Voicemail, keep `voicemail_message_played`. Set `prev_stage_category="machine-vm"` for the full IVR stage.

IVR can change to Human, Voicemail, or Unavailable. It cannot change to Screening.

<details color="gray">
<summary>Timeouts and error handling</summary>
	- Start `idle_timeout` after each spoken reply or DTMF action finishes.
	- An uncertain prediction or one inference fallback keeps `machine-ivr`. It does not count toward `max_uncertain_turns`.
	- After Voicemail → IVR, an idle or hard timeout completes AMD with `category="machine-ivr"`.
	- Keep the previous-stage field and message-played flag in the completion event.
</details>

### Human

{{image:human}}

Human completes AMD. The Agent returns to normal reply handling.

1. The gateway sends and flushes the Human prediction before it starts the clean WebSocket close.
2. The SDK updates its stage. It emits `amd_prediction` before the next action.
3. Remove AMD controls from the Agent and provider context.
4. Apply `StopResponse` before the reply decision.
5. When the hook permits a reply, use a valid held normal reply or generate one from the cleaned context.
6. Mark AMD complete. Emit `amd_completed(category="human", reason="finished")`.

AgentSession still controls current speech and its schedule. Later replies do not wait for AMD.

<details color="gray">
<summary>Timeouts and error handling</summary>
	- Apply the shared completion rules, including late predictions and clean connection closure.
	- After the Human transition, no AMD timer can complete the run again.
</details>

### Unavailable

{{image:unavailable}}

Unavailable means rejection, mailbox failure, or a call path that cannot continue. It completes AMD.

1. The gateway sends and flushes the Unavailable prediction before it starts the clean WebSocket close.
2. The SDK updates its stage. It emits `amd_prediction` before the next action.
3. Cancel the held reply. Remove AMD controls from the Agent and provider context.
4. Mark AMD complete. Emit `amd_completed(category="machine-unavailable", reason="finished")`.

AgentSession still controls current speech. The application uses the completion event to choose the next call action.

<details color="gray">
<summary>Timeouts and error handling</summary>
	- Apply the shared completion rules.
	- If the connection closes unexpectedly before the completion prediction is flushed, use the connection-error path.
	- After the Unavailable transition, ignore AMD timers and pending active-stage results.
</details>

### Wait prediction

`wait` identifies hold music, advertisements, or other non-interactive audio. It is a temporary prediction, not a stage.

- Keep the current stage and `prev_stage_category`.
- Emit the prediction. Stay silent without starting reply generation.
- Detect sustained wait audio without client-side EOT.
- Pause `idle_timeout` while wait audio continues. The hard `timeout` still applies.
- On the next non-wait prediction, resume the current stage's rules.

## Shared rules

### Timers and limits

A concrete category is any category other than `uncertain`.

| Setting | Default | Start and reset | Limit reached |
|---|---:|---|---|
| `idle_timeout` | 10 seconds | See the idle-timer rules below. | Complete with the current category and `reason="idle_timeout"`. |
| Hard `timeout` | 120 seconds | Starts when AMD starts listening. Never resets. | Complete with the current category and `reason="timeout"`. |
| Prediction deadline | 1.5 seconds, internal | Starts at each committed turn. Ends on its ordered prediction. | Emit a fallback. Three consecutive timeouts complete AMD. |
| `max_uncertain_turns` | 3 | Counts only before the first concrete category. A concrete category stops the counter. | Complete with `category="uncertain"` and `reason="max_uncertain_turns"`. |

### When listening starts

- By default, AMD starts listening after answer.
- With `wait_until_answered=False`, listening starts when the participant track is subscribed and early media is processed.
- With answer-gated listening, discard pre-answer audio from AMD and the Agent conversation.

### Idle timer

Start `idle_timeout` when listening starts. Restart it after the current stage's playback or action finishes.

Participant speech cancels the timer. A `wait` prediction pauses it. New participant speech also cancels any pending timer restart.

Restart the timer only when all these conditions are true:

- AMD is active and the current turn decision is complete.
- The participant is silent and no `wait` prediction is active.
- No reply or action is pending.
- No playback is active or paused.
- No interruption handling remains pending.

If a turn produces no reply or action, check these conditions after its decision is complete. This includes `StopResponse` and discarded speech handles. Do not wait for a playback event that cannot occur.

### Prediction fallbacks

If inference times out or fails, emit a fallback `amd_prediction`. Keep the current concrete category. If none exists, use `uncertain`. Apply that category's reply rule.

After a concrete stage, a raw uncertain prediction keeps the current category. It does not count toward `max_uncertain_turns`.

The SDK does not send fallback categories to the gateway. A local fallback must not change gateway detection.

### Errors and shutdown

| Condition | Required action |
|---|---|
| Initial gateway connection fails | `__aenter__` raises before AMD starts. Do not place the call. Do not emit `amd_completed`. |
| Three consecutive inference timeouts | Complete with the current category and `reason="inference_timeout"`. A successful prediction resets the count. |
| Unexpected WebSocket close | If a turn is pending, emit its fallback prediction. Complete with the current category and `reason="connection_error"`. Do not reconnect automatically. |
| Participant is missing | Complete with `reason="participant_missing"`. |
| Participant disconnects | Complete with `reason="participant_disconnected"`. |
| Manual close, context exit, or AgentSession shutdown | Complete once with the current category and `reason="cancelled"`. Repeated `aclose()` calls have no effect. |

All completion paths remove AMD controls. AgentSession continues to control existing speech.

### Completion rules

- The gateway flushes a Human or Unavailable prediction before it closes the connection.
- A clean close after that prediction is expected. It is not a connection error.
- While AMD is active, accept a completion prediction even if a newer local turn has started.
- Cancel obsolete work for pending active-stage predictions.
- After AMD completes or is canceled, ignore further predictions and timers.
- Emit exactly one `amd_completed` for each successful `__aenter__`. Entry failures emit none.

## Events

### `amd_prediction`

Emit this event for every committed turn. This includes unchanged categories and local fallbacks.

Emit it before a stage change or a new-reply decision. AgentSession's interruption events do not wait for this event.

| Field | Meaning |
|---|---|
| `turn_id` | Increasing identifier for the committed participant turn. |
| `category` | Category that the SDK uses for this event. |
| `prev_turn_category` | Category from the previous classified turn, if any. |
| `prev_stage_category` | Category before the current uninterrupted stage, if any. |
| `state_changed` | Whether this prediction starts a new stage. |
| `reason` | `prediction`, `late_prediction`, `inference_timeout`, or `inference_error`. |
| `transcript` | Available participant transcript. It can be empty. |
| `speech_duration` | Duration of the classified participant speech. |
| `inference_duration` | Cloud inference duration, when available. |
| `detection_delay` | Time from committed client-side EOT to this event. |
| `voicemail_message_played` | Whether the full voicemail message played locally. |

### `amd_completed`

Emit this event once, after the completion action takes effect. Every `execute()` waiter receives the same event instance.

The event contains the latest values of:

- `category`, `turn_id`, and `transcript`.
- `prev_turn_category` and `prev_stage_category`.
- `voicemail_message_played`.

Normal completion uses `reason="finished"`. Timeouts, errors, disconnection, and cancellation keep their specific reasons.

## Gateway and SDK responsibilities

| Inference gateway | SDK controller |
|---|---|
| Receive continuous participant audio. | Create one single-use stream for each AMD run. |
| Keep audio and detection history for the run. | Commit turns at AgentSession's client-side EOT. |
| Process turns in increasing `turn_id` order. | Hold reply playback and tools until the reply decision is ready. |
| Return exactly one ordered state snapshot for each committed turn. | Add stage controls. Apply the reply rule. |
| Detect category changes. | Apply fallback rules, timers, events, and completion. Use AgentSession's interruption handling. |
| After flushing Human or Unavailable, stop accepting audio and close. | Close for local timeout, error, disconnection, or cancellation. |

## Controller API sketch

This API is provisional. The public API choice remains a TODO.

```python
AMD(
    session,
    *,
    model=inference.AMD(),
    participant_identity=NOT_GIVEN,
    wait_until_answered=True,
    screening_instructions=DEFAULT_SCREENING_INSTRUCTIONS,
    voicemail_instructions=DEFAULT_VOICEMAIL_INSTRUCTIONS,
    ivr_instructions=DEFAULT_IVR_INSTRUCTIONS,
    idle_timeout=10.0,
    timeout=120.0,
    max_uncertain_turns=3,
)
```

- Context entry connects to the gateway. AMD starts when its listening condition is met.
- `execute()` waits for completion. Repeated calls return the same event.
- Canceling one waiter does not cancel AMD.
- Use each controller once. Each outbound attempt needs a new controller and inference stream.
- V1 instruction options are plain strings.

## V1 exclusions

- No separate screening persona.
- No per-stage callbacks.
- No special tool restrictions for Screening or Voicemail.
- No automatic gateway reconnect.
- No runtime switch from server-side to client-side realtime turn detection.
- No `conversation_item_added` event for AMD control messages.

## Open questions

- Choose how the gateway sends `wait`: a separate event or a temporary-prediction field. It must not change `category`. It can arrive without a committed speech turn.
- Define the default `ivr_instructions`. The prompt must guide speech and DTMF without duplicating the IVR helper's rules. The Screening and Voicemail defaults are settled.
