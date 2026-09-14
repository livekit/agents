# Draft AMD review

Reviewed [PR #6202](https://github.com/livekit/agents/pull/6202) at
`7dfc05f6e2331e0bfc2616489f31c22393d70c90`. Read its changed implementation,
tests, three submitted reviews, and all ten inline comments/replies.
No PR state or comments were changed.

## Taste Score

🟡 Acceptable. The draft improves v1 screening support. Its fixed-message loop
does not fit the new multi-turn design, so do not cherry-pick the full PR.

## Language Lens

Python. The review focuses on behavior worth retaining in the replacement.

## Fatal Problems

No fatal blocker to reusing individual parts. Do not carry over these limits:

1. Resetting all transcript history between screening turns loses evidence.
2. Restarting the detection budget after each screening reply can extend the run
   without a fixed overall bound.
3. Mapping screening to an older protocol cannot provide full v2 observability.

## Code Smells

### Defensive Code

The model allowlists can drift. Do not copy warnings that imply an unlisted
model is incompatible. Keep explicit model selection and document the default.

### Special Cases

The separate screening-message and voicemail-message playback loop duplicates
reply handling that now belongs to AgentSession. Keep the session reply guard.

### Post-Construction Mutation

The draft adds screening and playback fields to a classifier verdict later.
Keep separate prediction and completion events in v2.

### Function Design

Do not move fixed-message playback, IVR startup, and classification reset into
the new detector. Stage instructions can use the existing session flow.

### Nesting and Control Flow

Keep stage transitions explicit. An uncertain result must not reset an
established stage. A model result from replaced work must not change that stage.

### Error Handling

Preserve STT fallback and cleanup that releases pending callers. A failed
resource close must not leave the reply guard installed.

### Dead Code

Remove the replaced AMD gateway server, draft protobuf client, and transport
tests. The unrelated legacy one-shot classifier remains unchanged.

## Improvement Suggestions

| Draft idea or review concern | Decision in this branch |
| --- | --- |
| Optional separate STT, faster source wins | Keep. Race per turn. Use only final, non-empty text. The Agent's transcript stays unchanged. |
| Dedicated STT failure falls back to session text | Keep. Add provider-open failure coverage. Do not make a normal race outcome a warning. |
| Split screening-to-message-taking prompts | Keep the rolling turn history, including replaced inference and late text. Add the reviewer's split-turn case. |
| Human takes over during playback | Keep the session interruption path and transient human-state notice. Existing tests cover screening, voicemail, and IVR. |
| Repeated screening | Keep the independent 120-second hard deadline. Do not copy the per-screening reset of the overall budget. |
| VAD misses the greeting | Keep text-driven classification at client-side EOT. Do not infer human from a short or missing speech bracket. |
| AEC/agent audio contaminates classification | Give the optional STT the same muted audio frames as session STT. Do not disable AEC warmup from AMD. |
| Screening remote-session compatibility | Bring over the `AMD_UNKNOWN` mapping and regression test. Forward the event instead of dropping it. |
| Cleanup releases `execute()` and authorization | Keep and extend. Test failure while closing an owned model. |
| Automatic fast-model defaults | Make the change explicit. Omitted LLM uses the active Agent's LLM. Omitted/None STT adds no stream. Customers can supply fast models. |

The source review discusses
[split-turn rollover](https://github.com/livekit/agents/pull/6202#discussion_r3837424261),
[screening loops](https://github.com/livekit/agents/pull/6202#discussion_r3837424252),
[STT race logging](https://github.com/livekit/agents/pull/6202#discussion_r3837424254),
[AEC](https://github.com/livekit/agents/pull/6202#discussion_r3837424256),
[missing VAD](https://github.com/livekit/agents/pull/6202#discussion_r3837424263),
[model defaults](https://github.com/livekit/agents/pull/6202#discussion_r3837424250),
and [remote observability](https://github.com/livekit/agents/pull/6202#discussion_r3837424258).

Do not add a substring override yet. The reviewer
[clarified](https://github.com/livekit/agents/pull/6202#discussion_r3840387304)
that the reported misclassifications used the old build without a screening
category. The report is useful test evidence, but it does not measure v2 accuracy.
The prompt now includes more screening/voicemail contrasts. Add repeated model
evals before deciding whether narrow deterministic rules improve accuracy.

## Refactoring Priority

1. Local migration and the STT race/history/cleanup tests are complete.
2. Run repeated model evals for screeners, partial transcripts, split rollover,
   real human greetings, and misleading voicemail phrases.
3. Add full v2 remote-session fields, tracing, and job-tag coverage before a
   production replacement. Realtime support and the active-Agent facade remain
   separate decisions.

## Rewrite Examples

### Keep evidence when replacing inference

❌ Current draft concept:

```python
await classifier.reset()  # Clears the transcript between screening turns.
classifier.start_listening()
```

✅ Replacement concept:

```python
history.append(committed_turn)
cancel_previous_inference()
start_inference(snapshot(history))
```

Cancel work, not evidence. Late STT can update history for the next snapshot.

### Keep one fixed overall deadline

❌ Current draft concept:

```python
await play_screening_message()
classifier.arm_detection_timer()  # Restarts the overall detection budget.
```

✅ Replacement concept:

```python
start_hard_deadline_when_listening_starts()
# Stage changes do not extend the hard deadline.
rearm_idle_after_playback()
```

The replacement also keeps the 60-second voicemail idle period for late menus.
