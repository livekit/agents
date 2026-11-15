# Voice lifecycle implementation plan

This is the execution record for the voice lifecycle refactor agreed on 2026-11-14.
It distinguishes implemented work from intended contracts. Feature specifications describe
the public API; this file tracks implementation and verification, not a replacement API.

## Constraints

- Preserve existing uncommitted work. Work against this checkout's actual implementation.
- No compatibility aliases, duplicate decision owners, speculative public types, or temporary
  interruption checks. Change generators/declarations rather than generated output.
- Decisions remain synchronous in the core; runtime executes correlated effects.
- STT owns provider speech normalization, including synthesized start AND end boundaries,
  utterance numbering, owed finals, and connection-boundary settlement.
- Interruption does not cancel running tools. Their RunContext retains the originating
  SpeechHandle. Ordinary late results are recorded without continuing the interrupted reply.
  Async-tool update delivery is a separate policy.
- Conversation items have stable IDs. Revisions do not change their logical position.
- Acquire ownership and reconcile fresh database history before executing against it.
  Absence from a local view is not evidence that an unseen database item was deleted.
- Context persistence remains explicit. Storage failures are typed and acknowledged.
- Storage applies to both VoiceSession and TextSession through the shared storage run,
  checkpoint, item-diff, session-binding and fencing primitives. Voice owns live recognition
  and playback checkpoint timing. Text with plain context/state saves manually; an explicit
  `context=store.bind(app)` opts into managed conversation checkpoints. Do not introduce voice
  policy into text or duplicate the common database machinery in either coordinator.
- Voice recognition progress must be checkpointed while the person speaks, including before
  the first reply exists. Pending input stays distinct from accepted conversation history.

## Work sequence and completion criteria

- [x] Document component ownership on the session and detailed module contracts: input,
  user turns, interruption, speech/steps/tools, output/playback, conversation, persistence.
- [x] Rename STT Speech to SpeechNormalizer, including actual consumers and tests, without
  a legacy alias. Document boundary synthesis and final-transcript obligations together.
- [x] Specify and implement user-turn lifecycle across STT, VAD, manual, turn detector and
  realtime modes, including late corrections and accepted turns that produce no reply.
- [x] Specify and implement separate speech generation, playback, tool, floor and persistence
  milestones. Use the scenario: A speaks/starts a tool; user interrupts; B starts; A's late
  playback report and tool result arrive; storage answers out of order.
- [x] Define correlated commands, observations, decisions, effects and acknowledgments using
  actual concrete types. Retired sources/bindings cannot affect current work.
- [x] Preserve non-cancelling tool interruption and original-handle identity. Define close,
  handoff, pause/resume and output replacement explicitly.
- [x] Define persistence ownership independently of user TurnId. Preserve database RunId,
  speech/step, ChatItem.id and operation identity rather than conflating them as "turn".
- [x] Verify acquire/reconcile/execute/checkpoint/commit ordering, first-read and supplied
  history safety, concurrent writers, and turns without replies.
- [x] Verify conversation ordering, tie handling, tool grouping, stable-ID revisions, and
  interruption accounting for pipeline, realtime, avatar, replacement and close paths.
- [x] Extend DST database observations with persisted assistant payloads. Check text,
  interruption flags, identity and order against playback/session evidence and reloads.
- [x] Add focused regressions for late playback, live tools, reordered receipts, unseen foreign
  rows, and excluded output/close paths. Checkers must reject broken recordings; scenarios
  must prove the relevant path was exercised.
- [x] Run focused checks per slice, architecture/codegen checks, relevant host contracts,
  and a broader DST campaign. Record commands/results below; do not claim unrun checks.

## Resolved lifecycle contracts

- Resolved: an STT end with final words owed forever uses available interim words after a
  bounded wait; a late final corrects the original turn. STT emits the held-end observation;
  the core owns the deadline. `final_transcript_timeout` defaults to 2 seconds, independently
  of transcription diagnostics and close. Manual mode still requires an explicit commit.
- An interrupted speech can release its handle and floor while its tools and persistence
  remain owned by that speech. `RunContext.speech_handle` retains that original identity.
- Existing pause/resume ownership is retained. Shutdown owns bounded cleanup; ordinary
  interruption and the generation watchdog never cancel running application tools.
- User TurnId, provider utterance identity, speech/step IDs, ChatItem IDs, database RunId and
  save sequence are distinct. Overlapping speeches may share a database run.

## Verification record

Initial audit: DST Saved records currently retain item IDs and user text, not assistant
payloads. The storage ordering rule skips foreign-item histories and faulted stores. The
interruption/text rules have output-replacement, close and late-report exclusions. These are
coverage limitations to resolve, not evidence that the implementation is correct.


## Implementation and verification

- Component contracts are in `voice-lifecycle.md`, with ownership comments on the core driver,
  STT normalizer, conversation ledger, voice checkpoint coordinator and storage adapter.
- Stable conversation insertion and tool grouping live in the shared ABI chat-context ordering
  module. Voice and storage use that primitive. Database runs/diffs/fencing remain shared with
  bound TextSession; recognition and playback timing stay voice-specific.
- Live pending recognition saves before a reply exists and keeps its item ID on acceptance.
  Accepted no-reply turns save too. Pending input is separate from model-visible conversation.
- Acquisitions and progress writes reconcile unread database rows, including overlapping
  acquisitions. Narrow model views retain history-only records and their stored neighbors.
- Final saves acknowledge actual durability, including shared runs and a canceled original
  acquire. Progress and terminal snapshots share a sequence; stale terminal saves do not regress
  newer checkpoints. Failed preparation follows the same owned-run cleanup as other failures.
- Ordinary interruption leaves tools alive past the generation cleanup bound. The host regression
  checks their original SpeechHandle after a newer reply, plus the late output's original call ID
  and `reply_required=false`. MCP shutdown remains responsible for canceling a hung call.
- DST observations include full decoded persisted payloads, concrete submitted snapshots and
  checkpoint readbacks. Checkers follow authoritative reloads, preserve foreign compactions,
  scope claims to acknowledged snapshots, and reject generated tails, lost interruption flags,
  lost accepted items, duplicates and ordering errors. Telemetry is excluded from the conversation
  payload comparison because its protocol/JSON projection is lossy.

Verification in this checkout (private Cargo target under `/private/tmp`):

- Core/ABI/voice library tests: 986 passed; 16 existing ignored tests.
- Shared storage library tests: 152 passed; 32 existing ignored tests. All 13 focused voice
  storage regressions also passed and cover final saves without an acquire, stale terminal sequences,
  pending-input recovery, overlapping unread rows and timestamp ties in narrow model views.
- STT normalizer: 11 passed.
- Enabled DST suites: 26 library checks, 358 voice regressions, 6 text contracts and the ordinary
  soak smoke test passed. The ignored long campaigns were not included in that command.
- Voice campaign: 128 seeds, 3,000 inputs per seed, passed. This is not the default 32,768-seed
  release campaign or the long soak.
- Python host contracts: 44 passed across callback contexts, session regressions, session options
  and STT speech. The focused MCP interruption/shutdown and STT pending-end checks passed too.
- Rust formatting, three Clippy policies, architecture boundaries, generated-code drift and the
  architecture/codegen test suites passed after final formatting (23 architecture tests and 158
  codegen tests).

- `UV_NO_SYNC=1 make check`: Python formatting/lint, strict mypy and pyright passed. Pyright
  reported two existing warnings in `metrics.py` and `storage/context.py`, with no errors.
- `git diff --check`: passed.

The full default release DST campaign and credentialed E2E room tests have not been run.

Text endpoint storage follow-up: real HTTP requests across two servers cover manual saving,
bound checkpoints and an unsaved control. SQLite: 3 passed; AgentDB: 3 skipped without project
credentials. The manual journey exposed a dropped assigned SessionState; explicit context saves
now include that snapshot in the same revision-checked save as ordinary fields.
