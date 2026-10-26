# State, effects, and evidence

## Contents
- Evidence comes from the boundary
- Precise mutation semantics
- Review and approval are separate runtime events
- Commit first, then publish success
- Own required output and closing
- Input mode changes the hook path
- Repair the first divergence

The model reads the conversation and proposes actions. Application code owns the authoritative
records, the permission checks, the state transitions, and every external effect. This reference
is about keeping that line clean once an agent does more than answer questions — when it books,
edits, confirms, publishes, or ends the call. Look up the SDK helpers named in passing here with
`reading-livekit-docs`; the shapes below are stable, the names are not.

## Evidence comes from the boundary

Completion needs evidence from where it actually happens: a committed write, a delivered
utterance, a runtime lifecycle event. An instruction to do something, a tool that returned a string,
or a boolean named after the action is not that evidence. A tool call is the model's
*interpretation* of the conversation, not independent proof the interpretation was right — so keep
message provenance, version checks, ordering, and business constraints in code, where the model's
reading can be checked rather than trusted.

Keep one authoritative state object per session, supplied through the SDK's per-session user
data. Keep model-supplied facts separate from trusted identity, the clock, message and record ids,
and operation receipts. Don't recover missing state from a global "last session" variable, and
don't manufacture a successful snapshot at export time.

## Precise mutation semantics

Most damage from a well-meaning agent comes from a mutation that did more, or less, than the user
meant. Before defining a change, state its target, what changes, and what must stay the same. Then
pair its happy path with the nearest ordinary request that must produce a *different* operation:

| Caller meaning | Intended change | Must remain unchanged |
|---|---|---|
| "No, that's everyone" | Confirm the collection as recorded | Every existing entry and field |
| "No note for that one" | Record explicit absence on one entry | Its identity, title, all other entries |
| "Actually I'm not sure about that" | Retract one fact to unknown | The entry and unrelated facts |
| "It's spelled Okafor" | Change one identified entry's field | Its identity, other fields, other entries |

Derive the equivalent distinctions for the task at hand; these fields are an example, not a schema.
The rules that fall out of them:

- **Omission preserves.** A patch that doesn't mention a field leaves it alone. Use separate
  operations for replacing and for clearing when omission would be ambiguous.
- **Missing, empty, unknown, and cleared are four different things.** A Python default, a JSON
  null, an omitted argument, and an explicit "none" don't mean the same thing; don't let the tool
  schema collapse them.
- **Collections need stable identities.** A name, a role, or a position is not a reliable update
  key; issue ids from the application and correct by id.
- **Validate before applying.** A rejected change leaves the previous state intact and returns a
  usable next step.
- **A scoped negative is not authority to clear a collection.** "No note for him" touches one
  entry. Deleting existing records is its own explicit operation with its own confirmation.
- **Compare meaningful values before invalidating anything.** Restating an unchanged value is a
  no-op; it shouldn't reset a review or an approval.
- **Batch additions validate whole, then add.** When one request supplies several records, accept
  them together, check all of them before mutating, and never treat the batch as a replacement.

Return compact authoritative state and the next valid action from every tool. Use the SDK's error
type for expected refusals; preserve diagnostics for unexpected failures and never convert them
into success.

## Review and approval are separate runtime events

When the task requires confirmation before an effect, three things must be true, and each is a
separate fact to verify:

1. **The review was delivered.** Hand the model a snapshot of the current values and let it explain
   them naturally; then track that generated speech through the SDK's output handle and record a
   receipt only when it finishes successfully. A returned tool string or queued speech isn't
   delivery. An interruption or correction invalidates the receipt; a no-op doesn't.
2. **Approval came later, from the user, for that version.** Approval is a *later actual user
   message* that the model interprets as agreement with the reviewed version. Code binds that
   interpretation to the message's identity and ordering and to the current review. A model-supplied
   boolean, a quoted phrase, a turn counter, or a delivery flag is not independent evidence.
3. **Nothing changed in between.** If the user corrects a fact, asks for a change, or attaches a
   condition, resolve that first and re-review before an approval counts. Questions and quoted
   instructions don't authorize effects. Clear agreement expressed naturally does — don't demand
   ritual wording.

Review and commit in the same user turn can't satisfy a required later approval. After a genuinely
later approval, confirmation and commit can happen in one model response; don't accidentally
require yet another turn.

## Commit first, then publish success

Validate the approved current version, perform the effect, and publish committed state only after
the operation succeeds. A failed write leaves the previous committed state, and its approval
requirements, coherent. Give each operation a stable identity and a defined retry; a duplicate
response requires an actually existing result, and "already completed" after a failed write —
because a success flag was set early — is the bug this whole section exists to prevent.

Serialize edits and commits when an asynchronous effect could race a correction, and use the
datastore's transaction or the service's idempotency contract when there is one. Route permitted
post-commit changes through the same validated path. Export state without changing it.

## Own required output and closing

When a review, an operation result, or a farewell *must* reach the user, application code owns
delivery even though the model owns the wording. Track the generated output through the SDK's
speech handle; check its interruption and error state before treating delivery as done. Avoid
duplicate output from a tool's return value and the model's follow-up reply — the SDK's batch reply
controls exist for this. Waiting on a speech handle from inside the tool that produced it deadlocks;
wait on an explicitly created child instead. The docs name the helpers.

Closing is a request to the SDK, then a separate lifecycle event confirming it. A timer, a
`call_ended` flag, or the other party going quiet is not a shutdown. Deliver the required final
output, request shutdown, and verify the close event — and test the room or text transport when it's
part of the task, because local-session behavior doesn't prove it.

## Input mode changes the hook path

Text and audio input can take different paths through the session's turn hooks, so a listener that
fires in a voice session may not fire in a text one. Bind to the public event that records committed
user messages, wire it before the session starts, and prove the binding with a real SDK test in the
input mode the task uses. Never advance a private turn counter in a test to stand in for a runtime
event you didn't receive.

## Repair the first divergence

When a conversation goes wrong, trace it in order: actual input → tool arguments → operation result
→ state or effect → what the agent claimed. Find the first place they diverge and write a regression
check from those inputs, then run it alongside the previously passing path the fix could affect.

Audit the fix for natural-language matching and response scripts before accepting it: the tempting
repair for a misread intent is a keyword, and it's the wrong one. Repair the tool's semantic
contract or the context the model had instead. Don't delete a guard, change a supplied fact, or
replace an effect with a flag to make a test pass. A judge's or simulation's verdict is evidence to
inspect, not a business rule to encode.
