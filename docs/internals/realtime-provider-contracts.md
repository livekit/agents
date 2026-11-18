# Realtime provider contract probes

Synthetic requests against `gpt-realtime-mini` on 2026-11-17 verified the cases below.
No real conversation content or credentials are included in the fixtures. The recorded
error envelopes live in `lk-agents-inference/src/realtime/openai/wire/fixtures/openai_errors.json`.
The decoder tests consume those envelopes, and the DST's independent server implements the
operations the production client can issue. The DST runs the production realtime machine;
it does not open sockets.

| Operation | Observed OpenAI result |
| --- | --- |
| Cancel without an active default response | `response_cancel_not_active`; harmless cancellation race |
| Cancel an unknown response ID | `response_cancel_not_found`; client event ID can be absent |
| Create while a default response is active | `conversation_already_has_active_response`; original response continues |
| Create with an unregistered named tool | `response.created`, then `response.done` with `status=failed`, `invalid_tool_choice` |
| Parallel `conversation=none` responses | Accepted; they do not occupy the default conversation's response slot |
| Cancel without an ID while only out-of-band responses exist | `response_cancel_not_active` |
| Commit an empty or sub-100ms buffer, including after clear | `input_audio_buffer_commit_empty` |
| Create after an unknown predecessor | `item_create_invalid_previous_item_id` |
| Create a duplicate item ID | `item_create_duplicate_item_id` |
| Delete a missing or already deleted item | `item_delete_invalid_item_id` |
| Retrieve a missing item | `item_retrieve_invalid_item_id` |
| Output referencing an unknown function call | `invalid_tool_call_id`; client event ID absent |
| Multiple output items for one known call, with distinct item IDs | Accepted |
| Response input referencing a missing item | `invalid_reference_item_id`; client event ID absent |
| Truncate a missing item | `item_truncate_invalid_item_id` |
| Truncate a user or text-only assistant item | `unsupported_content_type`; client event ID absent |
| Truncate beyond the recorded audio duration | `invalid_value`; client event ID absent |
| Truncate content index 1 on a one-part message | `array_above_max_length`; client event ID absent |
| Truncate to zero milliseconds | Accepted |
| Negative truncation duration | `integer_below_min_value` |
| Change voice after assistant audio exists | `cannot_update_voice`; client event ID absent |
| Invalid voice | `invalid_value`, parameter `session.audio.output.voice` |
| Empty message content | `empty_array` |
| Zero output token limit | `integer_below_min_value` |
| Missing required field / unknown message type | `missing_required_parameter` / `invalid_value` |
| Item or call ID of 32 ASCII letters | Accepted |
| Item or call ID of 33+ characters, including UUIDs | `string_above_max_length` |
| Item ID containing colon, slash, spaces or non-ASCII | `invalid_value` |
| Item ID `root` / empty ID | `unsupported_value` / `empty_string` |

The connection remained usable after these validation failures. An error envelope is not
necessarily a failed connection. Likewise a failed generation can have no top-level error.
Tests must distinguish request rejection, accepted generation failure, transcription failure,
transport loss and terminal credential/quota failure. Credential, quota and rate failures are
injected; the probes do not deliberately exhaust a real account or trigger throttling.

The public client does not issue arbitrary malformed JSON, item retrieval or out-of-band
response requests. Their observed envelopes are decoder fixtures, rather than additional
DST transport features. The table is a bounded compatibility audit, not an exhaustive list
of every invalid parameter combination or every server/internal failure.

## Asynchronous transcription and identities

Input transcription is independent of response generation. Two OpenAI audio probes observed
final input transcription 127ms and 1083ms after `response.done`. The DST varies partial/final
timing independently, reverses final order across turns, fails an older ASR job after a newer
answer, and races transcription with reconnect and SQLite history checkpoints. A committed
write can lose its acknowledgement; the checkpoint is retried without duplicating items.
A refused storage checkpoint produces an explicit storage failure.

The xAI profile exercises cumulative transcript revisions and UUID-shaped native item IDs.
It is based on the documented protocol and deterministic simulations; no direct xAI key was
available for a live xAI probe. Importing a foreign history into a new OpenAI session assigns
legal wire aliases to item and call IDs, reverses acknowledgements/events to the original
application IDs, and retains those bindings across reconnects. An originating provider's
native IDs remain unchanged on that session. Tool calls and outputs share one call alias.

## Requests, handles and update failures

The request regressions inspect the actual `SpeechHandle` returned by `generate_reply`, not
only session history. They verify that a queued reply does not inherit another request's
creation deadline, an expired request's late acknowledgement cannot acquire its successor's
handle, cancelled waits do not cancel speech, and tool follow-up responses stay on their
original handle while another speech is queued. Each case checks that another reply succeeds.

| Provider path | Request coverage |
| --- | --- |
| OpenAI, xAI, LiveKit Inference, Azure Voice Live | Real dialect machines and voice executors in the DST; timeout and late-response isolation |
| OpenAI native protocol | Public `VoiceSession`, rejected option update, distinct handles and subsequent successful reply over a local wire peer |
| Phonic, Ultravox, Gemini, Nova Sonic | Public `VoiceSession` response-to-handle mapping; provider Future timeouts and subsequent turns |
| Phonic, Ultravox, Gemini | Terminal WebSocket failure before generation and after partial output releases request waiters and streams |
| Nova Sonic | The same terminal-failure contract over the fake Bedrock duplex stream; no-instructions waiters are tracked and bounded |
| PersonaPlex | Manual `generate_reply` explicitly raises `NotImplementedError`; readiness and close are checked separately |
| OpenAI GPT Live duplex | Its existing command/session lifecycle, delegation, response failure and close suites; it has a different API from a plugin reply Future |

Python request timers are fired explicitly using their real callbacks. Tests also deliver a
stale callback after a later Future exists. No request-timeout regression sleeps for the
provider's timeout duration. Terminal failures settle rather than pretending that a closed
connection can answer; recoverable update/refusal cases must permit another turn.

This audit fixed untracked Nova Sonic waiters, duplicate/premature response announcements,
reentrant generation callbacks binding the wrong Future, missing Ultravox response IDs,
Gemini turn rejection reasons being dropped by decoding, dangling streams after terminal
provider failures, and host update failures that were only logged. A host update failure is
reported without assigning it to an unrelated reply request.

Provider peers are local and deterministic. This request/handle coverage does not constitute
live verification of every service, and existing strict expected failures in the broader
provider suites remain tracked by their issue links.

## Switching API audit

The provider-switch DST runs the production fallback adapter through xAI → OpenAI → xAI.
It verifies real SQLite history, tool-call/output pairing, legal destination wire IDs and
stable public IDs. A separate case replaces the provider while a completed user turn waits
for ASR and a delayed response: the replacement answers once and the accepted partial is saved
before that answer. The retired provider's final ASR events must drain before context replay.

The agreed behavior for explicit switches is to finish a reply whose playout has started;
a reply that has played nothing can be cancelled. A completed user turn waiting for an answer
belongs to the replacement exactly once. A still-speaking user's utterance must remain intact.
Temporary selection belongs on `override_options(llm=...)`, with restoration on block exit;
a handoff must also be able to select another realtime provider.

API inspection found a separate implementation gap: this checkout exposes no `llm` argument
on either session options method, and a handoff updates the existing realtime connection.
The fallback-switch tests do not establish that explicit override/handoff replacement exists.

## Phonic

The checked-in Phonic plugin connects to `wss://api.phonic.ai/v1/sts/ws`. A live probe using
its exact configuration received `ready_to_start_conversation`, then a generated reply with
`assistant_started_speaking`, `audio_chunk` and `assistant_finished_speaking`.

This endpoint uses Phonic's `config` / `generate_reply` protocol. Adding the OpenAI `event_id`
field to its configuration was rejected. Sending `response.cancel` produced
`user_message_validation_failed`. The candidate `/v1/realtime` route returned HTTP 404.
The SDK's shared `RealtimeModel` interface and OpenAI-shaped tool schemas do not establish
wire compatibility. Phonic peer tests must use its own protocol.

## Sources

- [OpenAI realtime client events](https://developers.openai.com/api/reference/resources/realtime/client-events)
- [OpenAI realtime server events](https://developers.openai.com/api/reference/resources/realtime/server-events)
- [OpenAI realtime transcription](https://developers.openai.com/api/docs/guides/realtime-transcription)
- [xAI voice API](https://docs.x.ai/developers/rest-api-reference/inference/voice)
- [Gemini turn completion reasons](https://googleapis.github.io/js-genai/release_docs/enums/types.TurnCompleteReason.html)
- [Nova Sonic output event ordering](https://docs.aws.amazon.com/nova/latest/nova2-userguide/sonic-output-events.html)
- [Phonic conversations API](https://docs.phonic.co/api-reference/conversations/conversations)

## Plugin input audit

The incoming-message audit scans every Python plugin package, including STT, TTS, LLM,
realtime, duplex and avatar adapters. Shared JSON decoding skips unknown tags and ignores
extra fields, while validating fields a handler actually reads. The source rules reject
closed literal fields for provider-extensible values and now reject `assert_never` on incoming
message unions, including tag-narrowed unions. Exhaustive matches on framework input items
and local configuration remain valid internal checks.

The audit replaced 39 incoming `assert_never(message)` branches in 37 files with safe skips,
including handlers behind custom decoders. Unexpected exceptions from receive handlers and
multiplexer routing become typed, non-retryable protocol errors through the existing failure
and cleanup path, retaining request identity and private diagnostic detail. They are never
silently swallowed after possibly mutating stream state. Cancellation remains cancellation.
Unknown messages alone do not fail a stream. Malformed known messages may fail the affected
stream; this contract does not promise recovery from arbitrary corrupted protocol state.

Anthropic tool assembly now tracks content blocks by index. A foreign or unknown block's
JSON delta or stop cannot modify or prematurely emit another block's tool call. The regression
runs through the installed provider SDK and public chat stream, including an unknown SSE event.
Shared boundary regressions cover custom decoder additions, unexpected handler exceptions,
request attribution and routing failures. The full plugin suite uses local peers and mocked
SDK transports; it does not establish live compatibility with every hosted provider.
