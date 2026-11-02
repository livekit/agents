# Rust and Python type simplification inventory

Preserve familiar domain names for responsibilities that remain. Remove names that only hide
an ordinary language type or duplicate another name. Age alone does not determine usefulness;
a surviving type must express a distinct contract, ownership rule, or invariant.

The original candidate inventory was based on the 2026-11-01 working tree. The plan audit now
separates changes present in source from work still to do. The [transition plan](rust-transition-inventory.md)
owns sequencing; this page owns type-removal evidence and affected verification.

## AudioBuffer: removed in the working tree

The former aliases had no buffering implementation:

- Rust: `AudioBuffer<'a> = &'a [AudioFrame]`.
- Python: `AudioBuffer = list[rtc.AudioFrame] | rtc.AudioFrame`.

Both names are absent from current handwritten production sources. Rust uses a frame slice;
Python boundaries spell out the accepted single-frame/list types. No compatibility alias is
planned. This removes a name without removing batch recognition or changing accepted values.

Source owners: [Rust audio helpers](../../rust/crates/lk-agents-abi/src/utils/audio.rs),
[Python audio helpers](../../livekit-agents-next/livekit/agents/utils/audio.py),
[STT contract](../../rust/crates/lk-agents-abi/src/stt/stt.rs), and [provider docs](../models.md).

## Reviewed candidates and current status

“Present in source” is not the same as validation complete. The Rust architecture/core/runtime
checks recorded in the shared-policy audit do not by themselves validate every plugin's Python
annotations or single-frame/multiple-frame behavior. Keep those affected checks explicit.

| Type or name | Working-tree disposition | Remaining verification or decision |
| --- | --- | --- |
| Rust `AudioBuffer` | Removed alias/re-export; direct slice in production sources | Recognition call shapes and generated declaration drift |
| Python `AudioBuffer` | Removed alias/imports; explicit accepted frame types | Provider annotation checks and single/list recognition behavior |
| Python `AudioArrayBuffer` | Removed implementation and its dedicated tests | Confirm removed public surface matches the authorized cleanup; ensure active resampling paths remain covered |
| `combine_frames` | Removed unused convenience alias | Check public docs/exports and use direct RTC helper where needed |
| `merge_frames` | Removed alias; callers use direct RTC operations | Validate affected utilities/plugin annotations and documented exports |
| Rust `LegSet` | Removed synonym; `SessionLegs` remains | No new wrapper; preserve option-layer behavior |
| Rust `WorkerRegistered` | Removed synonym; `RegisteredInfo` remains | Notification/callback schema and generated drift checks |
| `AudioByteStream.write` | Removed declared synonym; `push` remains | Affected emitter/decoder callers and binding call contracts |
| Media `Frame` versus ABI `AudioFrame` | Retain distinct records after ownership review | The chunker/emitter owns mutable `Vec<u8>` chunks and splits/trims PCM before publishing; ABI `AudioFrame` carries sample rate/channel metadata and shared `Arc<[u8]>` for fan-out. Converting at publication avoids putting shared immutable ownership into the framing work. These are not aliases. |
| `SttSink`, `SttSource`, `TtsSink`, `TtsSource` | Retain meaningful protocol endpoint names | Reassess only if ordinary signatures are clearer without them |
| `Maybe<Option<T>>` and `NotGiven` | Retain omitted versus explicit-null distinction | Replacing with `Option<T>` would change update/binding behavior |
| `AudioByteStream` | Retain actual framing/buffering implementation | Preserve partial samples, sizing, flush, and clear contracts |

The removal statuses were checked with exact-name searches across handwritten Rust, public
Python, and provider plugins, plus the owning ABI/media byte-stream declarations. Provider
protocol names containing `AudioBuffer` as a substring are unrelated and remain valid.
Generated output is not a declaration source and must never be edited manually.

Additional source owners:

- [Session options](../../rust/crates/lk-agents-abi/src/voice/session_options.rs).
- [Worker records](../../rust/crates/lk-agents-abi/src/worker/worker.rs).
- [Byte framing API](../../rust/crates/lk-agents-abi/src/utils/byte_stream.rs) and
  [media implementation](../../rust/crates/lk-agents-media/src/audio_byte_stream.rs).
- [Omission semantics](../../rust/crates/lk-agents-abi/src/utils/maybe.rs).

## Verification checkpoint

The affected Python tests pass: 64 audio byte-stream/emitter/decoder cases, 30 room audio
adapter cases, and two STT fallback cases covering both a single frame and a frame list. The
last pair exercises batch recognition through the actual fallback/adapter boundary, preserving
the accepted values without the removed alias. Ruff lint/format checks pass for the extended
STT regression.

The final strict mypy rerun checked 739 public/plugin modules and reports 30 existing
diagnostics in 16 files, primarily API/stub contract mismatches. Explicit adapter keyword
arguments removed nine diagnostics from the earlier checkpoint; these fixes introduce no new
diagnostics. The full type gate is not passing. No missing `AudioBuffer` annotation/import
was reported. Exact-name searches, runtime tests and generated binding checks establish
different guarantees; none substitutes for the others.

## Cleanup acceptance

Preserve the source removals and affected frame-shape/PCM regressions; do not schedule removing
the same names a second time or call a source search behavioral proof. Generated declarations
and fresh Cargo-linked binding checks remain part of the main transition verification gate.
Distinct representations were retained where ownership and PCM conversion costs are concrete.
No new convenience aliases or compatibility wrappers are required by this inventory.

The [file cleanup inventory](rust-file-layout-inventory.md) covers physical source consolidation.
Moving correlation IDs, recorder facts, completion helpers or voice-run records beside their
component does not remove those types or combine their lifetime contracts.
