# Binding conformance: S0 skeleton

Run from the repository root against the next package and its built native extension:

```sh
PYTHONPATH=livekit-agents-next:. uv run --no-sync pytest tests/conformance --unit
```

`cases.json` contains language-neutral input/outcome tables. Non-finite floats and the
omission sentinel use string encodings (`nan`, `inf`, `NOT_GIVEN`). The pytest module
drives real `_core` entrypoints; it adds no product exports or alternate conversions.
The B-3/B-4 null, bool-as-int and invalid-duration cases now pass through the production
ABI registrations, so their five strict xfails have been removed. The buffer cases also
exercise mutable input after handoff and the documented `AudioByteStream` input types.

The matching Rust cases live in `lk-agents-abi/src/python/conversion_tests.rs` and the
poisoned-sink regression in `python/node.rs`. Run them with the Python-feature ABI suite
under nextest (one interpreter process per test). They cover both conversion directions,
lookup exception identity/cause, typed map keys and nested values, per-instance defaults,
Unicode/surrogates, float ranges, newtype validation, stale/foreign references, invalid
media elements, immutable retained bytes, writable aliases, i16le layout, and poisoned
generated objects. ABI fixtures model SDK buffers; the pytest cases use the installed SDK.

| Family | Checked behavior |
| --- | --- |
| Presence | Generated omission/null rejection; native nullable record round trips and lookup failures |
| Scalars | Unicode, bool/int separation, float ranges and newtype validation |
| Time | Invalid playback reports fail before settlement; native durations reject invalid ranges |
| Collections | Typed map/record paths, snapshots and per-instance defaults |
| Buffers | Immutable retention, writable aliases, i16le layouts, owner release and stream inputs |
| References/panics | Stale/foreign failures; generated mutex and sink refuse poisoned state |
| Errors | Domain class, argument detail and unchanged state on rejection |
| Cancellation | Before start/during wait, idempotence, late completion and cancelled waiters |

The bootstrap and telemetry modules are ordinary G-8/G-9/J-5 regression tests.
