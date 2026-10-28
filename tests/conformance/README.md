# Binding conformance: S0 skeleton

Run from the repository root against the next package and its built native extension:

```sh
PYTHONPATH=livekit-agents-next:. uv run --no-sync pytest tests/conformance --unit
```

`cases.json` contains language-neutral input/outcome tables. Non-finite floats and the
omission sentinel use string encodings (`nan`, `inf`, `NOT_GIVEN`). The pytest module
drives real `_core` entrypoints; it adds no product exports or alternate conversions.
Only demonstrated, scope-cited defects are strict xfails. An unexpected exception type
still fails, and an XPASS requires removing the xfail when its owning slice lands.

| Family | Current bridge and assertions | Later closure |
| --- | --- | --- |
| Presence | Generated ExpFilter arguments: omitted/sentinel/zero/null; invalid update preserves state | B-3: reject invalid null, attribute/item lookup failures, nullable updates and reverse bridge |
| Scalars | Generated results/options/filters: Unicode, empty string, bool/int separation, double precision, unsigned range | B-4: reject bool for int, remaining numeric ranges/newtypes and reverse bridge |
| Time | PlaybackTracker reports and native await: seconds, fractional precision, invalid sink reports before settlement | B-4: refuse invalid durations with field paths; generated secs_from_py inputs, timestamps and reverse bridge |
| Collections | Generated list conversion and native result: order, empty, invalid elements, snapshot | B-5: typed maps, nested error paths, records/default factories and reverse bridge |
| Errors | Generated constructor/method exceptions: domain class, argument detail, unchanged state on rejection | B-8/B-14: host failures with type/value/cause in both directions |
| Cancellation | Native future cancelled before start/during wait; idempotence, late completion, operation survives a cancelled waiter | B-10/B-12: host stream cancellation, owned-task settlement, cleanup and every runtime adapter await point |

This is explicitly the B-19 S0 skeleton, not final conformance sign-off. Records/variants,
buffers, calls, callbacks, streams, notifications and resources need their own complete
tables and both bridge directions as the shared generated adapters land. No required
future family is represented by a skipped or unconditional-xfail placeholder.

The bootstrap and telemetry modules exercise the separately assigned G-8/G-9/J-5 Python
regressions. Notebook tests replace the display and network boundaries, so they need no
IPython installation or credentials. The metric regression drives the real OTel SDK across
64 jobs through usage, turn, span-duration and loop-blocking measurements. Measurement
labels allow only the two finite connection-reuse and severity values; unknown values and
unique model/tool/job attributes cannot grow the aggregation table or leak content.
These are ordinary unit tests, not claims of final closure for those items.
