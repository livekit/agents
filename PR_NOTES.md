# PR notes — livekit/agents #7597 (+ #7602 companion)

## Branch
`fix/silero-vad-update-options` @ `fa9a995` — local, NOT pushed, NOT opened as PR (orchestrator ships).

## Diff summary (3 files, +135/−1)

1. **`livekit-plugins/livekit-plugins-silero/livekit/plugins/silero/vad.py`**
   - `VAD.update_options()` (~L196): when `activation_threshold` is given and `deactivation_threshold`
     is NOT, re-derive `deactivation_threshold = max(activation_threshold - 0.15, 0.01)` — the exact
     derivation `VAD.load()` uses. Explicit `deactivation_threshold` still wins (applied after).
   - `VADStream.update_options()` (~L300): same re-derivation.
   - NEW `VAD.with_options()` (~L115): returns a new VAD reusing `self._onnx_session` with options
     derived `load()`-style (addresses #7602 — session build is expensive + holds the GIL; session is
     safe to share since each stream() allocates its own OnnxModel state).
   - Docstrings updated to document the re-derivation rule.
2. **`livekit-agents/livekit/agents/inference/vad.py`**
   - Same fix in `VAD.update_options()` (~L108) and `_VADStream.update_options()` (~L168) — this file
     copied the same bug (called out in issue #7597). No `with_options` here (native session sharing
     differs; keeping scope minimal).
3. **`tests/test_silero_vad_options.py`** (new, 84 lines, marked `pytest.mark.plugin("silero")`)
   - `test_update_options_rederives_deactivation_default` — activation=0.7 → deact≈0.55 (was 0.35)
   - `test_update_options_low_activation_keeps_hysteresis` — activation=0.2 → deact=0.05 (was 0.35, i.e. inverted hysteresis)
   - `test_update_options_explicit_deactivation_wins` — both given → explicit deact kept
   - `test_update_options_deactivation_only_leaves_activation` — deact-only → activation untouched
   - `test_update_options_propagates_to_open_stream` — open stream gets same derivation (stream closed in finally to appease conftest teardown)
   - `test_with_options_shares_session_and_derives_defaults` — session identity shared, defaults derived, original VAD not mutated

## Test evidence

```
$ python -m pytest tests/test_vad.py tests/test_silero_vad_options.py --plugin silero -q
14 passed, 1 warning in 21.93s
```
- Full pre-existing `tests/test_vad.py` (8 tests, includes real ONNX inference) still passes with the fix.
- All 6 new tests pass. Bug reproduced BEFORE the fix (deact stayed 0.35 after activation=0.7; and
  0.35 > 0.2 inversion case), gone AFTER.
- Env: Windows, Python 3.11.8, `pip install -e livekit-plugins/livekit-plugins-silero` + livekit-agents 1.8.x from PyPI.
  Note: repo deps needed `jiwer` installed locally for test_vad.py collection.

## Suggested PR

**Title:** `fix(silero): keep activation/deactivation threshold pair consistent in update_options`

**Body:**
Fixes #7597, addresses #7602.

`VAD.load(activation_threshold=x)` derives `deactivation_threshold = max(x - 0.15, 0.01)` when not
given. But `VAD.update_options()` / `VADStream.update_options()` replaced only `activation_threshold`,
keeping the previously derived deactivation threshold. Consequences (from the issue):

- `update_options(activation_threshold=0.7)` leaves `deactivation_threshold=0.35` — speech ends later
  than configured; `load(activation_threshold=0.7)` would give 0.55. Same options, different VAD.
- `update_options(activation_threshold=0.2)` leaves deactivation **above** activation: the exit
  threshold can never take effect (inverted hysteresis).

This changes the pair consistently in both the silero plugin and the `livekit.agents.inference.VAD`
copy, re-deriving the paired default only when `activation_threshold` is given without an explicit
`deactivation_threshold` (explicit values always win, as before).

Companion (#7602): adds `VAD.with_options(...)` — build a VAD reusing the existing ONNX session with
load()-style option derivation. The session is safe to share (each `stream()` allocates its own
`OnnxModel` state); this avoids the per-call `InferenceSession` build that holds the GIL (~15 ms
measured in the issue). Left `inference/vad.py` without a `with_options` twin to keep the diff
minimal — happy to add it if maintainers want symmetry there too.

Tests: new `tests/test_silero_vad_options.py` (6 tests, no real ONNX session needed for the
options-logic); existing `tests/test_vad.py` passes unchanged (8 tests).

## Risks / reviewer notes
- Behavior change is the fix itself: anyone who relied on update_options keeping the stale derived
  deactivation will now get the load()-consistent pair. That inconsistency is the bug; flag in changelog.
- `with_options` returns a new object (not in-place) — matches "build a VAD with new options on an
  existing session" framing in #7602; alternative API shape (process-level session cache) was option 2
  in the issue and is intentionally out of scope here.
- CLA: livekit requires CLA? — not checked in CONTRIBUTING.md by this run (ultralytics does; livekit
  PRs are DCO/CLA-light — verify when opening).
