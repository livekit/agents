# Development and review

Local builds, automatic checks, code conventions and implementation review.

[Contributor index](README.md) · [Execution paths](architecture.md#execution-paths)

## Local builds

From the repository root, `make test` runs the hermetic tests. Set `TEST_ARGS` to select
paths and categories, for example `make test TEST_ARGS="python/tests/test_it_ipc.py --e2e"`.
`make type-check` runs both type checkers; `TYPE_CHECK_ARGS` selects a checker or its options.
`make bench` runs standalone measurements described in the [benchmark guide](../../bench/README.md).

Run `make install` once, then `make build` to rebuild the Python extension in
place without installing dependencies again. Python-only changes use the editable
sources immediately and do not need this rebuild. `make -C rust python` is the same
command. `make -C rust build` builds every workspace package and FFmpeg.
Editable installs made by `uv sync` also use the dev Cargo profile, so Rust source
changes do not trigger an optimized release rebuild through the packaging backend.
Published wheels retain Maturin's release default.
`make rust-check` (or `make -C rust check`) compiles and lints every
workspace crate and test target with FFmpeg enabled, and enforces boundaries and
generated output ([enforcement](#enforcement)). `make -C rust test-checks` runs the
`rust/check` and codegen tests; CI and the launch gate run both.
`MATURIN` and `VIRTUAL_ENV` can select an existing tool and virtual environment.
Cargo and Maturin use that environment's Python; `PYO3_PYTHON` can override Cargo's
interpreter explicitly. Unix environments use `bin/python`; Windows uses
`Scripts/python.exe`. The Make targets require GNU Make; alternatively activate
the environment and run `maturin develop --skip-install --locked` directly.

For fast feedback while editing Rust, check the affected crate first:

```bash
cd rust
cargo check --locked -p lk-agents-voice
```

This checks types without generating machine code or linking the Python extension.
A warm voice-only check took 0.6–0.9 s in the local Apple Silicon trial, compared
with roughly 7 s to rebuild the extension after the same edit. Rebuild with
`make build` to exercise Python integration, and use the full `make rust-check`
gate and relevant tests before finishing a change.

Native builds require a C/C++ toolchain (Xcode command-line tools, a Linux
compiler toolchain, or MSVC build tools). Building FFmpeg from source on macOS
or Linux also requires Bash, Make, pkg-config, and the Opus development library.
For Windows and cross compilation, provide a target-compatible static FFmpeg
installation through `LK_CODECS_FFMPEG_PREFIX`; Windows archives must be built
with FFmpeg's MSVC toolchain. The compatible `rust-sdks` sources live beside
this checkout, as declared by the workspace's relative Cargo paths.

The repository-root `.cargo/config.toml` supplies the same compiler flags when
Cargo runs from the root, `rust/`, or the Python package. It explicitly selects
`split-debuginfo=off` on macOS so Maturin does not force `.dSYM` generation.
Other targets retain their native format: Linux uses embedded debug info and
MSVC uses supported packed `.pdb` symbols. Our
runtime and decision crates retain the dev profile's line tables; dependency and
binding-glue debug info stays off on the fast path.
Our C/C++ adapters and vendored native libraries omit debug information on the
fast path. This keeps their archives independent of the checkout's debug paths;
Rust runtime and decision code retains its debug information. Set
`LK_AGENTS_NATIVE_DEBUG=1` to restore native symbols (`0` or unset selects the
fast path). Cargo tracks this setting and the native cache key includes its
effective compiler flags.
Debug native cache entries also include the source checkout, so their symbols
refer to that checkout; fast entries remain reusable across worktrees.
`make build-debug` (or `make -C rust python-debug`) sets that flag, restores
binding-glue line tables, and uses the platform's native debug format
(macOS `.dSYM`, Windows `.pdb`, or embedded debug info on Linux). Switching
that mode changes compiler flags and
therefore rebuilds its Rust artifacts.

The default Xcode linker already uses `ld_new`. An Apple Silicon extension-link
replay with Xcode 27.2 measured warm medians of 0.34 s for the default linker and
0.78 s for LLD 23.1.4 over three runs of identical inputs. Explicit `-ld_new`
produced the same binary as the default. Keep the system linker for this setup;
Rust 1.98's bundled LLD 22.1.8 also cannot parse that SDK's `arm64e.x1` stubs.

FFmpeg and the vendored BlingFire library use a persistent native artifact cache,
independent of Cargo's target directory and Rust-only compiler flags. Cache keys
include source contents, native build recipes, target, compiler version and
flags, and relevant toolchain configuration. FFmpeg also keys on the Opus
configuration. Our small native adapters still compile in Cargo's output tree.
An OS file lock serializes each cache entry. Builds use unique staging directories
and publish by atomic rename after validating their outputs; failed or interrupted
builds cannot overwrite a published entry. FFmpeg uses `DESTDIR` so cached
pkg-config files contain the final installation prefix.

The default cache is `~/Library/Caches/livekit-agents/native` on macOS,
`$XDG_CACHE_HOME/livekit-agents/native` (or `~/.cache/livekit-agents/native`) on
Linux, and `%LOCALAPPDATA%/livekit-agents/native` on Windows. Set
`LK_AGENTS_NATIVE_CACHE_DIR` to an absolute directory to relocate it. Removing
that directory forces a native rebuild. Missing or unusable default cache
locations fall back to Cargo's output directory; an invalid explicit override
reports an error. `LK_CODECS_FFMPEG_PREFIX` continues to
select an externally built FFmpeg, including for cross compilation. That
installation supplies its own debug configuration.

#### Parallel worktrees

Keep Cargo's default private `rust/target` in each worktree. A globally shared
`CARGO_TARGET_DIR` makes concurrent commands wait on Cargo's build-directory
lock; unset that override when running independent worktrees. Shared compiled
artifacts belong in a compiler cache, separate from each worktree's mutable
Cargo state. The native artifact cache above is already shared.

Make uses [mr-boxington](https://github.com/jdx/mr-boxington) by default for
Rust build and check commands. Install it once:

```bash
cargo install --locked mbx
make rust-check
make -C rust build
```

If mbx is missing, Make stops with installation instructions.
Use ordinary Cargo explicitly with `CARGO_RUNNER=cargo`. `CARGO_RUNNER` selects
one executable, including an absolute path containing spaces. Direct invocations
of the architecture script still default to Cargo. The first invocation fills
the cache. Equivalent work in another checkout can then reuse it; changed inputs
still compile.

Keep the toolchain, features, compiler flags, and native dependencies consistent
to maximize reuse. A warm `check` cache does not replace warming a `build` cache.
mr-boxington manages its own platform cache location and CPU/memory budget; no
repository path or machine size is hardcoded here. Its `explain --last` command
reports cache misses.

Maturin invokes Cargo internally. To cache Python extension builds as
well, use the tool's documented Cargo-shim setup, then keep using `make build`
and `make build-debug`. This is a developer choice; the repository does not
install a global shim during a build. Full Python extension builds may still
perform native linking.
Restored Rust debug information can name the producing checkout, as documented
by the cache tool; an uncached debug build is preferable when stepping through
code in a different checkout.

Other options include [Kache](https://github.com/kunobi-ninja/kache), which also
supports private worktrees with a shared artifact store, and
[cargo-reapi](https://github.com/TamedTornado/cargo-reapi), which caches library
and metadata compilations while leaving native link steps local. Read the
documentation for the release you install: Kache's current main-branch docs
include v2 preview behavior. A macOS/Apple Silicon trial with Rust 1.98 and Kache
2.0.0-rc.3 restored the extension into an empty target quickly, but repeated edits
rebuilt the generated Python binding without incremental reuse. Kache's adaptive
incremental policy requires Cargo's `extra-filename` unit ID, which this extension
does not have. mr-boxington remains the default for local extension development;
rebenchmark Kache's handling of this case before switching.
[Worktrunk](https://worktrunk.dev/) manages worktrees and can seed their ignored
build directories; it is a separate layer from compiler-action caching.
[Epicenter](https://github.com/TheAnalist/epicenter) is another approach: it
seeds private job directories from stored Cargo build state instead of wrapping
individual compiler actions. cargo-reapi, Worktrunk and Epicenter have not been
benchmarked on this workspace.

## Enforcement

The rules and how each is enforced live in one list: the "Rust architecture rules" section of
[`AGENTS.md`](../../AGENTS.md). `make -C rust check` runs every automatic one: `cargo fmt`, Clippy
with `[workspace.lints]` and the two policies in `rust/check/` (`purity/clippy.toml` for ABI, core,
voice and media, where media's thread pool `pool.rs` and `media_queue` are the only exempt
modules; `io/clippy.toml` for the crates beside the host), the `rust/check` crate (crate
graph and banned dependencies in `graph.rs`; binding placement, executor modules, `Unacknowledged::sent` callers, `select!`
order, generated markers and file length in `sources.rs`) and the generated drift checks
(`lk-agents-codegen --check`, `thread_api.py --check`). No rule has a baseline: a violation is
fixed, or one site carries `#[allow(lint, reason = "...")]`. Each rule's tag in AGENTS.md says
which of these enforces it, or that review does.

CI runs these rules and the native package smoke test in
[`ci.yml`](../../.github/workflows/ci.yml). Rust test suites and the DST run separately through
[`launch-gate.yml`](../../.github/workflows/launch-gate.yml); Python tests use
[`tests.yml`](../../.github/workflows/tests.yml). The launch gate uses bounded seed campaigns
for PRs and larger campaigns for scheduled or manually triggered runs.
## Codebase rules

### Rust-first implementation

- Implement shared framework behavior once in Rust: defaults and option
  resolution, validation, lifecycle policy, stream adaptation, transforms,
  protocol handling, media accounting and built-in provider/I/O behavior.
- Python and Node expose generated APIs and thin host adapters. Handwritten host
  code serves application callbacks, host SDK objects, event-loop integration
  or language-specific process launch/bootstrap.
- A native implementation is consumed through either binding. Binding code
  converts types and invokes the host; it does not reproduce domain behavior.
- Extend ABI declarations and generation when a host needs an existing Rust
  capability. Do not port the implementation into Python or JavaScript to work
  around a missing export or stream bridge.
- User plugins and callbacks may be authored in their host language. Their
  custom behavior remains there; shared framing, stream lifecycle, cancellation
  and error adaptation remain in Rust and the generated bridges.

### Names and layout

- Use the specified public API names. For internal concepts, preserve the
  Python reference's names and module grouping, adapted to Rust conventions:
  `AgentActivity`, `SpeechHandle`, `schedule_speech`, `drain`, `handoff` and
  `dtmf`. Do not rename an existing concept merely because it moves to Rust.
- Introduce a new concept or name only for a distinct responsibility. Reuse
  the domain's vocabulary rather than adding generic framework terminology.
- Name events, effects, decisions, verdicts and outcomes for the domain or operation
  whose protocol they describe, even inside a clearly named module. Prefer
  `AvatarEvent`, `VoiceEffect`, `AgentRunEffect` and `UserTurnCommitDecision` to
  bare `Event`, `Effect` or `Decision`. The name must retain its meaning in imports,
  signatures, matches and diagnostics; do not alias it back to a generic name.
  Prefer the existing owner name with the appropriate suffix; do not stack qualifiers that
  add no distinction or invent a generic owner for hypothetical reuse.
  Use the narrowest meaningful subject: a user-turn commit decision is more precise
  than `VoiceDecision`. Preserve established public Python names where they already
  identify the right domain.
- Scope execution-owner names the same way: use `VoiceDriver` and `VoiceRuntime`
  for the voice decision coordinator and its I/O executor, rather than bare
  `Driver` and `Runtime`. Preserve the familiar role names instead of introducing
  another synonym for the same responsibility.
- State each protocol's producer, handler and state owner beside its declaration.
  The domain prefix identifies the protocol; it does not mean the producer owns
  the receiving component's state. Distinguish core inputs, runtime execution
  envelopes and public notifications when their contracts differ; do not give
  all three the same event name or merge them solely for naming consistency.
- Reduce primitives before renaming them. Prefer ordinary domain methods, return
  values and an existing effect collection mechanism to generic dispatch wrappers.
  Keep an additional coordination type only for a concrete ordering, ownership or
  lifecycle obligation, and document the bug it prevents. Clearer names alone do
  not justify a type, an event/effect pair, or another collector.
- Keep effect storage private to its coordinator. Domain components expose ordinary
  operations and return meaningful outcomes; do not require every component or helper
  to accept a shared mutable effects collector. The coordinator translates outcomes
  into runtime effects and owns any required cross-component deferred callbacks.
  Preserve ordered multi-outcome contracts when removing existing transition plumbing.
- Coordinators own advancement between complete domain transitions. A handler must not need
  a manual scheduling pump, dirty flag, or schedule callback to make its changes take effect.
  For voice, schedule after the complete input/callback or ordered waiter batch, never inside
  an atomic handoff swap or close edge. Admission pauses allow accepted speech to drain;
  authorization holds block starting queued speech; the close hard edge forbids authorization.
  Preserve FIFO callbacks, same-step progress, and timer cancellation ordering.
- Current state determines scheduling eligibility; FIFO callbacks determine transition order.
  The scheduler may inspect its speech queue to select eligible work. Do not scan future
  callbacks for pause, handoff, or close intent to decide eligibility at an earlier checkpoint.
  Each complete transition installs the authoritative state before automatic advancement.
- Make matching lifecycle milestones recognizable across owners and execution boundaries.
  Effects request work with imperative verbs; observations identify what happened. Prefer
  shared completion handling over duplicate per-operation callbacks. Add a mirror event or
  effect only for an actual consumer or ordering obligation, never for naming symmetry alone.
  Keep input EOF, operation results, playout settlement, and resource retirement distinct;
  document intentional asymmetry beside its owner.
- Treat symmetry as a general rule across all Rust crates: methods, events, effects, callbacks,
  states, and ownership lifecycles. Audit operation, result, and state symmetry separately.
  Matching operations
  should use matching names (`pause_scheduling` / `resume_scheduling`, `open` / `close`),
  with the same subject and verb convention. For every reversible operation, identify its
  inverse, owner, allowed states, and completion semantics; document irreversible operations
  explicitly. A matching method does not require a matching callback: pause may await a drain
  while resume completes synchronously. Never add a no-op notification just to fill a pair.
  Apply this to every lifecycle change; scheduling and close are only examples.
- Name completion callbacks for the obligation that ended, including refused/cancelled results;
  do not imply success with names such as `Paused` when refusal is a valid payload. Separate
  admission pauses from playback/authorization holds. Close's admission pause is terminal and
  must not be lifted by an ordinary resume. Prefer coordinator-owned advancement of close from
  current phase/prerequisite state to a generic progress callback or handler-maintained dirty
  flag. Preserve deferred ordering before removing any existing notification.
- Keep internal coordination inside its owner. Do not turn “run my logic again” into a
  cross-component callback or effect such as `CloseProgress`, `Pump`, or `Recheck`. Prefer
  automatic advancement from authoritative state at complete transition boundaries. Callbacks
  carry concrete domain facts or required deferred completion; effects request work from an
  executor. Real timer/protocol inputs remain valid. Retain a generic progress notification
  only for a documented ordering obligation, and remove it when that obligation is represented
  by the coordinator. Do not substitute manual pump calls, dirty flags, or new tracking types.
- Use deadline vocabulary for timer operations: `set_deadline(timer, at)` replaces a deadline,
  `clear_deadline(timer)` removes it idempotently, and `deadline(timer)` reads it without side
  effects. Clearing a deadline does not cancel the work it bounds. Timer state and necessary
  ordering belong inside their owner; components must not construct generic deferred
  instructions such as `StepOp` or learn a renamed replacement. Resolve runtime requests at
  their required transition boundary while keeping that bookkeeping private.
- Keep domain modules recognizable (`generation`, `room_io`, `transcription`, `stt`, `tts`,
  `llm`). Put synchronous effect-producing owners in the domain's `machine` module and their
  effect declarations in its category `effects.rs`. For example, `generation::machine::AudioPlayback`
  produces `generation::effects::AudioPlaybackEffect`. Keep meaningful owner filenames/types,
  without `_machine.rs` or `StateMachine` suffixes. Link each effect declaration to its producer
  and executor, and map owners in `machine/mod.rs`. Plain data/helpers remain outside; private
  child state stays private. No common base trait or registry is required. Document each
  machine's states, inputs/time, effects, and terminal behavior.
- Downstream code must not manually pump accepted work. Starting an operation, submitting
  input, and awaiting/reading results is sufficient; its coordinator/executor processes
  transitions, effects, I/O outcomes, and deadlines automatically until blocked or terminal.
  A pure core still accepts explicit inputs/time, driven by its executor. Name a long-running
  async owner an executor (for example `SttStreamExecutor`), rather than exposing a `Pump`
  concept or a per-handler obligation to call `pump`/`advance`/`schedule`. Preserve legitimate
  waiting/backpressure and the agreed atomic transition boundaries; do not spin or run forever.
- Keep effect declarations in dedicated category `effects.rs` files inside their owning crate.
  Group by lifecycle/domain (generation, room I/O, transcript synchronization, provider TTS),
  not one file per enum or a global workspace catalog. Keep state machines and dispatch code
  beside their owners, outside declaration files. A crate-level effect index can re-export
  scoped public types and show an ownership map in one place; it adds no wrapper enum and
  exports no private submachine protocol. Distinguish actual nesting from projection and
  independent execution in that map.
- Keep one policy owner for the same domain decision. Coordinators consume canonical outcomes
  and supply their own constraints; they do not independently select terminal outcomes or
  rebuild authoritative records from the same raw inputs. Share ordinary conversion helpers
  where semantics match, while preserving distinct protocol, presentation, and cleanup facts.
- Keep the invocation loop generic to conversation/delegation question orchestration. Core
  reports correlated input requirements; it does not discard pending expert questions,
  synthesize answers for unselected requests, or decide their presentation order. The live
  conversation owns the foreground question and answer routing; delegation owns backend
  task/request continuation. Integrate the question behavior separately from core cleanup.

- Repeated subject names across ABI, decision and execution crates identify real boundaries.
  Private helpers need not mirror that hierarchy; keep them beside their owner.
- Keep every git-tracked handwritten `.rs` file within 1,250 lines; skip marked
  generated output. Enforce the limit in architecture checks.
- Organize modules by subject and developer-visible behavior. Keep helpers
  beside their users and use module documentation to explain file structure.
- Use `name/mod.rs` for modules with child modules, keeping the parent and
  its children in one directory. Use `name.rs` for leaf modules; do not pair
  `name.rs` with a sibling `name/` directory.
- Extend existing mechanisms before adding new ones. Remove dead code and
  obsolete documentation.
- Methods preserve invariants or name nontrivial reusable computations. Use
  fields instead of bare getters/setters when no invariant requires a method.
- Reuse LiveKit crates rather than vendoring protocol definitions. Evaluate
  third-party dependencies against build time and wheel size.
- Keep lifecycle and decision code in safe Rust. Isolate necessary unsafe code
  in narrow OS/FFI/codec adapters and document its ownership, lifetime and
  thread-safety invariants. Validate those boundaries with focused tests.

### Types

- Public Python and Node APIs are fully typed, without `Any`/`any`.
- Preserve the public API's context and output types through Agent,
  `RunContext[Context]`, `RunResult[Output]` and session frames. Infer output
  types from the declared output contract rather than adding public parameters.
- Validate persisted fields, outputs and tool arguments at their boundaries.
- Generate typed wire records from one schema shared by the language bindings.
- Check generated stubs against Rust declarations and use `assert_type` /
  `expectType` tests for generic contracts.
- Follow [shared binding semantics](bindings.md#shared-binding-semantics): declare once in
  Rust, generate both bindings, and verify equivalent conformance cases.

### Schemas and field access

- Export schemas have one source of truth in ABI. External protocol schemas
  use the upstream generated types. Generate models, serializers, conversions,
  field accessors and export registration from those declarations.
- Access known fields through typed records, enums and generated APIs, such as
  `options.name`. Do not use `dict.get("name")`, `value["name"]`, string-based
  attribute lookup or handwritten JSON construction for schema-defined fields.
- Decode and validate untyped input once at its adapter boundary. Core logic
  receives typed values; it does not repeatedly extract or cast JSON fields.
- Application-defined map keys and opaque provider payloads may remain dynamic.
  Keep them at their declared boundary. Known framework metadata keys use
  generated constants and typed accessors, not duplicated string literals.
- A missing generated accessor or conversion is fixed in the schema or
  generator. Do not work around it with a parallel handwritten representation.

### Build performance

- Keep optimizations with large, repeatable gains that justify their code complexity.
  Revert changes whose gains are small, noisy or offset by runtime cost.
- Measure clean builds, warm edits and cache restores separately. Compare the same
  source, toolchain, features, profile and flags in private target directories;
  keep native dependencies equally warm and repeat comparisons.
- Use [cargo llvm-lines](https://github.com/dtolnay/cargo-llvm-lines) to find
  generic functions with many instantiations and large total LLVM IR. Use
  [cargo bloat](https://github.com/RazrFalcon/cargo-bloat) for binary size.
  Neither symbol counts nor IR or binary size replaces build-time measurements.
- Keep generic adapters small when they feed large scheduling or conversion
  implementations. Share the expensive implementation where the operation no
  longer needs the caller's concrete type.
- Preserve useful generics and type safety, including `HostStream<T>`. Do not
  replace typed code with trait objects or function pointers solely to reduce
  instantiation counts.
- Erase future types at the shared Python async bridge before entering PyO3's
  scheduling machinery. Keep the output type concrete and result conversion
  inside PyO3's completion path; run the process/fork guard before allocating.
- Preserve cancellation, future drop, exception conversion and captured task
  context when changing async adapters. Verify these behaviors with boundary tests.
- Account for allocations per call and indirect calls per poll. Compare creation
  CPU and end-to-end latency using ready and repeatedly yielding futures, with
  warmed, alternating-order samples. Do not infer end-to-end runtime cost from
  an isolated microbenchmark.
- Measure the actual bottleneck before changing linkers, LTO, codegen units or
  compiler caches. Require improvements on representative edits; a fast artifact
  restore does not establish a fast edit loop.
- Change generated binding behavior in the generator or its shared handwritten
  helpers. Never optimize generated files by hand.

### Helper tools

- Prefer compiler checks, existing test runners and generated schemas to custom
  source policing or bespoke verdict logic.
- Keep helpers focused on one operation. Make mutations explicit; check modes
  must not write files. Separate data capture from validation and publication.
- Validate inputs before running expensive work or replacing artifacts. Missing
  datasets, empty runs and partial output must not produce success or overwrite
  a valid baseline.
- Preserve subprocess exit codes and actionable errors. A parseable transcript
  does not turn a failed subprocess into a successful run.
- Test validation tools with deliberately failing inputs. State exactly what a
  check covers; a successful narrow check does not establish full architecture
  compliance.

### Tests

- Observe public APIs, core events/effects and journal observations; do not add
  test-only product APIs.
- Give rules negative cases. Prefer behavioral failures to shape/default checks,
  merge duplicate tests.
- Repair generator or fake coverage gaps in the harness after verifying the
  product's refusal. Product behavior changes require a product contract.
- Derive thresholds from the driver's contract rather than arbitrary constants.
## Implementation review

### Extraction unit

Establish `lk-agents-runtime` before extracting executors and host bridges.
The first migrated API uses a real generated boundary, runtime invocation and
synchronous core transition. Grow shared runtime services from working consumers;
a registration shell or forwarding path alone does not establish the design.

Migrate a lifecycle or pipeline leg end to end: its state owner, typed inputs
and outcomes, runtime adapter, host entrypoint and tests. Move its eligibility,
retry, cancellation and deadline decisions into synchronous transitions.
Moving the same policy loop to another async module does not complete extraction.
Keep I/O sequencing and independent resource tasks in runtime.

Executors move into the runtime by splitting, never by moving a
decision-bearing module wholesale. The decisions stay in their subject crate as
synchronous state machines emitting effects (voice for speech and generation
policy, core's AgentRun for tool dispatch); the runtime keeps the pumps, node
calls, timers and output binding that execute them.

Choose slices by dependency and failure risk across voice, inference, worker,
storage and A2A. Generation can move one leg at a time while preserving the
whole speech lifecycle. Each operation has one decision authority throughout
the change; partially updated old and new owners cannot both control it.
Remove the superseded path once its replacement satisfies the contract.

Validate shared interfaces against different real lifecycles before generalizing:
a streaming generation/tool path and an independently failing resource-creation
path. Include replacement/cancellation during open, late completion, rejected
delivery and actual cleanup. A common abstraction must simplify those cases
while preserving their concurrency and declared context.

Move a component's regression cases and observation rules with its owner.
Preserve typed diagnostic observations and the inputs required for replay when
changing message or journal representations. Serializable effects alone do not
establish replayability. Keep observers independent of policy ownership.

### Acceptance evidence

| Property | Required evidence |
| --- | --- |
| One authority | Each mutable fact has an owner; no duplicate lifecycle/retry policy remains in host glue or adapter tasks |
| Core purity | Compiler/dependency checks and source review establish supplied inputs/time, synchronous transitions and no retained live resources |
| Causality and settlement | Immediate launch failure, repeated requests, supersession, stale outcomes and cleanup deadlines settle callers without invalidating replacement work |
| Validation and views | Invalid framework/API input shapes fail before the requested operation's effects; valid-shaped requests against replaced/closed state reach core refusal; host views match state after the API's declared acceptance boundary |
| Host execution | Real callback/node round trips preserve job/speech/approval ownership, current trace parent, stream context and release of captured references |
| Deterministic decisions | Direct core stepping reproduces effects and observations for identical supplied inputs/time, including wire/history timestamps; event-level schedules explore permitted races |
| Real execution | Adapter, process, SDK, database and client tests observe the boundaries and cleanup outcomes defined by their component contracts |
| Resource and latency budgets | Fixed supported workloads measure retained resources, queue/control latency, copying/allocation and media throughput before and after the change |

Direct core simulation and runtime conformance, stress and real-peer tests
exercise different boundaries; each is required for the decision or execution
behavior it covers. Independent I/O tasks remain valid runtime mechanisms.
Campaigns name the policies and cores actually exercised and compare findings
and behavioral reach under matched seeds, workloads and budgets.

Ratchet remaining core violations by kind and location, including async policy,
ambient time/entropy, shared decision state and live objects in core messages.
Reduced spawn counts, smaller files or additional wrappers do not establish
semantic progress. Acceptance requires simpler ownership and failure handling,
preserved concurrency and the behavioral evidence above.

An audit finding names the violated rule and source evidence. Missing coverage
is distinguished from a demonstrated defect. Compilation, a narrow source check,
or one passing E2E run cannot establish all contracts.

Reviews compare implementation and evidence against these ownership and
behavioral contracts. They distinguish missing coverage from defects without
changing the architecture merely to accommodate existing code. Implementation
status and migration checklists are tracked separately from this contract.
