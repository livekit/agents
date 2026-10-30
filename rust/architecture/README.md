# Rust architecture checks

Run `bash rust/architecture/check.sh` from any directory in a full Git checkout.
CI and the launch gate also run `bash rust/architecture/check.sh --tests` for
checker counterexamples and codegen tests. CI runs this step once, then calls
`launch-gate.sh --skip-architecture` for the remaining suites.
Checks never rewrite sources or baselines. Cargo honors `CARGO_TARGET_DIR` and
the workspace Rust flags.

From the repository root, `make -C rust check` runs this enforcement gate.
`make -C rust format-check` checks formatting separately.
`make -C rust clippy` runs just the scoped diagnostic pass.
`make -C rust test-checks` tests the checkers themselves. Run `make -C rust` for help.

- `check.py` validates all-feature Cargo dependency closures, including normal,
  build and dev edges, resolved for each platform a wheel ships for (an edge only
  another target takes, such as wasm32's, is never built), and workspace layering
  against the documented crate graph
  (`GRAPH`, rust-core.md "Crates and dependencies").
  Forbidden executor, transport, host and SDK families include Tokio, futures,
  PyO3, rusqlite and LiveKit SDKs; protocol records are allowed.
  The async I/O crates (inference, worker, net, storage) are outside the purity bans
  and the restricted Clippy policy; ordinary workspace Clippy still checks them.
- A `#[path]` module stays inside its own crate's tree (`src/`, `tests/`, ..): a file under a core's
  directory compiled by another crate would escape that core's dependency and Clippy policy, so
  every core-directory file is either compiled by its own crate (and linted with it) or an
  identity of the shrink-only `paths-baseline.json`.
- pyo3 is the extension crate's alone (decision 74): a pyo3-family dependency of any other
  workspace crate, or another crate's tracked Rust file naming pyo3 (generated output included),
  is an identity of the shrink-only `pyo3-baseline.json`. The generator crates, which write the
  extension's code as text, are exempt.
- Each use is ratcheted too, not only its file or crate: every pyo3 path outside the extension
  and generator crates, and every `async_trait` in a subject core or ABI, is counted per file,
  enclosing item and path (comments and strings excluded) in `uses-baseline.json`; a new use in
  a file or item that already has others grows a ceiling.
- Every Rust-to-host call goes through HostInvoker: a file naming pyo3 that calls into host code
  (`call*`, `call_soon_threadsafe`, `into_future`, `future_into_py`) outside the Python entry
  `lk-agents-py/src/host/` is an identity of the shrink-only `host-calls-baseline.json`; generated
  code calls through the entry. Generators and integration tests are exempt.
- The runtime executors the DST runs (decision 92: `lk-agents-runtime/src/{voice,room_io,avatar,rtc}`,
  outside tests) use no Tokio runtime (`tokio::spawn`, `tokio::time`, `tokio::net`, `Handle`) and
  no unbiased `select!`; per file and kind, counts are ceilings in the shrink-only
  `executors-baseline.json`.
- A workspace Clippy pass compiles every crate and target with FFmpeg enabled,
  replacing a separate workspace `cargo check`. Metadata is resolved once for
  dependency and warning checks; Cargo's reported executable is reused for the
  syntax helper and codegen drift check. Each compiler configuration keeps its
  own state beside Cargo's target directory (`target-checks/workspace`,
  `target-checks/default`, `target-checks/python`, `target-checks/all`, and
  `target-checks/tools`). This prevents successive
  invocations from replacing each other's fingerprints. mbx still shares
  equivalent compiled artifacts across configurations and worktrees. The syntax
  helper uses the optimized `check-tools` profile; agent builds keep their normal
  development profile.
- Every workspace crate is warning-clean (`WARNING_CLEAN` in `check.py`): compiler and
  Clippy warnings in the workspace pass fail the check. External path dependencies
  are outside this gate. The separate purity passes below force their policy diagnostics
  on and compare them with the shrinking baselines.
- Git-tracked handwritten Rust files have a 1,250-line limit. A generated file's
  first line is exactly codegen's header, `//! GENERATED FILE -- DO NOT EDIT.` Existing oversized files have individual
  shrinking ceilings.
- Clippy resolves banned methods, types and macros over all targets in default,
  Python and all-feature configurations. One handwritten `clippy.toml` in this
  directory defines the policy. The checker explicitly sets `CLIPPY_CONF_DIR`
  to this directory and selects only ABI, core, voice and media with `-p`.
- A second policy, `io/clippy.toml`, covers runtime, worker, net, inference, storage and py
  (default and all-feature configurations): no poisoning `std::sync` lock (they lock with
  `parking_lot`), and no entry to Tokio's blocking pool (`spawn_blocking`, `block_in_place`,
  `tokio::fs`, `lookup_host`, Tokio's stdio, tokio-tungstenite's resolving connects). No lock is
  held across a host call, and every generated synchronous Python entry releases the GIL for
  its Rust call (rust-core.md "The GIL boundary"), so a lock never deadlocks against the GIL.
  `io/clippy-baseline.json` holds the existing sites and only shrinks.
- Codegen's shared source visitor inventories reasoned lint exemptions, identifies
  compiler sites and validates export placement, including inactive branches and
  macro bodies. Effective exemptions are required for every compiler site.
  New calls inside old allowances fail the site ratchet. Argument attributes
  cannot authorize signature type lints; expanded sites require module exemptions.
- `lk-agents-codegen --check` validates generated output and export placement.
  Export declarations and metadata belong in ABI. PyO3 belongs in generated output
  or the module registration shell.

Every baseline is compared with every reachable committed version, including
merge-hidden retirements. Identities cannot be replaced or reintroduced; numeric
ceilings cannot grow. No check grants or broadens exemptions.
RTC dependency/Clippy baselines bootstrap the previously unchecked crate once;
the SQLite dependency baseline bootstraps the newly banned storage driver. Each
retains independent history and rejects identities from older baseline families.

## Selecting a Clippy policy

An explicit `CLIPPY_CONF_DIR` selects one configuration for the invocation.
Package selection determines where it applies; no generated configuration or
crate-local copy is needed.
For example, from the repository root:

```bash
CLIPPY_CONF_DIR="$PWD/rust/architecture" cargo clippy \
    --manifest-path rust/Cargo.toml --locked --no-deps --all-targets \
    -p lk-agents-abi -p lk-agents-core -p lk-agents-voice -p lk-agents-media
```

This is a diagnostic command. The full `check.sh` also forces diagnostics for
locally allowed sites, checks all three feature configurations, and validates
each violation and exemption against its baseline. Plain `cargo clippy` uses
ordinary lint configuration; it is not the architecture gate.

If packages need different policies, run them in separate invocations, each
with its own explicit `CLIPPY_CONF_DIR` and `-p` list. Do not apply the purity
policy to `--workspace`: I/O packages intentionally use these APIs.

Unittest and codegen tests include deliberately failing dependency, compiler,
exemption, length, placement and history fixtures. Compiler counterexamples use
one example in the existing codegen package and lockfile. Empty, partial or failed
compiler transcripts and source changes during collection fail validation.
