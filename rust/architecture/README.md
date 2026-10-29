# Rust architecture checks

Run `bash rust/architecture/check.sh` from any directory in a full Git checkout.
CI and the launch gate call this entry point directly. CI runs the architecture
step once, then calls `launch-gate.sh --skip-architecture` for the remaining suites.
Checks never rewrite sources or baselines. Cargo honors `CARGO_TARGET_DIR` and
the workspace Rust flags.

- `check.py` validates all-feature Cargo dependency closures, including normal,
  build and dev edges, and workspace layering against the documented crate graph
  (`GRAPH`, rust-core.md "Crates and dependencies").
  Forbidden executor, transport, host and SDK families include Tokio, futures,
  PyO3, rusqlite and LiveKit SDKs; protocol records are allowed.
  The async I/O crates (inference, worker, net, storage) are outside the purity bans
  and the Clippy policy; their workspace edges are checked.
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
- Git-tracked handwritten Rust files have a 1,250-line limit. A generated file's
  first line is exactly codegen's header, `//! GENERATED FILE -- DO NOT EDIT.` Existing oversized files have individual
  shrinking ceilings.
- Clippy resolves banned methods, types and macros over all targets in default,
  Python and all-feature configurations. Per-core lint configurations are checked.
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

Unittest and codegen tests include deliberately failing dependency, compiler,
exemption, length, placement and history fixtures. Compiler counterexamples use
one example in the existing codegen package and lockfile. Empty, partial or failed
compiler transcripts and source changes during collection fail validation.
