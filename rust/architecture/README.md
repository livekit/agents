# Rust architecture checks

Run `bash rust/architecture/check.sh` from any directory. CI and the non-Python
ABI integration suite use this entry point. Full Git history is required.
Checks never rewrite baselines. Cargo uses the caller's `CARGO_TARGET_DIR`
(or the workspace default) and the gate's Rust flags for every build.

- `check.py` checks locked all-feature Cargo metadata, normal/build/dev closure
  and workspace layering for subjects and ABI. Forbidden executor, transport,
  host and SDK families include Tokio, futures, PyO3 and LiveKit SDKs; protocol
  records are allowed. Debt belongs to the manifest introducing it.
- It checks Git-tracked handwritten Rust files against 1,250 lines. A generated
  marker must start the first line. Existing oversized files have shrinking
  individual ceilings; strings cannot grant generated-file exemptions.
- `clippy.py` runs one compiler pass per default/Python/all-feature configuration,
  over all targets. Required per-core configurations resolve banned methods,
  types and macros through the compiler. This is a host-platform feature matrix.
  Build scripts are tooling; their dependencies still participate in closures.
- The existing `allows.rs` parser inventories reasoned exemptions and maps every
  forced-warning site to its lexical identity. Coverage also requires an effective
  source exemption; deleting an allowance cannot leave a baselined call permitted.
  Unknown conditional permissions fail closed. Expanded sites require module
  permissions; macros require crate scope, matching legacy Clippy limitations.
  Argument attributes cannot authorize signature type lints.
  New calls inside old allowances fail the site ratchet.
- `lk-agents-codegen --check` checks parsed export/metadata placement and generated
  drift. Declarations belong in ABI; PyO3 belongs in generated output or the exact
  module registration shell. Existing placement debt has separate identity sets.

Every baseline is compared with every reachable committed version, including
merge-hidden retirements. Identities cannot be replaced or reintroduced; numeric
ceilings cannot grow. No check grants or broadens exemptions.

Unittest and codegen tests retain deliberately failing graph, lint, exemption,
length, placement and history fixtures. Compiler counterexamples are one example
in the existing codegen package and lockfile: one Cargo invocation captures all
construct families, growth under old permissions, and missing permissions.
Compiler transcripts must contain successful completion and all required core
artifacts. Source changes during collection fail validation.
