# Rust architecture checks

Run `bash rust/architecture/check.sh` from any directory. CI calls this step;
the non-Python ABI integration test also calls it, so the launch gate executes
it through its existing ABI suite. Use a full Git checkout: baseline history is
part of the check. The tools never rewrite a baseline in check mode.

- `check.py` runs locked `cargo metadata --all-features`, validates the resolved
  graph, and traverses normal, build and development dependencies of every
  subject and ABI. It rejects Tokio, futures, PyO3 and their package families,
  async-trait, async-std, smol, reqwest, hyper/hyper-util, mio, tungstenite,
  livekit/livekit-ffi/livekit-api and the agents runtime. Protocol records remain
  permitted. Direct workspace edges separately enforce the documented layering;
  proc-macro/build tools are exempt from that product-layer edge restriction.
- Dependency identities name the subject/ABI manifest introducing an impure
  dependency path and the forbidden package. Shared ABI debt stays attributed to
  ABI when another pure subject consumes it. A new direct impure dependency in
  that consumer creates a new identity even if ABI already uses that package.
- The same tool reads `git ls-files '*.rs'`. Handwritten files may contain at
  most 1,250 lines. A generated header comment must begin the first line; a
  matching string literal cannot grant an exemption. Existing oversized files
  have individual line ceilings, which can decrease but cannot increase.
- `clippy.py` validates the required per-core Clippy configuration and uses the
  compiler to resolve banned methods, types and macros. The gate checks all
  targets in separate default, Python and all-feature passes, including optional
  native codec code. This is a host-platform feature matrix, not an enumeration
  of every feature combination or target OS. Build-script execution is build
  tooling, outside core policy; dependency closure checks still include it.
- `allows.rs` uses `syn` to identify lint exemptions. The compiler pass forces
  warnings despite existing allowances, then checks each offending site too.
  A new call inside an already-exempted function therefore fails. The Clippy
  baseline records exemption attributes and compiler sites separately, including
  lexical ownership, source identity and local occurrence. Exceptions require
  reasons; new exceptions are not a substitute for fixing a violation.
  Permissions are identified per lint, so removing one lint from a combined
  attribute is a valid shrink. Legacy macro lint allowances also exist at crate
  scope because Clippy ignores narrower allowances for `disallowed_macros`;
  forced compiler warnings still ratchet each macro call individually. Build
  scripts have explicit tooling allowances and are outside the debt baseline.
  Affected legacy host-generation modules have Python-only type allowances:
  generated sibling adapters do not inherit the declaration's lint attributes.
  The source declarations remain annotated and every generated-origin type site
  is still checked by the forced-warning ratchet. New core code gets no such
  legacy allowance.
- `lk-agents-codegen --check` scans parsed handwritten attributes, including
  qualified attributes, inactive `cfg_attr` branches and annotation tokens in
  macros. Export declarations/metadata, including `class` and `binding`, belong in ABI.
  PyO3 annotations belong in
  marked generated output or the exact top-level `_core` module registration.
  Quoted output templates in the three binding-generation crates are allowed;
  strings and forwarded macro metavariables are not expanded. Existing Role and
  other misplaced exports remain explicit debt in the export baseline.
  A separate, restricted metadata baseline bootstraps the previously unchecked
  `class`/`binding` families; it cannot grant exemptions for the older families.

Baseline identities are sets, never aggregate allowances. The Python tools
compare each baseline against every reachable committed version using full Git
history. Retiring an identity prevents its reintroduction, including across a
merge. To fix debt, remove the offending construct and retire its baseline
identity (or lower a file's line ceiling). Do not replace an old identity with a
new one. Baseline creation was a one-time S0 bootstrap; no check updates it.

The standard-library unittest modules and codegen's `placement` Rust tests
exercise deliberately failing Cargo graphs, compiler-resolved construct families,
new/broadened exemptions, same-count replacements, inherited allowances,
1,251-line files, generated-header/string distinctions, missing inputs and
history that hides a retirement behind a merge. Compiler fixtures may download
Rust crates; they require no provider credentials or external service.

These checks enforce mechanical dependency, lint, size and export contracts.
They do not prove synchronous state ownership, correct effect execution,
executable proxy/bridge ownership or complete public API generation. Those
require the owning slices' behavioral tests and review.
