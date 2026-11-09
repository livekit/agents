//! Crate graph rules (AGENTS.md "Crate graph" and "Sans-I/O cores"), resolved with
//! `cargo metadata` for every platform a wheel ships for:
//!
//! 1. Each crate depends only on the workspace crates listed for it in [`GRAPH`].
//! 2. Every workspace crate is listed in [`GRAPH`] or [`TOOLS`].
//! 3. abi, core, voice and media resolve no executor, transport, host or SDK crate ([`BANNED`],
//!    [`BANNED_FAMILIES`]), through any normal, build or dev edge.
//! 4. Only `lk-agents-py` depends on pyo3, so the compiler rejects pyo3 anywhere else.
//! 5. Only abi, storage and py depend on `lk-agents-host-derive`, and ABI does not re-export
//!    `host`, so the compiler rejects `#[host]` anywhere else (storage uses `#[table]` only).

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::process::Command;

use serde_json::Value;

const ABI: &str = "lk-agents-abi";
const CORE: &str = "lk-agents-core";
const VOICE: &str = "lk-agents-voice";
const MEDIA: &str = "lk-agents-media";
const TELEMETRY: &str = "lk-agents-telemetry";
const NET: &str = "lk-agents-net";
const INFERENCE: &str = "lk-agents-inference";
const STORAGE: &str = "lk-agents-storage";
const WORKER: &str = "lk-agents-worker";
const CLI: &str = "lk-agents-cli";
const RUNTIME: &str = "lk-agents-runtime";
const FUZZ: &str = "lk-agents-fuzz";
const PY: &str = "lk-agents-py";
const HOST_DERIVE: &str = "lk-agents-host-derive";

/// Each crate's allowed workspace dependencies (rust-core.md "Crates and dependencies").
const GRAPH: &[(&str, &[&str])] = &[
    (ABI, &[HOST_DERIVE]),
    (CORE, &[ABI]),
    (VOICE, &[ABI, CORE, MEDIA, TELEMETRY]),
    (MEDIA, &[ABI, TELEMETRY]),
    (TELEMETRY, &[ABI]),
    (NET, &[ABI, CORE, MEDIA, TELEMETRY]),
    (INFERENCE, &[ABI, CORE, MEDIA, TELEMETRY, NET]),
    (STORAGE, &[ABI, TELEMETRY, NET, HOST_DERIVE]),
    (WORKER, &[ABI, TELEMETRY, NET]),
    (CLI, &[ABI, NET, WORKER]),
    (RUNTIME, &[ABI, CORE, VOICE, MEDIA, TELEMETRY, NET, INFERENCE, STORAGE, WORKER, CLI]),
    (FUZZ, &[ABI, CORE, VOICE, MEDIA, TELEMETRY, RUNTIME]),
    (PY, &[ABI, RUNTIME, HOST_DERIVE]),
];

/// Proc macros and generators: they define no product dependency direction.
const TOOLS: &[&str] = &["lk-agents-bindgen", "lk-agents-check", "lk-agents-codegen", HOST_DERIVE];

const SANS_IO: &[&str] = &[ABI, CORE, VOICE, MEDIA];

/// Executors, transports, host and SDK crates; `livekit-protocol` records are fine.
const BANNED: &[&str] = &[
    "async-std",
    "async-trait",
    "hyper",
    "hyper-util",
    "livekit",
    "livekit-api",
    "livekit-ffi",
    "mio",
    "reqwest",
    "rusqlite",
    "smol",
    "tungstenite",
];
/// Banned with every `<family>-*` crate.
const BANNED_FAMILIES: &[&str] = &["futures", "pyo3", "tokio"];

/// The wheel targets (.github/workflows/build.yml): an edge only another target takes is never built.
const PLATFORMS: &[&str] = &[
    "x86_64-unknown-linux-gnu",
    "aarch64-unknown-linux-gnu",
    "x86_64-unknown-linux-musl",
    "aarch64-unknown-linux-musl",
    "x86_64-apple-darwin",
    "aarch64-apple-darwin",
    "x86_64-pc-windows-msvc",
];

pub fn check(rust: &Path) -> Vec<String> {
    let mut errors = BTreeSet::new();
    for platform in PLATFORMS {
        errors.extend(violations(&Resolved::new(&metadata(rust, platform))));
    }
    errors.into_iter().collect()
}

fn metadata(rust: &Path, platform: &str) -> Value {
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".into());
    let output = Command::new(cargo)
        .args(["metadata", "--locked", "--all-features", "--format-version", "1"])
        .args(["--filter-platform", platform])
        .current_dir(rust)
        .output()
        .expect("cargo runs");
    assert!(
        output.status.success(),
        "cargo metadata failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).expect("cargo metadata prints JSON")
}

/// The resolved graph by package name: every edge kind (normal, build and dev), all features.
struct Resolved {
    members: Vec<String>,
    deps: BTreeMap<String, BTreeSet<String>>,
}

impl Resolved {
    fn new(metadata: &Value) -> Self {
        let names: BTreeMap<&str, &str> = list(&metadata["packages"])
            .iter()
            .filter_map(|package| Some((package["id"].as_str()?, package["name"].as_str()?)))
            .collect();
        let name = |id: &Value| -> String {
            id.as_str()
                .and_then(|id| names.get(id))
                .expect("every resolved id is a package")
                .to_string()
        };
        let members = list(&metadata["workspace_members"]).iter().map(name).collect();
        let deps = list(&metadata["resolve"]["nodes"])
            .iter()
            .map(|node| {
                (name(&node["id"]), list(&node["deps"]).iter().map(|d| name(&d["pkg"])).collect())
            })
            .collect();
        Self { members, deps }
    }

    fn direct(&self, name: &str) -> impl Iterator<Item = &str> {
        self.deps.get(name).into_iter().flatten().map(String::as_str)
    }

    /// Each crate `root` resolves, with the chain that pulls it in.
    fn closure(&self, root: &str) -> BTreeMap<&str, Vec<&str>> {
        let mut chains: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
        let mut pending: Vec<(&str, Vec<&str>)> = vec![(root, vec![])];
        while let Some((name, chain)) = pending.pop() {
            for dep in self.direct(name) {
                if dep != root && !chains.contains_key(dep) {
                    let mut via = chain.clone();
                    via.push(dep);
                    chains.insert(dep, via.clone());
                    pending.push((dep, via));
                }
            }
        }
        chains
    }
}

fn list(value: &Value) -> &[Value] {
    value.as_array().map_or(&[], Vec::as_slice)
}

/// `name` is `family` or one of its `family-*` crates.
fn in_family(name: &str, family: &str) -> bool {
    name.strip_prefix(family).is_some_and(|rest| rest.is_empty() || rest.starts_with('-'))
}

fn banned(name: &str) -> bool {
    BANNED.contains(&name) || BANNED_FAMILIES.iter().any(|family| in_family(name, family))
}

fn violations(graph: &Resolved) -> BTreeSet<String> {
    let mut errors = BTreeSet::new();
    for name in &graph.members {
        let name = name.as_str();
        if SANS_IO.contains(&name) {
            for (dep, chain) in graph.closure(name) {
                if banned(dep) {
                    errors.insert(format!(
                        "{name} resolves {dep} ({}): sans-I/O cores take no executor, transport, host or SDK crate",
                        chain.join(" -> ")
                    ));
                }
            }
        }
        if name != PY {
            for dep in graph.direct(name).filter(|dep| in_family(dep, "pyo3")) {
                errors.insert(format!("{name} depends on {dep}: pyo3 belongs to {PY} alone"));
            }
        }
        if TOOLS.contains(&name) {
            continue;
        }
        let Some((_, allowed)) = GRAPH.iter().find(|(crate_name, _)| *crate_name == name) else {
            errors.insert(format!(
                "{name} is missing from the crate graph in rust/check/src/graph.rs"
            ));
            continue;
        };
        for dep in graph.direct(name) {
            if dep.starts_with("lk-agents-") && dep != name && !allowed.contains(&dep) {
                errors.insert(format!("{name} must not depend on {dep} (crate graph)"));
            }
        }
    }
    errors
}

#[cfg(test)]
mod tests {
    use super::*;

    fn graph(edges: &[(&str, &[&str])]) -> Resolved {
        Resolved {
            members: edges
                .iter()
                .filter(|(name, _)| name.starts_with("lk-agents-"))
                .map(|(name, _)| (*name).to_owned())
                .collect(),
            deps: edges
                .iter()
                .map(|(name, deps)| {
                    ((*name).to_owned(), deps.iter().map(|d| (*d).to_owned()).collect())
                })
                .collect(),
        }
    }

    #[test]
    fn allowed_edges_pass() {
        let graph = graph(&[(ABI, &["serde", HOST_DERIVE]), (CORE, &[ABI]), ("serde", &[])]);
        assert!(violations(&graph).is_empty());
    }

    #[test]
    fn a_core_resolving_tokio_fails_with_its_chain() {
        let graph = graph(&[
            (ABI, &[]),
            (CORE, &[ABI, "helper"]),
            ("helper", &["tokio-util"]),
            ("tokio-util", &[]),
        ]);
        let errors = violations(&graph);
        assert!(
            errors
                .iter()
                .any(|e| e.contains("lk-agents-core resolves tokio-util (helper -> tokio-util)"))
        );
    }

    #[test]
    fn an_edge_outside_the_graph_fails() {
        let graph = graph(&[(ABI, &[]), (CORE, &[ABI, STORAGE]), (STORAGE, &[ABI])]);
        assert!(
            violations(&graph)
                .contains("lk-agents-core must not depend on lk-agents-storage (crate graph)")
        );
    }

    #[test]
    fn host_derive_outside_its_crates_fails() {
        let graph = graph(&[(ABI, &[]), (CORE, &[ABI, HOST_DERIVE])]);
        assert!(
            violations(&graph)
                .contains("lk-agents-core must not depend on lk-agents-host-derive (crate graph)")
        );
    }

    #[test]
    fn pyo3_outside_the_binding_fails() {
        let graph = graph(&[(ABI, &[]), (RUNTIME, &[ABI, "pyo3"]), ("pyo3", &[])]);
        assert!(
            violations(&graph).iter().any(|e| e.contains("pyo3 belongs to lk-agents-py alone"))
        );
    }

    #[test]
    fn an_unlisted_crate_fails() {
        let graph = graph(&[("lk-agents-new", &[])]);
        assert!(violations(&graph).iter().any(|e| e.contains("missing from the crate graph")));
    }
}
