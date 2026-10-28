//! Grandfathered ABI proxy conversions for remaining in-place consumers.
//! New declarations extract their proxies in generated Python code and need no ABI implementation.

use crate::decl::Declarations;
use std::collections::BTreeSet;
use std::path::Path;
use std::process::{Command, Output};

// The exact HostProxy implementations generated before this registry was introduced. This list
// only shrinks as legacy exports and handwritten record conversions move to declared adapters.
const IN_PLACE_PROXIES: &[&str] = &[
    "::lk_agents_abi::a2a::turn::TurnHost",
    "::lk_agents_abi::agent::AgentHost",
    "::lk_agents_abi::agent::HostInstructions",
    "::lk_agents_abi::evals::EvaluationHost",
    "::lk_agents_abi::llm::fallback::ChainObserver",
    "::lk_agents_abi::Llm",
    "::lk_agents_abi::llm::mcp::McpTransport",
    "::lk_agents_abi::llm::mcp::McpProgress",
    "::lk_agents_abi::llm::tool_executor::HostToolRunner",
    "::lk_agents_abi::Stt",
    "::lk_agents_abi::Tts",
    "::lk_agents_abi::EotDetector",
    "::lk_agents_abi::Vad",
    "::lk_agents_abi::voice::avatar::ClearBufferListener",
    "::lk_agents_abi::voice::avatar::AvatarAudioReceiver",
    "::lk_agents_abi::voice::avatar::AvatarVideoGenerator",
    "::lk_agents_abi::voice::avatar::AvatarMedia",
    "::lk_agents_abi::voice::background_audio::FrameStream",
    "::lk_agents_abi::voice::io::PlaybackListener",
    "::lk_agents_abi::voice::io::AudioInput",
    "::lk_agents_abi::voice::io::VideoInput",
    "::lk_agents_abi::voice::io::AudioOutput",
    "::lk_agents_abi::voice::io::TextOutput",
    "::lk_agents_abi::voice::session_effects::SessionHooks",
    "::lk_agents_abi::voice::session_effects::SessionEvents",
    "::lk_agents_abi::worker::job::SessionJob",
    "::lk_agents_abi::worker::remote_session::SessionPort",
    "::lk_agents_abi::worker::cli::DevServer",
];

pub(super) fn proxy_in_place(path: &str) -> bool {
    IN_PLACE_PROXIES.contains(&path)
}

const REGISTRY: &str = "rust/crates/lk-agents-codegen/src/records/legacy.rs";

/// Local history lookups and fixture commits must not inherit a caller's Git repository/config.
fn git(root: &Path, args: &[&str]) -> std::io::Result<Output> {
    Command::new("git")
        .args(args)
        .current_dir(root)
        .env_clear()
        .env("PATH", std::env::var_os("PATH").unwrap_or_default())
        .env("GIT_CONFIG_NOSYSTEM", "1")
        .env("GIT_CONFIG_SYSTEM", "/dev/null")
        .env("GIT_CONFIG_GLOBAL", "/dev/null")
        .env("GIT_TERMINAL_PROMPT", "0")
        .output()
}

fn at_revision(root: &Path, revision: &str, path: &str) -> Option<String> {
    let output = git(root, &["show", &format!("{revision}:{path}")]).ok()?;
    output.status.success().then(|| String::from_utf8_lossy(&output.stdout).into_owned())
}

/// Read only the existing Rust const's array; comments, docs and tests cannot grant membership.
fn registry(source: &str) -> Result<BTreeSet<String>, String> {
    let file = syn::parse_file(source).map_err(|error| format!("legacy registry: {error}"))?;
    let Some(value) = file.items.iter().find_map(|item| match item {
        syn::Item::Const(value) if value.ident == "IN_PLACE_PROXIES" => Some(&*value.expr),
        _ => None,
    }) else {
        return Err("legacy registry has no IN_PLACE_PROXIES const".into());
    };
    let value = match value {
        syn::Expr::Reference(value) => &*value.expr,
        value => value,
    };
    let syn::Expr::Array(values) = value else {
        return Err("IN_PLACE_PROXIES is an array of string identities".into());
    };
    values
        .elems
        .iter()
        .map(|value| match value {
            syn::Expr::Lit(syn::ExprLit { lit: syn::Lit::Str(value), .. }) => Ok(value.value()),
            _ => Err("IN_PLACE_PROXIES holds string identities".into()),
        })
        .collect()
}

struct History {
    registries: Vec<BTreeSet<String>>,
    /// Parents without a registry, proving which ABI bridges predate its introduction.
    introductions: Vec<String>,
}

fn history(root: &Path) -> Result<History, String> {
    let shallow = git(root, &["rev-parse", "--is-shallow-repository"])
        .map_err(|error| format!("legacy registry history: {error}"))?;
    if !shallow.status.success() {
        return Err("cannot determine whether legacy registry history is shallow".into());
    }
    if String::from_utf8_lossy(&shallow.stdout).trim() == "true" {
        return Err("legacy registry requires full Git history; run `git fetch --unshallow` or use checkout fetch-depth: 0 before codegen".into());
    }
    let revisions = git(root, &["rev-list", "--full-history", "HEAD", "--", REGISTRY])
        .map_err(|error| format!("legacy registry history: {error}"))?;
    if !revisions.status.success() {
        return Err(format!(
            "legacy registry history: {}",
            String::from_utf8_lossy(&revisions.stderr)
        ));
    }
    let mut history = History { registries: Vec::new(), introductions: Vec::new() };
    for revision in String::from_utf8_lossy(&revisions.stdout).lines() {
        let source = at_revision(root, revision, REGISTRY);
        // Deleting the registry retires every identity; reintroduction cannot reset the ratchet.
        history.registries.push(source.as_deref().map_or_else(|| Ok(BTreeSet::new()), registry)?);
        if source.is_some() {
            let parents = git(root, &["rev-list", "--parents", "-n", "1", revision])
                .map_err(|error| format!("legacy registry introduction: {error}"))?;
            if !parents.status.success() {
                return Err("cannot read legacy registry introduction parents".into());
            }
            for parent in String::from_utf8_lossy(&parents.stdout).split_whitespace().skip(1) {
                if at_revision(root, parent, REGISTRY).is_none() {
                    history.introductions.push(parent.to_owned());
                }
            }
        }
    }
    history.introductions.sort();
    history.introductions.dedup();
    Ok(history)
}

fn grandfathered(root: &Path, path: &str, family: &str, previous: &History) -> bool {
    if !previous.registries.iter().all(|registry| registry.contains(path)) {
        return false;
    }
    let introduction = ["HEAD".to_owned()];
    let revisions =
        if previous.registries.is_empty() { &introduction[..] } else { &previous.introductions };
    // An initial registry at the repository's root commit has no older bridge to grandfather.
    !revisions.is_empty()
        && revisions.iter().all(|revision| {
            at_revision(
                root,
                revision,
                &format!("rust/crates/lk-agents-abi/src/python/generated/{family}.rs"),
            )
            .is_some_and(|source| {
                source.contains(&format!("impl crate::python::HostProxy for dyn {path} {{"))
            })
        })
}

pub(super) fn check_proxies(root: &Path, declarations: &Declarations) -> Result<(), String> {
    let previous = history(root)?;
    for path in IN_PLACE_PROXIES {
        let proxy = declarations.proxies.iter().find(|proxy| proxy.trait_path == *path)
            .ok_or_else(|| format!("records::IN_PLACE_PROXIES: `{path}` is no longer declared; remove its legacy bridge"))?;
        if !grandfathered(root, path, &proxy.family, &previous) {
            return Err(format!(
                "records::IN_PLACE_PROXIES: `{path}` is new or retired in history; extract it in generated Python bindings instead of adding an ABI bridge"
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Repository(std::path::PathBuf);

    impl Drop for Repository {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    impl Repository {
        fn run(&self, args: &[&str]) {
            let output = git(&self.0, args).expect("isolated git runs");
            assert!(
                output.status.success(),
                "git {args:?}: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
        fn commit(&self, contents: &str) {
            self.commit_path(REGISTRY, contents);
        }
        fn commit_path(&self, file: &str, contents: &str) {
            let path = self.0.join(file);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, contents).unwrap();
            self.run(&["add", file]);
            self.run(&[
                "-c",
                "user.name=codegen fixture",
                "-c",
                "user.email=codegen@invalid",
                "-c",
                "commit.gpgsign=false",
                "-c",
                "core.hooksPath=/dev/null",
                "commit",
                "-qm",
                "fixture",
            ]);
        }
    }

    #[test]
    fn committed_additions_and_retired_identities_cannot_be_grandfathered_later() {
        let path =
            std::env::temp_dir().join(format!("codegen-proxy-history-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir(&path).unwrap();
        let repo = Repository(path);
        repo.run(&["-c", "init.templateDir=", "init", "-q"]);
        repo.commit_path("rust/crates/lk-agents-abi/src/python/generated/unused.rs",
            "impl crate::python::HostProxy for dyn old {}\nimpl crate::python::HostProxy for dyn kept {}\nimpl crate::python::HostProxy for dyn new {}\n");
        repo.commit(
            r#"
            // A quoted identity outside the array grants nothing: "new".
            const QUOTED: &str = "new";
            const IN_PLACE_PROXIES: &[&str] = &["old", "kept", "unproved"];
        "#,
        );
        repo.commit(r#"const IN_PLACE_PROXIES: &[&str] = &["old", "kept", "new", "unproved"];"#);
        let snapshots = history(&repo.0).unwrap();
        assert!(
            snapshots.registries[0].contains("new"),
            "the unauthorized addition is already committed at HEAD"
        );
        assert!(
            !grandfathered(&repo.0, "new", "unused", &snapshots),
            "committing a broadened list cannot make it valid"
        );
        repo.commit(r#"const IN_PLACE_PROXIES: &[&str] = &["kept", "new", "unproved"];"#);
        repo.commit(r#"const IN_PLACE_PROXIES: &[&str] = &["old", "kept", "new", "unproved"];"#);
        let snapshots = history(&repo.0).unwrap();
        assert_eq!(snapshots.registries.len(), 4);
        assert!(
            !grandfathered(&repo.0, "old", "unused", &snapshots),
            "a committed reintroduction cannot undo retirement"
        );
        assert!(grandfathered(&repo.0, "kept", "unused", &snapshots));
        assert!(
            !grandfathered(&repo.0, "unproved", "unused", &snapshots),
            "committing an initial registry cannot manufacture its prior ABI bridge proof"
        );
    }

    #[test]
    fn shallow_checkout_reports_how_to_restore_required_history() {
        let path =
            std::env::temp_dir().join(format!("codegen-shallow-source-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir(&path).unwrap();
        let source = Repository(path);
        source.run(&["-c", "init.templateDir=", "init", "-q"]);
        source.commit("const IN_PLACE_PROXIES: &[&str] = &[\"held\"];\n");
        source.commit("// another revision\nconst IN_PLACE_PROXIES: &[&str] = &[\"held\"];\n");
        let path =
            std::env::temp_dir().join(format!("codegen-shallow-clone-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir(&path).unwrap();
        let shallow = Repository(path);
        source.run(&[
            "-c",
            "init.templateDir=",
            "-c",
            "core.hooksPath=/dev/null",
            "clone",
            "-q",
            "--no-local",
            "--depth=1",
            source.0.to_str().unwrap(),
            shallow.0.to_str().unwrap(),
        ]);
        let error = history(&shallow.0).err().expect("shallow history cannot enforce retirement");
        assert!(
            error.contains("full Git history") && error.contains("git fetch --unshallow"),
            "{error}"
        );
        assert!(
            !error.contains("new or retired"),
            "a shallow clone must not misdiagnose valid identities: {error}"
        );
    }
}
