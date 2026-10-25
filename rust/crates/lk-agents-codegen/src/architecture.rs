//! Syntactic module boundaries. No type resolution or macro expansion is attempted.
use std::collections::BTreeMap;
use std::path::Path;
use syn::spanned::Spanned;
use syn::visit::Visit;

const POLICY: &[&str] =
    &["endpointing", "filler_scheduler", "agent_activity::options", "turn", "amd::classifier"];
const SYSTEM_INTERNALS: &[&str] = &[
    "agent_activity",
    "generation",
    "turn",
    "agent_session::cycle::Cycle",
    "agent_session::driver::Driver",
    "agent_session::driver::DriverBuilder",
    "agent_session::driver::Input",
    // The same items through `agent_session`'s re-exports.
    "agent_session::Driver",
    "agent_session::DriverBuilder",
    "agent_session::Input",
];
const SYSTEMS: &[&str] = &["amd::detector", "ivr::ivr_activity"];
/// The core: it must not depend on host systems. A module, or `module::Type` for one type's
/// definition and its impls there: the core types that live beside host-facing code.
const CORE: &[&str] = &[
    "agent_activity",
    "agent_session::cycle",
    "generation",
    "turn",
    "agent_session::SessionShared",
    "agent_session::ActivitySnapshot",
    "io::RealtimeAudioNode",
    "tool_executor::Toolset",
    "tool_executor::SessionTools",
    "tool_executor::ToolRunner",
];
const SESSION_EXECUTION: &[&str] = &[
    // With its api, commands, cycle, driver, host_events and time.
    "agent_session",
    "agent_activity",
    "generation",
    "audio_recognition",
    "audio",
    "tool_executor",
    "run_result",
];
const WALL_WAITS: &[&str] = &[
    "std::thread::sleep",
    "tokio::time::sleep",
    "tokio::time::sleep_until",
    "tokio::time::interval",
];

fn under(path: &str, prefix: &str) -> bool {
    path == prefix || path.strip_prefix(prefix).is_some_and(|rest| rest.starts_with("::"))
}

fn test_only(attrs: &[syn::Attribute]) -> bool {
    // Evaluate cfg with test/testkit disabled. Unknown flags may be true, so ordinary
    // platform-specific production code is still checked.
    fn production(meta: &syn::Meta) -> bool {
        match meta {
            syn::Meta::Path(path) => !path.is_ident("test"),
            syn::Meta::NameValue(value) if value.path.is_ident("feature") => {
                !matches!(&value.value, syn::Expr::Lit(v) if matches!(&v.lit, syn::Lit::Str(s) if s.value() == "testkit"))
            }
            syn::Meta::List(list) => {
                let Ok(items) = list.parse_args_with(
                    syn::punctuated::Punctuated::<syn::Meta, syn::Token![,]>::parse_terminated,
                ) else {
                    return true;
                };
                if list.path.is_ident("all") {
                    items.iter().all(production)
                } else if list.path.is_ident("any") {
                    items.iter().any(production)
                } else {
                    true
                }
            }
            _ => true,
        }
    }
    attrs.iter().any(|attr| {
        attr.path().segments.last().is_some_and(|s| s.ident == "test")
            || attr.path().is_ident("cfg")
                && attr.parse_args::<syn::Meta>().is_ok_and(|meta| !production(&meta))
    })
}

fn attrs(item: &syn::Item) -> &[syn::Attribute] {
    match item {
        syn::Item::Fn(v) => &v.attrs,
        syn::Item::Mod(v) => &v.attrs,
        syn::Item::Impl(v) => &v.attrs,
        syn::Item::Struct(v) => &v.attrs,
        syn::Item::Enum(v) => &v.attrs,
        syn::Item::Use(v) => &v.attrs,
        syn::Item::Const(v) => &v.attrs,
        syn::Item::Static(v) => &v.attrs,
        syn::Item::Type(v) => &v.attrs,
        syn::Item::Trait(v) => &v.attrs,
        syn::Item::Macro(v) => &v.attrs,
        _ => &[],
    }
}

fn imports(tree: &syn::UseTree, prefix: &str, aliases: &mut BTreeMap<String, String>) {
    match tree {
        syn::UseTree::Path(v) => imports(&v.tree, &format!("{prefix}{}::", v.ident), aliases),
        syn::UseTree::Name(v) => {
            let path = if v.ident == "self" {
                prefix.trim_end_matches("::").to_owned()
            } else {
                format!("{prefix}{}", v.ident)
            };
            let name = path.rsplit("::").next().unwrap().to_owned();
            aliases.insert(name, path);
        }
        syn::UseTree::Rename(v) => {
            aliases.insert(v.rename.to_string(), format!("{prefix}{}", v.ident));
        }
        syn::UseTree::Group(v) => {
            for item in &v.items {
                imports(item, prefix, aliases);
            }
        }
        syn::UseTree::Glob(_) => {
            aliases.insert("*".into(), prefix.trim_end_matches("::").to_owned());
        }
    }
}

/// The type a module-level item belongs to: its own name, or an impl's self type.
fn subject(item: &syn::Item) -> Option<String> {
    match item {
        syn::Item::Struct(v) => Some(v.ident.to_string()),
        syn::Item::Enum(v) => Some(v.ident.to_string()),
        syn::Item::Union(v) => Some(v.ident.to_string()),
        syn::Item::Trait(v) => Some(v.ident.to_string()),
        syn::Item::Type(v) => Some(v.ident.to_string()),
        syn::Item::Impl(v) => match &*v.self_ty {
            syn::Type::Path(p) => p.path.segments.last().map(|s| s.ident.to_string()),
            _ => None,
        },
        _ => None,
    }
}

struct Scan<'a> {
    file: &'a str,
    module: String,
    /// The [`subject`] of the module-level item being scanned.
    item: Option<String>,
    aliases: BTreeMap<String, String>,
    in_test: bool,
    errors: Vec<String>,
}

impl Scan<'_> {
    fn report(&mut self, span: proc_macro2::Span, rule: &str, detail: &str) {
        self.errors.push(format!("{}:{}: {rule}: {detail}", self.file, span.start().line));
    }

    fn canonical(&self, path: &str) -> String {
        let mut path = path.to_owned();
        // Follow renamed and grouped imports, with a bound for cyclic aliases.
        for _ in 0..self.aliases.len() + 1 {
            let (head, tail) = path.split_once("::").unwrap_or((&path, ""));
            let Some(alias) = self.aliases.get(head) else { break };
            let next = if tail.is_empty() { alias.clone() } else { format!("{alias}::{tail}") };
            if next == path {
                break;
            }
            path = next;
        }
        if let Some(rest) = path.strip_prefix("self::") {
            path = format!("crate::{}::{rest}", self.module);
        }
        if path.starts_with("super::") {
            let mut base: Vec<_> = self.module.split("::").collect();
            let mut rest = path.as_str();
            while let Some(next) = rest.strip_prefix("super::") {
                base.pop();
                rest = next;
            }
            path = format!("crate::{}::{rest}", base.join("::")).replace("::::", "::");
        }
        path.strip_prefix("lk_agents_voice::").map(|p| format!("crate::{p}")).unwrap_or(path)
    }

    fn policy(&self) -> bool {
        !self.in_test && POLICY.iter().any(|p| under(&self.module, p))
    }

    fn check_path(&mut self, path: &str, span: proc_macro2::Span) {
        let path = self.canonical(path);
        let local = path.strip_prefix("crate::").unwrap_or(&path);
        if self.policy() {
            let forbidden = [
                "crate::agent_session",
                "crate::io",
                "crate::transcription",
                "crate::room_io",
                "lk_agents_rtc",
                "lk_agents_media",
                "std::thread",
            ];
            if forbidden.iter().any(|p| under(&path, p))
                || (under(&path, "tokio") && !under(&path, "tokio::sync::watch"))
            {
                self.report(
                    span,
                    "voice-policy",
                    &format!("{path} belongs to resource-owning code"),
                );
            }
            if ["std::time::Instant::now", "std::time::SystemTime::now"].contains(&path.as_str()) {
                self.report(span, "voice-policy", "time must arrive as an input");
            }
        }
        if !self.in_test
            && SYSTEMS.iter().any(|p| under(&self.module, p))
            && SYSTEM_INTERNALS.iter().any(|p| under(local, p))
        {
            self.report(
                span,
                "voice-systems",
                &format!("{path}: use the host-facing Peer interface"),
            );
        }
        let scope = match &self.item {
            Some(item) => format!("{}::{item}", self.module),
            None => self.module.clone(),
        };
        if !self.in_test
            && CORE.iter().any(|p| under(&scope, p))
            && SYSTEMS.iter().any(|p| under(local, p))
        {
            self.report(span, "voice-systems", "the core must not depend on host systems");
        }
        // Background audio and host systems own runtime timing. Driver/session simulation
        // and tests use injected clocks; timeout bounds on tests remain permitted.
        if (self.in_test
            || SESSION_EXECUTION.iter().any(|p| under(&self.module, p))
            || self.policy())
            && WALL_WAITS.contains(&path.as_str())
        {
            self.report(span, "voice-clock", "use the injected clock or a completion signal");
        }
    }

    fn items(&mut self, items: &[syn::Item]) {
        let previous = self.aliases.clone();
        for item in items {
            if let syn::Item::Use(v) = item
                && !test_only(&v.attrs)
            {
                imports(&v.tree, "", &mut self.aliases);
            }
        }
        for item in items {
            let outer = std::mem::replace(&mut self.item, subject(item));
            self.visit_item(item);
            self.item = outer;
        }
        self.aliases = previous;
    }
}

impl<'ast> Visit<'ast> for Scan<'_> {
    fn visit_item(&mut self, item: &'ast syn::Item) {
        let before = self.in_test;
        self.in_test |= test_only(attrs(item));
        syn::visit::visit_item(self, item);
        self.in_test = before;
    }
    fn visit_impl_item_fn(&mut self, item: &'ast syn::ImplItemFn) {
        let before = self.in_test;
        self.in_test |= test_only(&item.attrs);
        syn::visit::visit_impl_item_fn(self, item);
        self.in_test = before;
    }
    fn visit_trait_item_fn(&mut self, item: &'ast syn::TraitItemFn) {
        let before = self.in_test;
        self.in_test |= test_only(&item.attrs);
        syn::visit::visit_trait_item_fn(self, item);
        self.in_test = before;
    }
    fn visit_item_mod(&mut self, item: &'ast syn::ItemMod) {
        if let Some((_, items)) = &item.content {
            let before = self.module.clone();
            self.module.push_str(&format!("::{}", item.ident));
            self.items(items);
            self.module = before;
        }
    }
    fn visit_block(&mut self, block: &'ast syn::Block) {
        let before = self.aliases.clone();
        for statement in &block.stmts {
            if let syn::Stmt::Item(syn::Item::Use(v)) = statement {
                imports(&v.tree, "", &mut self.aliases);
            }
        }
        syn::visit::visit_block(self, block);
        self.aliases = before;
    }
    fn visit_item_use(&mut self, item: &'ast syn::ItemUse) {
        let mut leaves = BTreeMap::new();
        imports(&item.tree, "", &mut leaves);
        for path in leaves.values() {
            self.check_path(path, item.span());
        }
    }
    fn visit_path(&mut self, path: &'ast syn::Path) {
        let name = path.segments.iter().map(|s| s.ident.to_string()).collect::<Vec<_>>().join("::");
        self.check_path(&name, path.span());
        syn::visit::visit_path(self, path);
    }
    fn visit_expr_await(&mut self, expr: &'ast syn::ExprAwait) {
        if self.policy() && !under(&self.module, "speech_handle") {
            self.report(expr.span(), "voice-policy", "await belongs to resource-owning code");
        }
        syn::visit::visit_expr_await(self, expr);
    }
    fn visit_expr_method_call(&mut self, expr: &'ast syn::ExprMethodCall) {
        if self.policy() && expr.method == "spawn" {
            self.report(expr.span(), "voice-policy", "spawning belongs to resource-owning code");
        }
        syn::visit::visit_expr_method_call(self, expr);
    }
}

fn source(file: &str, text: &str) -> Result<Vec<String>, String> {
    let parsed = syn::parse_file(text).map_err(|e| format!("{file}: {e}"))?;
    let module = file.trim_end_matches(".rs").trim_end_matches("/mod").replace('/', "::");
    let mut scan = Scan {
        file,
        module,
        item: None,
        aliases: BTreeMap::new(),
        in_test: file.starts_with("tests/")
            || file.contains("/tests/")
            || file.ends_with("/tests.rs")
            || file.ends_with("_tests.rs"),
        errors: Vec::new(),
    };
    scan.items(&parsed.items);
    scan.errors.sort();
    scan.errors.dedup();
    Ok(scan.errors)
}

pub(crate) fn check(root: &Path) -> Result<(), String> {
    fn walk(dir: &Path, base: &Path, errors: &mut Vec<String>) -> Result<(), String> {
        for entry in std::fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))? {
            let path = entry.map_err(|e| e.to_string())?.path();
            if path.is_dir() {
                walk(&path, base, errors)?;
            } else if path.extension().is_some_and(|e| e == "rs") {
                let text = std::fs::read_to_string(&path).map_err(|e| e.to_string())?;
                let file = path.strip_prefix(base).unwrap().to_string_lossy();
                errors.extend(
                    source(&file, &text)?
                        .into_iter()
                        .map(|e| format!("rust/crates/lk-agents-voice/src/{e}")),
                );
            }
        }
        Ok(())
    }
    let base = root.join("rust/crates/lk-agents-voice/src");
    for module in POLICY {
        if !base.join(module.replace("::", "/")).is_dir()
            && !base.join(format!("{}.rs", module.replace("::", "/"))).is_file()
        {
            return Err(format!("voice-policy: configured module {module} does not exist"));
        }
    }
    // A core type that moves would otherwise drop out of the rule unnoticed.
    for entry in CORE {
        let (module, item) = match entry.rsplit_once("::") {
            Some((module, item)) if item.starts_with(char::is_uppercase) => (module, Some(item)),
            _ => (*entry, None),
        };
        let dir = base.join(module.replace("::", "/"));
        let file = [dir.with_extension("rs"), dir.join("mod.rs")].into_iter().find(|f| f.is_file());
        let exists = match (file, item) {
            (None, _) => false,
            (Some(_), None) => true,
            (Some(file), Some(item)) => {
                let text = std::fs::read_to_string(&file).map_err(|e| e.to_string())?;
                syn::parse_file(&text).is_ok_and(|f| {
                    f.items.iter().any(|i| {
                        !matches!(i, syn::Item::Impl(_)) && subject(i).as_deref() == Some(item)
                    })
                })
            }
        };
        if !exists {
            return Err(format!("voice-systems: configured core {entry} does not exist"));
        }
    }
    let mut errors = Vec::new();
    walk(&base, &base, &mut errors)?;
    errors.sort();
    if errors.is_empty() { Ok(()) } else { Err(errors.join("\n")) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn syntax_not_comments_strings_or_braces_drives_checks() {
        let text = r###"
            /* tokio::spawn(async {}); { */
            const EXAMPLE: &str = r#"} tokio::spawn(async {});"#;
            fn f() { use tokio::{task::spawn as launch}; launch(async {}); }
        "###;
        let errors = source("endpointing.rs", text).unwrap();
        assert_eq!(errors.len(), 1); // Import and call are on the same line.
        assert!(errors[0].contains("tokio::task::spawn"));
    }

    #[test]
    fn tests_can_own_resources_but_cannot_sleep_and_cfg_any_is_not_all() {
        let text = r#"
            #[cfg(test)] mod tests { fn f() { tokio::spawn(async {}); tokio::time::sleep(d); } }
            #[cfg(any(test, unix))] fn production() { tokio::spawn(async {}); }
            #[cfg(all(test, unix))] fn fixture() { tokio::spawn(async {}); }
        "#;
        let errors = source("endpointing.rs", text).unwrap();
        assert_eq!(errors.len(), 2);
        assert!(errors.iter().any(|e| e.contains("voice-clock")));
        assert!(errors.iter().any(|e| e.contains("voice-policy")));
    }

    #[test]
    fn test_functions_and_test_only_methods_are_not_policy() {
        let text = r#"
            #[tokio::test] async fn test_case() { tokio::spawn(async {}); }
            struct State;
            impl State { #[cfg(test)] async fn fixture() { tokio::spawn(async {}); } }
        "#;
        assert!(source("endpointing.rs", text).unwrap().is_empty());
    }

    #[test]
    fn glob_imports_and_test_aliases_do_not_bypass_boundaries() {
        assert!(!source("endpointing.rs", "use tokio::*;").unwrap().is_empty());
        let text = r#"
            #[cfg(test)] mod tests { use std::time::Instant as Clock; fn f() { Clock::now(); } }
            use std::time::Instant as Clock;
            fn production() { Clock::now(); }
        "#;
        let errors = source("endpointing.rs", text).unwrap();
        assert_eq!(errors.len(), 1);
    }

    #[test]
    fn renamed_clocks_and_relative_module_paths_are_checked() {
        let errors =
            source("endpointing.rs", "use std::time::Instant as Clock; fn f() { Clock::now(); }")
                .unwrap();
        assert_eq!(errors.len(), 1);
        assert!(
            !source("agent_activity/events.rs", "use super::super::amd::detector;")
                .unwrap()
                .is_empty()
        );
        assert!(
            !source("amd/detector.rs", "use crate::agent_activity::scheduling::Scheduler;")
                .unwrap()
                .is_empty()
        );
        assert!(source("speech_handle.rs", "async fn f() { cell.await; }").unwrap().is_empty());
        assert!(!source("endpointing.rs", "async fn f() { cell.await; }").unwrap().is_empty());
    }

    #[test]
    fn a_core_type_is_checked_in_a_host_facing_module_and_its_neighbours_are_not() {
        let text = r#"
            use crate::amd::detector::AmdSetup;
            pub struct SessionShared { pub amd: Option<AmdSetup> }
            impl SessionShared { fn f() { crate::ivr::ivr_activity::IvrSystem::spawn(); } }
            fn attach(setup: AmdSetup) {}
        "#;
        let errors = source("agent_session/mod.rs", text).unwrap();
        assert_eq!(errors.len(), 2, "{errors:?}");
        assert!(errors[0].contains(":3: voice-systems: the core must not depend"), "{errors:?}");
        assert!(errors[1].contains(":4: voice-systems: the core must not depend"), "{errors:?}");
    }
}
