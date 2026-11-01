//! Select exported classes from their actual Rust trait implementations. No second class list:
//! each ABI class must have exactly one concrete implementation reachable by the backend.
use crate::decl::Declarations;
use crate::decl::tree::{Crate, Found};
use crate::python::Out;
use quote::ToTokens;
use std::collections::BTreeMap;
use std::path::Path;
use syn::visit_mut::VisitMut;

struct Sources {
    crates: BTreeMap<String, Crate>,
    routes: Vec<(Vec<String>, String)>,
}

impl Sources {
    fn read(root: &Path) -> Result<Self, String> {
        let metadata = std::process::Command::new("cargo")
            .args([
                "metadata",
                "--no-deps",
                "--format-version",
                "1",
                "--locked",
                "--offline",
                "--manifest-path",
            ])
            .arg(root.join("rust/Cargo.toml"))
            .output()
            .map_err(|e| format!("cargo metadata: {e}"))?;
        if !metadata.status.success() {
            return Err(format!("cargo metadata: {}", String::from_utf8_lossy(&metadata.stderr)));
        }
        let metadata: serde_json::Value =
            serde_json::from_slice(&metadata.stdout).map_err(|e| e.to_string())?;
        let packages = metadata["packages"].as_array().ok_or("Cargo packages missing")?;
        let backend = packages
            .iter()
            .find(|p| p["name"] == "lk-agents-py")
            .ok_or("Python backend package missing")?;
        let mut crates = BTreeMap::new();
        let mut pending = vec![backend];
        while let Some(package) = pending.pop() {
            let name =
                package["name"].as_str().ok_or("Cargo package name missing")?.replace('-', "_");
            if crates.contains_key(&name) {
                continue;
            }
            let manifest =
                Path::new(package["manifest_path"].as_str().ok_or("Cargo manifest missing")?);
            let directory = manifest.parent().ok_or("Cargo package directory missing")?;
            if !directory.join("src/lib.rs").is_file() {
                continue;
            }
            let relative = directory.strip_prefix(root).map_err(|e| e.to_string())?;
            let krate = Crate::read(root, &relative.to_string_lossy(), &name)?;
            crates.insert(name, krate);
            for dependency in
                package["dependencies"].as_array().ok_or("Cargo dependencies missing")?
            {
                if !dependency["kind"].is_null() {
                    continue;
                }
                let Some(path) = dependency["path"].as_str() else { continue };
                if let Some(package) = packages.iter().find(|p| {
                    p["manifest_path"].as_str().is_some_and(|manifest| {
                        Path::new(manifest).parent() == Some(Path::new(path))
                    })
                }) {
                    pending.push(package);
                }
            }
        }
        Self::new(crates)
    }

    fn new(mut crates: BTreeMap<String, Crate>) -> Result<Self, String> {
        // Resolve workspace wildcard imports from declarations too, including public re-exports.
        // Repeat until transitive re-exports stop introducing names; never infer a type by name.
        loop {
            let mut imports = Vec::new();
            for (owner, source) in &crates {
                for (module, target, public) in source.external_globs()? {
                    let (name, rest) = target.split_first().ok_or("empty wildcard import")?;
                    let Some(target_crate) = crates.get(name) else { continue };
                    let path: syn::Path = syn::parse_str(&format!(
                        "crate{}",
                        if rest.is_empty() {
                            String::new()
                        } else {
                            format!("::{}", rest.join("::"))
                        }
                    ))
                    .map_err(|e| e.to_string())?;
                    let Found::Module(target_module) = target_crate.resolve(0, &path)? else {
                        continue;
                    };
                    for name in target_crate.public_names(target_module) {
                        let mut path = target.clone();
                        path.push(name.clone());
                        imports.push((owner.clone(), module, name, path, public));
                    }
                }
            }
            let mut changed = false;
            for (owner, module, name, path, public) in imports {
                changed |= crates
                    .get_mut(&owner)
                    .expect("discovered crate")
                    .add_import(module, name, path, public);
            }
            if !changed {
                break;
            }
        }
        let routes = crates["lk_agents_runtime"].external_exports()?;
        Ok(Self { crates, routes })
    }

    fn identity(
        &self,
        krate: &str,
        module: usize,
        path: &syn::Path,
    ) -> Result<Option<(String, usize, usize)>, String> {
        let Some(source) = self.crates.get(krate) else { return Ok(None) };
        match source.resolve(module, path)? {
            Found::Item { module, item } => Ok(Some((krate.into(), module, item))),
            Found::External(path) => {
                let (krate, rest) = path.split_first().ok_or("empty external path")?;
                if !self.crates.contains_key(krate) {
                    return Ok(None);
                }
                let path = syn::parse_str(&format!("crate::{}", rest.join("::")))
                    .map_err(|e| e.to_string())?;
                self.identity(krate, 0, &path)
            }
            Found::Module(_) => Ok(None),
        }
    }

    fn reachable(&self, krate: &str, module: usize, path: &syn::Path) -> Result<String, String> {
        let Some((owner, module, item)) = self.identity(krate, module, path)? else {
            return Ok(path.to_token_stream().to_string());
        };
        let canonical = self.crates[&owner].extern_path(module, item)?;
        if owner == "lk_agents_py" {
            return Ok(canonical.replacen("::lk_agents_py", "crate", 1));
        }
        if owner == "lk_agents_abi" || owner == "lk_agents_runtime" {
            return Ok(canonical);
        }
        let segments: Vec<_> =
            canonical.trim_start_matches("::").split("::").map(str::to_owned).collect();
        let mut routes: Vec<_> =
            self.routes.iter().filter(|(target, _)| segments.starts_with(target)).collect();
        routes.sort_by_key(|(target, exported)| {
            (std::cmp::Reverse(target.len()), exported.len(), exported)
        });
        let Some((target, exported)) = routes.first() else {
            return Err(format!("{canonical}: class implementation is not exposed by the runtime"));
        };
        let suffix = &segments[target.len()..];
        Ok(if suffix.is_empty() {
            exported.clone()
        } else {
            format!("{exported}::{}", suffix.join("::"))
        })
    }
}

struct TypeName<'a> {
    sources: &'a Sources,
    krate: &'a str,
    module: usize,
    error: Option<String>,
}
impl VisitMut for TypeName<'_> {
    fn visit_type_path_mut(&mut self, ty: &mut syn::TypePath) {
        syn::visit_mut::visit_type_path_mut(self, ty);
        if self.error.is_some() || ty.qself.is_some() {
            return;
        }
        let arguments = ty.path.segments.last().map(|s| s.arguments.clone());
        match self
            .sources
            .reachable(self.krate, self.module, &ty.path)
            .and_then(|name| syn::parse_str::<syn::Path>(&name).map_err(|e| e.to_string()))
        {
            Ok(mut path) => {
                if let (Some(last), Some(arguments)) = (path.segments.last_mut(), arguments) {
                    last.arguments = arguments;
                }
                ty.path = path;
            }
            Err(error) => self.error = Some(error),
        }
    }
}

pub fn emit(root: &Path, declarations: &Declarations) -> Result<Out, String> {
    let sources = Sources::read(root)?;
    select(&sources, declarations)
}

fn select(sources: &Sources, declarations: &Declarations) -> Result<Out, String> {
    let classes: BTreeMap<_, _> = declarations
        .classes
        .iter()
        .map(|class| {
            let path = syn::parse_str(&class.trait_path).map_err(|e| e.to_string())?;
            let identity = sources
                .identity("lk_agents_abi", 0, &path)?
                .ok_or_else(|| format!("{}: missing class declaration", class.trait_path))?;
            Ok((identity, class))
        })
        .collect::<Result<_, String>>()?;
    let mut selected = BTreeMap::new();
    for (name, krate) in &sources.crates {
        for (module, scope) in krate.modules.iter().enumerate() {
            for item in &scope.items {
                let syn::Item::Impl(implementation) = item else { continue };
                if implementation.attrs.iter().any(|a| {
                    a.path().is_ident("cfg")
                        && a.parse_args::<syn::Ident>().is_ok_and(|i| i == "test")
                }) {
                    continue;
                }
                let Some((None, path, _)) = &implementation.trait_ else { continue };
                let Some(identity) = sources
                    .identity(name, module, path)
                    .map_err(|e| format!("{}: {}: {e}", scope.file, path.to_token_stream()))?
                else {
                    continue;
                };
                let Some(class) = classes.get(&identity) else { continue };
                if !implementation.generics.params.is_empty() {
                    return Err(format!(
                        "{}: exported class {} needs a concrete implementation",
                        scope.file, class.rust_name
                    ));
                }
                let mut ty = (*implementation.self_ty).clone();
                let mut rewrite = TypeName { sources, krate: name, module, error: None };
                rewrite.visit_type_mut(&mut ty);
                if let Some(error) = rewrite.error {
                    return Err(error);
                }
                let ty = ty.to_token_stream().to_string();
                if let Some(previous) = selected.insert(class.rust_name.clone(), ty.clone()) {
                    return Err(format!(
                        "{}: ambiguous implementations {previous} and {ty}",
                        class.trait_path
                    ));
                }
            }
        }
    }
    let mut text = String::from(
        "//! GENERATED FILE -- DO NOT EDIT.\n//! Concrete implementations discovered from the ABI classes' Rust impls.\npub struct Selected;\nimpl super::Implementations for Selected {\n",
    );
    for class in &declarations.classes {
        let ty = selected
            .get(&class.rust_name)
            .ok_or_else(|| format!("{}: no concrete implementation found", class.trait_path))?;
        text.push_str(&format!("type {} = {ty};\n", class.rust_name));
    }
    text.push_str("}\n");
    Ok(Out { path: "implementations.rs".into(), text })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model(implementation: &str, facade: &str) -> (Sources, Declarations) {
        let abi = Crate::from_files(
            "lk_agents_abi",
            &[(
                "lib.rs",
                r#"
            pub mod rules { #[host(functions)] pub trait Rules { #[host(function)] fn check() -> bool; } }
            pub use rules::Rules;
        "#,
            )],
        )
        .unwrap();
        let declarations = crate::decl::resolve::declarations(&abi).unwrap();
        let runtime = Crate::from_files("lk_agents_runtime", &[("lib.rs", facade)]).unwrap();
        let core = Crate::from_files("lk_agents_core", &[("lib.rs", implementation)]).unwrap();
        (
            Sources::new(BTreeMap::from([
                ("lk_agents_abi".into(), abi),
                ("lk_agents_runtime".into(), runtime),
                ("lk_agents_core".into(), core),
            ]))
            .unwrap(),
            declarations,
        )
    }

    #[test]
    fn selection_follows_trait_identity_concrete_arguments_and_runtime_reexports() {
        let (sources, declarations) = model(
            r#"
            mod imports { pub use lk_agents_abi::rules::{self as contract}; pub use contract::*; }
            use imports::Rules as Contract;
            pub struct Generic<T>(T);
            impl Contract for Generic<u8> {}
            pub trait Rules {}
            pub struct Unrelated;
            impl Rules for Unrelated {}
            #[cfg(test)] impl Contract for Unrelated {}
        "#,
            "pub use lk_agents_core::{self as algorithms};",
        );
        let text = select(&sources, &declarations).unwrap().text;
        assert!(
            text.contains("type Rules = :: lk_agents_runtime :: algorithms :: Generic < u8 >"),
            "{text}"
        );
        assert!(!text.contains("Unrelated"));
    }

    #[test]
    fn missing_ambiguous_generic_and_unreachable_selections_fail() {
        for (implementation, facade, error) in [
            ("pub struct A;", "pub use lk_agents_core as cores;", "no concrete implementation"),
            (
                "pub struct A; pub struct B; impl lk_agents_abi::Rules for A {} impl lk_agents_abi::Rules for B {}",
                "pub use lk_agents_core as cores;",
                "ambiguous implementations",
            ),
            (
                "pub struct A<T>(T); impl<T> lk_agents_abi::Rules for A<T> {}",
                "pub use lk_agents_core as cores;",
                "concrete implementation",
            ),
            ("pub struct A; impl lk_agents_abi::Rules for A {}", "", "not exposed by the runtime"),
        ] {
            let (sources, declarations) = model(implementation, facade);
            assert!(select(&sources, &declarations).err().unwrap().contains(error));
        }
    }
}
