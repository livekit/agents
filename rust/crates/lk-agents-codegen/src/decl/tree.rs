//! The ABI crate as rustc lays it out: modules found from `src/lib.rs` through `mod` items, each
//! with its items and its `use` names, and path resolution over them. This is how declarations are
//! discovered (no source list) and how a type a declaration names is found (never by its last
//! segment).

use std::path::Path;

/// One module of the crate. `path` is from the crate root: `["llm", "llm"]`.
pub struct Module {
    pub path: Vec<String>,
    /// Repo-relative, for messages and generated headers.
    pub file: String,
    pub items: Vec<syn::Item>,
    parent: Option<usize>,
    uses: Vec<Use>,
}

enum Use {
    /// `use a::b::C;` / `use a::b::C as D;`: `name` stands for `path`.
    Name { name: String, path: Vec<String> },
    /// `use a::b::*;`
    Glob(Vec<String>),
}

/// A path's meaning.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Found {
    /// An item of the crate: `items[item]` of `modules[module]`.
    Item { module: usize, item: usize },
    /// A module of the crate.
    Module(usize),
    /// Outside the crate, canonical (`core::`/`alloc::` read as `std::`): `["serde_json", "Value"]`.
    External(Vec<String>),
}

pub struct Crate {
    /// Root first.
    pub modules: Vec<Module>,
    /// The name other crates use for this one (`lk_agents_abi`).
    name: String,
}

/// The prelude names a declaration may write bare, as their canonical paths.
const PRELUDE: &[(&str, &[&str])] = &[
    ("String", &["std", "string", "String"]),
    ("Vec", &["std", "vec", "Vec"]),
    ("Option", &["std", "option", "Option"]),
    ("Result", &["std", "result", "Result"]),
    ("Box", &["std", "boxed", "Box"]),
];

impl Crate {
    /// Reads the crate whose `src/lib.rs` is under `dir` (repo-relative).
    pub fn read(root: &Path, dir: &str, name: &str) -> Result<Crate, String> {
        let mut krate = Crate { modules: Vec::new(), name: name.to_owned() };
        let lib = format!("{dir}/src/lib.rs");
        let text = std::fs::read_to_string(root.join(&lib)).map_err(|e| format!("{lib}: {e}"))?;
        let file = syn::parse_file(&text).map_err(|e| format!("{lib}: {e}"))?;
        krate.add(root, Vec::new(), lib, format!("{dir}/src"), file.items, None)?;
        Ok(krate)
    }

    /// A crate from in-memory files (`path -> text`), for tests: `lib.rs` is the root.
    #[cfg(test)]
    pub fn from_files(name: &str, files: &[(&str, &str)]) -> Result<Crate, String> {
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("lk-codegen-{}-{n}", std::process::id()));
        for (path, text) in files {
            let at = dir.join("c/src").join(path);
            std::fs::create_dir_all(at.parent().expect("has a parent"))
                .map_err(|e| e.to_string())?;
            std::fs::write(at, text).map_err(|e| e.to_string())?;
        }
        let krate = Crate::read(&dir, "c", name);
        let _ = std::fs::remove_dir_all(&dir);
        krate
    }

    fn add(
        &mut self,
        root: &Path,
        path: Vec<String>,
        file: String,
        dir: String,
        items: Vec<syn::Item>,
        parent: Option<usize>,
    ) -> Result<usize, String> {
        let index = self.modules.len();
        let uses = items
            .iter()
            .filter_map(|i| match i {
                syn::Item::Use(u) if !test_only(&u.attrs) => Some(u),
                _ => None,
            })
            .flat_map(|u| flatten_use(&u.tree, Vec::new()))
            .collect();
        self.modules.push(Module {
            path: path.clone(),
            file: file.clone(),
            items: Vec::new(),
            parent,
            uses,
        });
        let mut kept = Vec::new();
        for item in items {
            let syn::Item::Mod(m) = &item else {
                kept.push(item);
                continue;
            };
            if test_only(&m.attrs) {
                continue;
            }
            let name = m.ident.to_string();
            let mut child = path.clone();
            child.push(name.clone());
            match &m.content {
                Some((_, inner)) => {
                    self.add(
                        root,
                        child,
                        file.clone(),
                        format!("{dir}/{name}"),
                        inner.clone(),
                        Some(index),
                    )?;
                }
                None => {
                    let (child_file, child_dir) = locate(root, &dir, &file, &name)?;
                    let text = std::fs::read_to_string(root.join(&child_file))
                        .map_err(|e| format!("{child_file}: {e}"))?;
                    let parsed =
                        syn::parse_file(&text).map_err(|e| format!("{child_file}: {e}"))?;
                    self.add(root, child, child_file, child_dir, parsed.items, Some(index))?;
                }
            }
            kept.push(item);
        }
        self.modules[index].items = kept;
        Ok(index)
    }

    /// What `path`, written in `module`, names. Generic arguments are the caller's: only the
    /// segments' identifiers are read.
    pub fn resolve(&self, module: usize, path: &syn::Path) -> Result<Found, String> {
        let segments: Vec<String> = path.segments.iter().map(|s| s.ident.to_string()).collect();
        if path.leading_colon.is_some() {
            return self.external_or_self(&segments);
        }
        self.walk(module, &segments, &mut Vec::new())
    }

    fn walk(
        &self,
        module: usize,
        segments: &[String],
        seen: &mut Vec<(usize, String)>,
    ) -> Result<Found, String> {
        let Some((first, rest)) = segments.split_first() else {
            return Ok(Found::Module(module));
        };
        let start = match first.as_str() {
            "crate" => Found::Module(0),
            "self" => Found::Module(module),
            "super" => {
                Found::Module(self.modules[module].parent.ok_or("`super` above the crate root")?)
            }
            name if *name == self.name => Found::Module(0),
            name => match self.lookup(module, name, seen)? {
                Some(found) => found,
                None => return self.external_or_self(segments),
            },
        };
        self.descend(start, rest, seen)
    }

    fn descend(
        &self,
        mut at: Found,
        rest: &[String],
        seen: &mut Vec<(usize, String)>,
    ) -> Result<Found, String> {
        for (i, segment) in rest.iter().enumerate() {
            at = match at {
                Found::External(mut path) => {
                    path.push(segment.clone());
                    Found::External(path)
                }
                Found::Module(m) if segment == "super" => {
                    Found::Module(self.modules[m].parent.ok_or("`super` above the crate root")?)
                }
                Found::Module(m) => self
                    .lookup(m, segment, seen)?
                    .ok_or_else(|| format!("`{}` has no item `{segment}`", self.display(m)))?,
                Found::Item { module, item } => {
                    // `Enum::Variant` or `Type::Assoc`: not a type a declaration may name.
                    return Err(format!(
                        "`{}::{}` names inside an item of `{}`",
                        self.item_name(module, item),
                        rest[i..].join("::"),
                        self.display(module)
                    ));
                }
            };
        }
        Ok(at)
    }

    /// `name` in `module`'s namespace: its items and submodules, then its `use`s, then its globs.
    fn lookup(
        &self,
        module: usize,
        name: &str,
        seen: &mut Vec<(usize, String)>,
    ) -> Result<Option<Found>, String> {
        if seen.contains(&(module, name.to_owned())) {
            return Ok(None);
        }
        seen.push((module, name.to_owned()));
        let m = &self.modules[module];
        if let Some(item) = m.items.iter().position(|i| item_ident(i).is_some_and(|n| n == name)) {
            if let syn::Item::Mod(_) = &m.items[item] {
                let mut path = m.path.clone();
                path.push(name.to_owned());
                let child =
                    self.modules.iter().position(|c| c.path == path).ok_or("unread module")?;
                return Ok(Some(Found::Module(child)));
            }
            return Ok(Some(Found::Item { module, item }));
        }
        for u in &m.uses {
            if let Use::Name { name: n, path } = u
                && n == name
            {
                return self.walk(module, path, seen).map(Some);
            }
        }
        for u in &m.uses {
            if let Use::Glob(path) = u
                && let Found::Module(target) = self.walk(module, path, seen)?
                && let Some(found) = self.lookup(target, name, seen)?
            {
                return Ok(Some(found));
            }
        }
        Ok(None)
    }

    fn external_or_self(&self, segments: &[String]) -> Result<Found, String> {
        if segments.first().is_some_and(|s| *s == self.name) {
            return self.descend(Found::Module(0), &segments[1..], &mut Vec::new());
        }
        if let [one] = segments
            && let Some((_, path)) = PRELUDE.iter().find(|(n, _)| n == one)
        {
            return Ok(Found::External(path.iter().map(|s| (*s).to_owned()).collect()));
        }
        let mut path = segments.to_vec();
        if matches!(path[0].as_str(), "core" | "alloc") {
            path[0] = "std".into();
        }
        Ok(Found::External(path))
    }

    /// `crate::a::b` for messages.
    pub fn display(&self, module: usize) -> String {
        std::iter::once("crate")
            .chain(self.modules[module].path.iter().map(String::as_str))
            .collect::<Vec<_>>()
            .join("::")
    }

    pub fn item_name(&self, module: usize, item: usize) -> String {
        item_ident(&self.modules[module].items[item]).unwrap_or_default()
    }

    /// The item's path as another crate writes it: `::lk_agents_abi::llm::llm::Llm`.
    pub fn extern_path(&self, module: usize, item: usize) -> String {
        let mut path = vec![String::new(), self.name.clone()];
        path.extend(self.modules[module].path.iter().cloned());
        path.push(self.item_name(module, item));
        path.join("::")
    }
}

fn test_only(attrs: &[syn::Attribute]) -> bool {
    attrs.iter().any(|a| {
        a.path().is_ident("cfg") && a.parse_args::<syn::Ident>().is_ok_and(|i| i == "test")
    })
}

fn item_ident(item: &syn::Item) -> Option<String> {
    Some(match item {
        syn::Item::Struct(i) => i.ident.to_string(),
        syn::Item::Enum(i) => i.ident.to_string(),
        syn::Item::Trait(i) => i.ident.to_string(),
        syn::Item::Type(i) => i.ident.to_string(),
        syn::Item::Fn(i) => i.sig.ident.to_string(),
        syn::Item::Const(i) => i.ident.to_string(),
        syn::Item::Static(i) => i.ident.to_string(),
        syn::Item::Mod(i) => i.ident.to_string(),
        syn::Item::Union(i) => i.ident.to_string(),
        _ => return None,
    })
}

fn flatten_use(tree: &syn::UseTree, prefix: Vec<String>) -> Vec<Use> {
    let with = |s: String| {
        let mut p = prefix.clone();
        p.push(s);
        p
    };
    match tree {
        syn::UseTree::Path(p) => flatten_use(&p.tree, with(p.ident.to_string())),
        syn::UseTree::Name(n) if n.ident == "self" => {
            let name = prefix.last().cloned().unwrap_or_default();
            vec![Use::Name { name, path: prefix }]
        }
        syn::UseTree::Name(n) => {
            vec![Use::Name { name: n.ident.to_string(), path: with(n.ident.to_string()) }]
        }
        syn::UseTree::Rename(r) => {
            vec![Use::Name { name: r.rename.to_string(), path: with(r.ident.to_string()) }]
        }
        syn::UseTree::Glob(_) => vec![Use::Glob(prefix)],
        syn::UseTree::Group(g) => {
            g.items.iter().flat_map(|t| flatten_use(t, prefix.clone())).collect()
        }
    }
}

/// `mod name;` declared in `file`: `dir/name.rs` or `dir/name/mod.rs`, where `dir` is the
/// directory that owns the declaring file's children.
fn locate(root: &Path, dir: &str, file: &str, name: &str) -> Result<(String, String), String> {
    for candidate in [format!("{dir}/{name}.rs"), format!("{dir}/{name}/mod.rs")] {
        if root.join(&candidate).exists() {
            return Ok((candidate, format!("{dir}/{name}")));
        }
    }
    Err(format!("{file}: `mod {name};` has no file under {dir}"))
}
