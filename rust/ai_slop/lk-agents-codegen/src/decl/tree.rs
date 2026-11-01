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
    /// Declared `pub mod` (the root is public): another crate can name what is inside.
    public: bool,
    /// Each `use` and whether it is `pub use`.
    uses: Vec<(Use, bool)>,
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
        krate.add(root, Vec::new(), lib, format!("{dir}/src"), file.items, None, true)?;
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

    #[allow(
        clippy::too_many_arguments,
        reason = "Each argument describes one module being discovered"
    )]
    fn add(
        &mut self,
        root: &Path,
        path: Vec<String>,
        file: String,
        dir: String,
        items: Vec<syn::Item>,
        parent: Option<usize>,
        public: bool,
    ) -> Result<usize, String> {
        let index = self.modules.len();
        let uses = items
            .iter()
            .filter_map(|i| match i {
                syn::Item::Use(u) if !test_only(&u.attrs) => Some(u),
                _ => None,
            })
            .flat_map(|u| {
                let public = is_pub(&u.vis);
                flatten_use(&u.tree, Vec::new()).into_iter().map(move |use_| (use_, public))
            })
            .collect();
        self.modules.push(Module {
            path: path.clone(),
            file: file.clone(),
            items: Vec::new(),
            parent,
            public,
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
            let child_public = is_pub(&m.vis);
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
                        child_public,
                    )?;
                }
                None => {
                    let (child_file, child_dir) = match module_path(&m.attrs)? {
                        Some(path) => {
                            let path = Path::new(&file)
                                .parent()
                                .expect("a source file has a directory")
                                .join(path);
                            let children = if path.file_name().is_some_and(|name| name == "mod.rs")
                            {
                                path.parent().unwrap().to_path_buf()
                            } else {
                                path.with_extension("")
                            };
                            (
                                path.to_string_lossy().into_owned(),
                                children.to_string_lossy().into_owned(),
                            )
                        }
                        None => locate(root, &dir, &file, &name)?,
                    };
                    let text = std::fs::read_to_string(root.join(&child_file))
                        .map_err(|e| format!("{child_file}: {e}"))?;
                    // Generated output is never input to declaration discovery.
                    if text.starts_with("//! GENERATED FILE -- DO NOT EDIT.") {
                        continue;
                    }
                    let parsed =
                        syn::parse_file(&text).map_err(|e| format!("{child_file}: {e}"))?;
                    self.add(
                        root,
                        child,
                        child_file,
                        child_dir,
                        parsed.items,
                        Some(index),
                        child_public,
                    )?;
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
        for (u, _) in &m.uses {
            if let Use::Name { name: n, path } = u
                && n == name
            {
                return self.walk(module, path, seen).map(Some);
            }
        }
        for (u, _) in &m.uses {
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

    /// The item's shortest public path as another crate writes it
    /// (`::lk_agents_abi::errors::ApiError`, re-exported from a private `api` module): its own
    /// module when every module on the way is `pub`, else the nearest `pub use` of it.
    pub fn extern_path(&self, module: usize, item: usize) -> Result<String, String> {
        let name = self.item_name(module, item);
        let target = Found::Item { module, item };
        let own = is_pub(item_vis(&self.modules[module].items[item]));
        // Breadth first from the root through public modules: the first hit is the shortest.
        let mut queue = std::collections::VecDeque::from([0]);
        while let Some(m) = queue.pop_front() {
            let at = &self.modules[m];
            let reexported = at.uses.iter().filter(|(_, public)| *public).find_map(|(u, _)| {
                let alias = match u {
                    Use::Name { name, .. } => name.as_str(),
                    Use::Glob(_) => name.as_str(),
                };
                let found = self.lookup(m, alias, &mut Vec::new()).ok().flatten()?;
                (found == target).then(|| alias.to_owned())
            });
            if let Some(alias) = (m == module && own).then(|| name.clone()).or(reexported) {
                let mut path = vec![String::new(), self.name.clone()];
                path.extend(at.path.iter().cloned());
                path.push(alias);
                return Ok(path.join("::"));
            }
            queue.extend(
                (0..self.modules.len())
                    .filter(|c| self.modules[*c].parent == Some(m) && self.modules[*c].public),
            );
        }
        Err(format!(
            "`{name}` of `{}` is not reachable from outside the crate",
            self.display(module)
        ))
    }

    /// Public re-exports of external items or crates, used to reach implementations through
    /// the runtime facade without adding dependencies to a language backend.
    pub fn external_exports(&self) -> Result<Vec<(Vec<String>, String)>, String> {
        let mut exports = Vec::new();
        let mut queue = std::collections::VecDeque::from([0]);
        while let Some(m) = queue.pop_front() {
            let module = &self.modules[m];
            for (entry, public) in &module.uses {
                if !public {
                    continue;
                }
                let Use::Name { name, .. } = entry else { continue };
                let path: syn::Path = syn::parse_str(name).map_err(|e| e.to_string())?;
                if let Found::External(target) = self.resolve(m, &path)? {
                    let mut exported = vec![self.name.clone()];
                    exported.extend(module.path.iter().cloned());
                    exported.push(name.clone());
                    exports.push((target, format!("::{}", exported.join("::"))));
                }
            }
            queue.extend(
                (0..self.modules.len())
                    .filter(|c| self.modules[*c].parent == Some(m) && self.modules[*c].public),
            );
        }
        Ok(exports)
    }

    /// External wildcard imports. A workspace reader can expand these from the referenced
    /// module's public names before resolving implementations across crate boundaries.
    pub fn external_globs(&self) -> Result<Vec<(usize, Vec<String>, bool)>, String> {
        let mut globs = Vec::new();
        for (m, module) in self.modules.iter().enumerate() {
            for (entry, public) in &module.uses {
                if let Use::Glob(path) = entry
                    && let Found::External(path) = self.walk(m, path, &mut Vec::new())?
                {
                    globs.push((m, path, *public));
                }
            }
        }
        Ok(globs)
    }

    pub fn public_names(&self, module: usize) -> Vec<String> {
        let scope = &self.modules[module];
        let mut names: Vec<_> = scope
            .items
            .iter()
            .filter(|item| is_pub(item_vis(item)))
            .filter_map(item_ident)
            .collect();
        names.extend(scope.uses.iter().filter_map(|(entry, public)| match entry {
            Use::Name { name, .. } if *public => Some(name.clone()),
            _ => None,
        }));
        names.sort();
        names.dedup();
        names
    }

    pub fn add_import(
        &mut self,
        module: usize,
        name: String,
        path: Vec<String>,
        public: bool,
    ) -> bool {
        let uses = &mut self.modules[module].uses;
        if uses.iter().any(|(entry, p)| {
            matches!(entry, Use::Name { name: n, path: target } if *n == name && *target == path)
                && *p == public
        }) {
            return false;
        }
        uses.push((Use::Name { name, path }, public));
        true
    }

    /// Resolve a data constant without evaluating Rust code. Only literals, arrays, record
    /// literals, references, option constructors and aliases to those constants are accepted.
    pub fn constant(&self, module: usize, value: &syn::Expr) -> Result<syn::Expr, String> {
        self.constant_at(module, value, &mut Vec::new())
    }

    fn constant_at(
        &self,
        module: usize,
        value: &syn::Expr,
        stack: &mut Vec<(usize, usize)>,
    ) -> Result<syn::Expr, String> {
        Ok(match value {
            syn::Expr::Lit(_) => value.clone(),
            syn::Expr::Reference(reference) if reference.mutability.is_none() => {
                self.constant_at(module, &reference.expr, stack)?
            }
            syn::Expr::Paren(paren) => self.constant_at(module, &paren.expr, stack)?,
            syn::Expr::Unary(unary) if matches!(unary.op, syn::UnOp::Neg(_)) => {
                let mut unary = unary.clone();
                unary.expr = Box::new(self.constant_at(module, &unary.expr, stack)?);
                syn::Expr::Unary(unary)
            }
            syn::Expr::Array(array) => {
                let mut array = array.clone();
                array.elems = array
                    .elems
                    .iter()
                    .map(|value| self.constant_at(module, value, stack))
                    .collect::<Result<_, _>>()?;
                syn::Expr::Array(array)
            }
            syn::Expr::Call(call)
                if matches!(call.func.as_ref(), syn::Expr::Path(path) if path.path.is_ident("Some"))
                    && call.args.len() == 1 =>
            {
                self.constant_at(module, &call.args[0], stack)?
            }
            syn::Expr::Path(path) if path.path.is_ident("None") => value.clone(),
            syn::Expr::Path(path) => {
                let Found::Item { module, item } = self.resolve(module, &path.path)? else {
                    return Err("a constant alias must name a declared constant".into());
                };
                let syn::Item::Const(item_value) = &self.modules[module].items[item] else {
                    return Err("a constant alias must name a declared constant".into());
                };
                if stack.contains(&(module, item)) {
                    return Err("cyclic constant aliases".into());
                }
                stack.push((module, item));
                let value = self.constant_at(module, &item_value.expr, stack)?;
                stack.pop();
                value
            }
            syn::Expr::Struct(record) if record.rest.is_none() => {
                let mut record = record.clone();
                let Found::Item { module: owner, item } = self.resolve(module, &record.path)?
                else {
                    return Err("a record constant names its declared type".into());
                };
                record.path =
                    syn::parse_str(&self.extern_path(owner, item)?).map_err(|e| e.to_string())?;
                for field in &mut record.fields {
                    field.expr = self.constant_at(module, &field.expr, stack)?;
                }
                syn::Expr::Struct(record)
            }
            _ => return Err("a constant must be explicit data, not executable Rust".into()),
        })
    }
}

fn is_pub(vis: &syn::Visibility) -> bool {
    matches!(vis, syn::Visibility::Public(_))
}

fn item_vis(item: &syn::Item) -> &syn::Visibility {
    match item {
        syn::Item::Struct(i) => &i.vis,
        syn::Item::Enum(i) => &i.vis,
        syn::Item::Trait(i) => &i.vis,
        syn::Item::Type(i) => &i.vis,
        syn::Item::Fn(i) => &i.vis,
        syn::Item::Const(i) => &i.vis,
        syn::Item::Static(i) => &i.vis,
        syn::Item::Mod(i) => &i.vis,
        syn::Item::Union(i) => &i.vis,
        _ => &syn::Visibility::Inherited,
    }
}

fn module_path(attrs: &[syn::Attribute]) -> Result<Option<String>, String> {
    let Some(attr) = attrs.iter().find(|attr| attr.path().is_ident("path")) else {
        return Ok(None);
    };
    if let syn::Meta::NameValue(value) = &attr.meta
        && let syn::Expr::Lit(syn::ExprLit { lit: syn::Lit::Str(path), .. }) = &value.value
    {
        return Ok(Some(path.value()));
    }
    Err("a module's #[path] is a string literal".into())
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
            let path = if r.ident == "self" { prefix } else { with(r.ident.to_string()) };
            vec![Use::Name { name: r.rename.to_string(), path }]
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn follows_explicit_module_paths_without_renaming_the_source() {
        let krate = Crate::from_files(
            "abi",
            &[
                ("lib.rs", "pub mod outer;"),
                ("outer/mod.rs", "#[path = \"legacy.rs\"] pub mod types;"),
                ("outer/legacy.rs", "pub struct Value;"),
            ],
        )
        .unwrap();
        let path: syn::Path = syn::parse_quote!(crate::outer::types::Value);
        let Found::Item { module, item } = krate.resolve(0, &path).unwrap() else {
            panic!("a declared type")
        };
        assert_eq!(krate.modules[module].file, "c/src/outer/legacy.rs");
        assert!(
            matches!(&krate.modules[module].items[item], syn::Item::Struct(item) if item.ident == "Value")
        );
    }
}
