//! Fixed Python member names belong to generated accessors, including calls through the generic
//! extraction helpers. Dynamic application keys and reflected Python field objects remain legal.
use std::collections::BTreeSet;
use std::path::Path;
use syn::{spanned::Spanned, visit::Visit};

pub fn check(root: &Path) -> Result<(), String> {
    let mut errors = Vec::new();
    scan_dir(&root.join("rust/crates/lk-agents-py/src"), &mut errors)?;
    if errors.is_empty() { Ok(()) } else { Err(errors.join("\n")) }
}

fn scan_dir(dir: &Path, errors: &mut Vec<String>) -> Result<(), String> {
    for entry in std::fs::read_dir(dir).map_err(|e| e.to_string())? {
        let path = entry.map_err(|e| e.to_string())?.path();
        if path.is_dir() {
            scan_dir(&path, errors)?;
        } else if path.extension().is_some_and(|e| e == "rs")
            && !path.file_stem().is_some_and(|n| n.to_string_lossy().ends_with("_tests"))
        {
            let source = std::fs::read_to_string(&path).map_err(|e| e.to_string())?;
            if source.starts_with("//! GENERATED FILE -- DO NOT EDIT.\n") {
                continue;
            }
            for line in violations_at(&source, path.file_name().unwrap().to_str().unwrap())? {
                errors.push(format!(
                    "{}:{line}: Python record/member access must be generated; runtime keys are restricted to data/reflection primitives",
                    path.display()
                ));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
fn violations(source: &str) -> Result<Vec<usize>, String> {
    violations_at(source, "business.rs")
}

fn violations_at(source: &str, file: &str) -> Result<Vec<usize>, String> {
    let parsed = syn::parse_file(source).map_err(|e| e.to_string())?;
    let mut scan = Members { file: file.into(), ..Members::default() };
    scan.visit_file(&parsed);
    Ok(scan.errors)
}

#[derive(Default)]
struct Members {
    file: String,
    function: String,
    parameters: BTreeSet<String>,
    constants: BTreeSet<String>,
    errors: Vec<usize>,
}

impl Members {
    fn bridge(&self) -> bool {
        match self.file.as_str() {
            "py_values.rs" => [
                "attr",
                "attr_with",
                "attr_or",
                "attribute_missing",
                "item",
                "item_with",
                "item_or",
                "item_missing",
            ]
            .contains(&self.function.as_str()),
            "py_errors.rs" => {
                ["host_exception", "host_exception_with"].contains(&self.function.as_str())
            }
            "py_calls.rs" => self.function == "read",
            _ => false,
        }
    }
    fn runtime_key(&self, method: &str, expr: &syn::Expr) -> bool {
        if self.bridge() {
            return matches!(expr, syn::Expr::Path(path) if path.path.get_ident().is_some_and(|name| self.parameters.contains(&name.to_string())));
        }
        if method == "get_item"
            && matches!(expr, syn::Expr::Lit(lit) if matches!(lit.lit, syn::Lit::Int(_)))
        {
            return true;
        }
        // These functions serialize user-supplied map entries. They do not select framework fields.
        if method == "set_item"
            && matches!(
                (self.file.as_str(), self.function.as_str()),
                ("py_values.rs", "extra_to_py" | "value_to_py" | "map_to_py")
                    | ("py_logging.rs", "emit")
            )
        {
            return matches!(expr, syn::Expr::Path(path) if path.path.is_ident("key"));
        }
        // A reflected dataclass field is a Python str object obtained from its declared fields.
        self.file == "py_values.rs"
            && self.function == "record_fields"
            && method == "getattr"
            && matches!(expr, syn::Expr::Try(value) if matches!(value.expr.as_ref(), syn::Expr::MethodCall(call) if call.method == "cast" && call.turbofish.as_ref().is_some_and(|args| args.args.iter().any(|arg| matches!(arg, syn::GenericArgument::Type(syn::Type::Path(ty)) if ty.path.is_ident("PyString"))))))
    }
    fn enter(&mut self, signature: &syn::Signature) -> (String, BTreeSet<String>) {
        let previous = (
            std::mem::replace(&mut self.function, signature.ident.to_string()),
            std::mem::take(&mut self.parameters),
        );
        self.parameters = signature
            .inputs
            .iter()
            .filter_map(|arg| match arg {
                syn::FnArg::Typed(arg) => match arg.pat.as_ref() {
                    syn::Pat::Ident(name) => Some(name.ident.to_string()),
                    _ => None,
                },
                _ => None,
            })
            .collect();
        previous
    }
    fn fixed(&self, expr: &syn::Expr) -> bool {
        use syn::Expr as E;
        match expr {
            E::Lit(lit) => {
                matches!(lit.lit, syn::Lit::Str(_) | syn::Lit::CStr(_) | syn::Lit::ByteStr(_))
            }
            E::Reference(e) => self.fixed(&e.expr),
            E::Paren(e) => self.fixed(&e.expr),
            E::Group(e) => self.fixed(&e.expr),
            E::Path(e) => {
                e.path.get_ident().is_some_and(|n| self.constants.contains(&n.to_string()))
            }
            E::MethodCall(e) if e.args.is_empty() => self.fixed(&e.receiver),
            E::Macro(e) => e
                .mac
                .tokens
                .clone()
                .into_iter()
                .any(|t| matches!(t, proc_macro2::TokenTree::Literal(_))),
            _ => false,
        }
    }
    fn key(&mut self, expr: Option<&syn::Expr>) {
        if let Some(expr) = expr
            && self.fixed(expr)
        {
            self.errors.push(expr.span().start().line);
        }
    }
}

impl<'ast> Visit<'ast> for Members {
    fn visit_item_fn(&mut self, node: &'ast syn::ItemFn) {
        let before = self.constants.clone();
        let previous = self.enter(&node.sig);
        syn::visit::visit_item_fn(self, node);
        (self.function, self.parameters) = previous;
        self.constants = before;
    }
    fn visit_impl_item_fn(&mut self, node: &'ast syn::ImplItemFn) {
        let before = self.constants.clone();
        let previous = self.enter(&node.sig);
        syn::visit::visit_impl_item_fn(self, node);
        (self.function, self.parameters) = previous;
        self.constants = before;
    }

    fn visit_macro(&mut self, node: &'ast syn::Macro) {
        fn tokens(stream: proc_macro2::TokenStream, errors: &mut Vec<usize>) {
            use proc_macro2::{Delimiter, TokenTree};
            let trees: Vec<_> = stream.into_iter().collect();
            for pair in trees.windows(2) {
                if let [TokenTree::Ident(name), TokenTree::Group(args)] = pair
                    && args.delimiter() == Delimiter::Parenthesis
                    && [
                        "getattr",
                        "setattr",
                        "hasattr",
                        "delattr",
                        "get_item",
                        "set_item",
                        "call_method",
                        "call_method0",
                        "call_method1",
                    ]
                    .iter()
                    .any(|method| name == method)
                    && let Some(TokenTree::Literal(value)) = args.stream().into_iter().next()
                    && syn::parse_str::<syn::LitStr>(&value.to_string()).is_ok()
                {
                    errors.push(value.span().start().line);
                }
            }
            for tree in trees {
                if let TokenTree::Group(group) = tree {
                    tokens(group.stream(), errors);
                }
            }
        }
        tokens(node.tokens.clone(), &mut self.errors);
    }
    fn visit_item_use(&mut self, node: &'ast syn::ItemUse) {
        fn generated(tree: &syn::UseTree) -> bool {
            match tree {
                syn::UseTree::Path(path) => path.ident == "generated" || generated(&path.tree),
                syn::UseTree::Group(group) => group.items.iter().any(generated),
                syn::UseTree::Name(name) => name.ident == "generated",
                syn::UseTree::Rename(name) => name.ident == "generated",
                _ => false,
            }
        }
        fn bridge_import(tree: &syn::UseTree, namespace: &str) -> bool {
            match tree {
                syn::UseTree::Path(path) => bridge_import(
                    &path.tree,
                    if ["py_values", "py_calls", "py_errors"].iter().any(|name| path.ident == name)
                    {
                        if path.ident == "py_values" {
                            "py_values"
                        } else if path.ident == "py_calls" {
                            "py_calls"
                        } else {
                            "py_errors"
                        }
                    } else {
                        namespace
                    },
                ),
                syn::UseTree::Group(group) => {
                    group.items.iter().any(|tree| bridge_import(tree, namespace))
                }
                syn::UseTree::Name(name) => bridge_name(namespace, &name.ident.to_string()),
                syn::UseTree::Rename(name) => bridge_name(namespace, &name.ident.to_string()),
                _ => false,
            }
        }
        fn bridge_name(namespace: &str, name: &str) -> bool {
            match namespace {
                "py_values" => {
                    ["attr", "attr_with", "attr_or", "item", "item_with", "item_or"].contains(&name)
                }
                "py_calls" => name == "read",
                "py_errors" => ["host_exception", "host_exception_with"].contains(&name),
                _ => false,
            }
        }
        if bridge_import(&node.tree, "") {
            self.errors.push(node.span().start().line);
        }
        if !matches!(node.vis, syn::Visibility::Inherited) && generated(&node.tree) {
            self.errors.push(node.span().start().line);
        }
        syn::visit::visit_item_use(self, node);
    }
    fn visit_item_mod(&mut self, node: &'ast syn::ItemMod) {
        if node.attrs.iter().any(|a| {
            a.path().is_ident("cfg")
                && a.meta.require_list().is_ok_and(|m| m.tokens.to_string() == "test")
        }) {
            return;
        }
        syn::visit::visit_item_mod(self, node);
    }
    fn visit_item_const(&mut self, node: &'ast syn::ItemConst) {
        if self.fixed(&node.expr) {
            self.constants.insert(node.ident.to_string());
        }
        syn::visit::visit_item_const(self, node);
    }
    fn visit_local(&mut self, node: &'ast syn::Local) {
        if let (syn::Pat::Ident(name), Some(init)) = (&node.pat, &node.init)
            && self.fixed(&init.expr)
        {
            self.constants.insert(name.ident.to_string());
        }
        syn::visit::visit_local(self, node);
    }
    fn visit_expr_method_call(&mut self, node: &'ast syn::ExprMethodCall) {
        if [
            "getattr",
            "setattr",
            "hasattr",
            "delattr",
            "get_item",
            "set_item",
            "del_item",
            "call_method",
            "call_method0",
            "call_method1",
        ]
        .iter()
        .any(|name| node.method == name)
        {
            if let Some(key) = node.args.first()
                && (self.fixed(key) || !self.runtime_key(&node.method.to_string(), key))
            {
                self.errors.push(key.span().start().line);
            }
        } else if node.method == "import" && node.args.len() > 1 {
            // PyOnceLock's class lookup is also a fixed member access.
            self.key(node.args.last());
        }
        syn::visit::visit_expr_method_call(self, node);
    }
    fn visit_expr_path(&mut self, node: &'ast syn::ExprPath) {
        if let Some(name) = node.path.segments.last() {
            let field_helper = [
                "attr_with",
                "attr_or",
                "item_with",
                "item_or",
                "host_exception",
                "host_exception_with",
            ]
            .iter()
            .any(|helper| name.ident == helper)
                || (["attr", "item"].iter().any(|helper| name.ident == helper)
                    && node.path.segments.iter().any(|s| s.ident == "py_values"));
            let method_helper =
                name.ident == "read" && node.path.segments.iter().any(|s| s.ident == "py_calls");
            if (field_helper || method_helper) && !self.bridge() {
                self.errors.push(node.span().start().line);
            }
        }
        syn::visit::visit_expr_path(self, node);
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn fixed_members_and_helper_keys_are_rejected() {
        for expression in [
            r#"obj.getattr("field")"#,
            r##"obj.get_item(r#"field"#)"##,
            r#"obj.call_method0("close")"#,
            r#"obj.setattr("field", value)"#,
            r#"attr_with(obj, owner, "field", read)"#,
            r#"item_or(obj, owner, "field", read, default)"#,
            r#"cache.import(py, "module", "Class")"#,
            r#"{ const KEY: &str = "field"; obj.getattr(KEY) }"#,
            r#"{ let key = "field".to_owned(); obj.get_item(&key) }"#,
            r#"obj.getattr(concat!("fi", "eld"))"#,
            r#"macro_rules! access { ($obj:expr) => { $obj.getattr("field") }; }"#,
        ] {
            assert!(
                !super::violations(&format!("fn test() {{ {expression}; }}")).unwrap().is_empty(),
                "{expression}"
            );
        }
    }
    #[test]
    fn generated_converters_and_test_fixtures_are_allowed() {
        let source = r#"
            fn convert(obj: Object, key: Key) {
                obj.get_item(0);
                access::Audio::sample_rate(&obj);
                // obj.getattr("comment")
                let message = "obj.getattr(\"example\")";
            }
            #[cfg(test)] mod tests { fn fixture() { obj.getattr("fixture"); } }
        "#;
        assert!(super::violations(source).unwrap().is_empty());
    }
    #[test]
    fn computed_framework_keys_and_generic_helper_indirection_are_rejected() {
        for body in [
            r#"let field = format!("{}_conn_options", leg); obj.getattr(field);"#,
            "obj.getattr(key);",
            "let lookup = crate::py_values::attr_with; lookup(obj, owner, key, read);",
            "crate::py_values::item_or(obj, owner, key, read, default);",
            "crate::py_calls::read(obj, key);",
            "use crate::py_values::attr_with as read_field; read_field(obj, owner, key, read);",
        ] {
            assert!(
                !super::violations(&format!(
                    "fn business(obj: Object, key: Key, leg: Leg) {{ {body} }}"
                ))
                .unwrap()
                .is_empty(),
                "{body}"
            );
        }
    }
    #[test]
    fn only_value_maps_declared_bridges_and_reflection_have_runtime_keys() {
        for (file, source) in [
            (
                "py_values.rs",
                "fn value_to_py() { for (key, value) in map { out.set_item(key, value); } }",
            ),
            ("py_values.rs", "fn attr_with(obj: Object, key: Key) { obj.getattr(key); }"),
            (
                "py_errors.rs",
                "fn host_exception(obj: Object, class: Class) { obj.getattr(class); }",
            ),
            (
                "py_values.rs",
                "fn record_fields(obj: Object) { for name in fields { obj.getattr(name.cast::<PyString>()?); } }",
            ),
        ] {
            assert!(super::violations_at(source, file).unwrap().is_empty(), "{file}: {source}");
        }
        assert!(
            !super::violations_at(
                "fn attr_with(obj: Object, key: Key) { obj.getattr(format!(\"{key}\")); }",
                "py_values.rs"
            )
            .unwrap()
            .is_empty()
        );
    }

    #[test]
    fn handwritten_generated_reexport_inventories_are_rejected() {
        assert!(
            !super::violations(
                "pub use crate::generated::records::room_options::audio_input_config_from_py;"
            )
            .unwrap()
            .is_empty()
        );
        assert!(
            !super::violations("pub(crate) use crate::generated::records::*;").unwrap().is_empty()
        );
        assert!(super::violations("use crate::generated::records; fn convert(x: Object) { records::room_options::audio_input_config_from_py(x); }").unwrap().is_empty());
    }
}
