//! `#[python(provider(..))]` on a `#[host(proxy)]` trait: the Python base class, method by method
//! (see [`Role`]). Every method of the trait must say what it is to Python, so a method added to
//! the core is a codegen error until the base class answers it.

use super::Res;
use super::meta::{Meta, meta_str, metas};
use crate::ir::*;
use lk_agents_bindgen::doc;
use syn::spanned::Spanned as _;

fn python(attrs: &[syn::Attribute]) -> Vec<Meta> {
    metas(attrs)
        .into_iter()
        .filter_map(|m| Some(Meta { key: m.key.strip_prefix("python.")?.to_owned(), ..m }))
        .collect()
}

fn has(m: &[Meta], key: &str) -> bool {
    m.iter().any(|x| x.key == key)
}

fn only(at: &str, m: &[Meta], known: &[&str]) -> Res<()> {
    for x in m {
        let ok =
            known.iter().any(|k| if k.ends_with('.') { x.key.starts_with(k) } else { x.key == *k });
        if !ok {
            return Err(format!(
                "{at}: python({}) is not a key here ({})",
                x.key,
                known.join(", ")
            ));
        }
    }
    Ok(())
}

/// `None` for a trait without a Python face.
pub fn provider(
    label: &str,
    tr: &syn::ItemTrait,
    host: Option<&str>,
    text: &str,
) -> Res<Option<Provider>> {
    let at = format!("{label}: trait {}", tr.ident);
    let m = python(&tr.attrs);
    if m.is_empty() {
        return Ok(None);
    }
    only(&at, &m, &["provider", "provider."])?;
    let class = match (meta_str(&m, "provider.name"), host) {
        (Some(n), _) => n,
        (None, Some(h)) => h.rsplit('.').next().unwrap_or(h).to_owned(),
        (None, None) => {
            return Err(format!("{at}: provider(name = \"..\") on a proxy without host = \"..\""));
        }
    };
    let module = meta_str(&m, "provider.module")
        .ok_or_else(|| format!("{at}: provider(module = \"..\") is its _generated module"))?;
    let path = meta_str(&m, "provider.path")
        .ok_or_else(|| format!("{at}: provider(path = \"..\") is its public module"))?;
    let events = meta_str(&m, "provider.events")
        .map(|e| e.split(',').map(|s| s.trim().to_owned()).filter(|s| !s.is_empty()).collect())
        .unwrap_or_default();
    let mut methods = Vec::new();
    for item in &tr.items {
        let syn::TraitItem::Fn(f) = item else { continue };
        methods.push(method(&at, f, text)?);
    }
    Ok(Some(Provider {
        source: label.to_owned(),
        class,
        module,
        path,
        doc: doc(&tr.attrs),
        events,
        error: match (meta_str(&m, "provider.error.name"), meta_str(&m, "provider.error.kind")) {
            (Some(name), Some(kind)) => Some((name, kind)),
            (None, None) => None,
            _ => return Err(format!("{at}: provider(error(name = \"..\", kind = \"..\")))")),
        },
        methods,
    }))
}

const ROLES: &[&str] =
    &["label", "init", "identity", "hook", "conn", "markup", "chat", "value", "report"];

fn method(at: &str, f: &syn::TraitItemFn, text: &str) -> Res<ProviderMethod> {
    let rust = f.sig.ident.to_string();
    let at = format!("{at}::{rust}");
    let m = python(&f.attrs);
    only(
        &at,
        &m,
        &[
            "label",
            "init",
            "identity",
            "hook",
            "impl",
            "retry",
            "unsupported",
            "store",
            "default",
            "conn",
            "markup",
            "chat",
            "value",
            "report",
            "when",
            "stream.",
            "params",
            "call",
            "import",
        ],
    )?;
    let roles: Vec<&str> = ROLES
        .iter()
        .copied()
        .filter(|r| m.iter().any(|x| x.key.split('.').next() == Some(r)))
        .collect();
    let role_word = match roles.as_slice() {
        [one] => *one,
        [] => {
            return Err(format!(
                "{at}: what is this method to the Python base class? #[python({})]",
                ROLES.join(" | ")
            ));
        }
        many => return Err(format!("{at}: one role, not {}", many.join(" and "))),
    };
    let role = match role_word {
        "label" => Role::Label,
        "init" => Role::Init,
        "identity" => Role::Identity,
        "hook" => Role::Hook {
            name: meta_str(&m, "hook").unwrap_or_else(|| rust.clone()),
            implementation: meta_str(&m, "impl"),
            retry: meta_str(&m, "retry"),
            unsupported: meta_str(&m, "unsupported"),
            store: match (meta_str(&m, "store"), meta_str(&m, "default")) {
                (Some(attr), Some(default)) => Some((attr, default)),
                (None, None) => None,
                _ => {
                    return Err(format!(
                        "{at}: store = \"_attr\" and default = \"..\" go together"
                    ));
                }
            },
        },
        "conn" => Role::Conn,
        "markup" => Role::Markup,
        "chat" => Role::Chat { name: meta_str(&m, "chat").unwrap_or_else(|| rust.clone()) },
        "report" => Role::Report,
        "value" => {
            Role::Value(meta_str(&m, "value").ok_or_else(|| format!("{at}: value = \"expr\""))?)
        }
        _ => unreachable!("one of ROLES"),
    };
    let mut params = Vec::new();
    for input in &f.sig.inputs {
        let syn::FnArg::Typed(t) = input else { continue };
        let syn::Pat::Ident(id) = &*t.pat else {
            return Err(format!("{at}: a destructuring parameter has no Python name"));
        };
        let pm = python(&t.attrs);
        only(&at, &pm, &["ty", "attr", "import", "default"])?;
        let name = id.ident.to_string();
        params.push(ProviderParam {
            name: name.strip_prefix('_').unwrap_or(&name).to_owned(),
            rust_ty: span_text(text, t.ty.span()),
            py_ty: meta_str(&pm, "ty"),
            import: meta_str(&pm, "import"),
            default: meta_str(&pm, "default"),
            attr: has(&pm, "attr"),
        });
    }
    // An awaited call is an `async fn`, or a `fn` returning a boxed future of its output.
    let boxed = crate::decl::resolve::boxed_future(&f.sig.output);
    let ret = match (&boxed, &f.sig.output) {
        (Some((_, output)), _) => Some(span_text(text, output.span())),
        (None, syn::ReturnType::Default) => None,
        (None, syn::ReturnType::Type(_, t)) => Some(span_text(text, t.span())),
    };
    let stream = match meta_str(&m, "stream.name") {
        None => None,
        Some(name) => Some(StreamClass {
            name,
            kind: meta_str(&m, "stream.kind")
                .ok_or_else(|| format!("{at}: stream(kind = \"..\")"))?,
            alias: meta_str(&m, "stream.alias"),
            proxy: meta_str(&m, "stream.proxy"),
            doc: meta_str(&m, "stream.doc")
                .map(|d| d.lines().map(str::to_owned).collect())
                .unwrap_or_default(),
        }),
    };
    Ok(ProviderMethod {
        rust,
        doc: doc(&f.attrs),
        role,
        params,
        ret,
        is_async: f.sig.asyncness.is_some() || boxed.is_some(),
        has_default: f.default.is_some(),
        when: meta_str(&m, "when"),
        stream,
    })
}

fn span_text(text: &str, span: proc_macro2::Span) -> String {
    let (start, end) = (span.start(), span.end());
    let lines: Vec<&str> = text.lines().collect();
    let mut out = String::new();
    for l in start.line..=end.line {
        let line: Vec<char> = lines.get(l - 1).copied().unwrap_or("").chars().collect();
        let a = if l == start.line { start.column } else { 0 };
        let b = if l == end.line { end.column.min(line.len()) } else { line.len() };
        out.push_str(&line[a.min(b)..b].iter().collect::<String>());
        out.push(' ');
    }
    let mut collapsed = out.split_whitespace().collect::<Vec<_>>().join(" ");
    for (from, to) in [("( ", "("), (" )", ")"), (", )", ")"), (",)", ")")] {
        collapsed = collapsed.replace(from, to);
    }
    collapsed
}
