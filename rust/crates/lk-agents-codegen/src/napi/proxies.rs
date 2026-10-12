//! `#[host(proxy)]`: a core trait implemented over a JS object through `crate::bridge`, the other
//! direction from every other item.

use super::records::collect_deps;
use super::types::{n_of_syn, to_js};
use super::{linked, snake, Cx};
use crate::ir::*;

/// `#[host(proxy)]` over a JS object via `crate::bridge`. `soon` is fire-and-forget; `await`
/// awaits the Promise; `anext` pulls the async iterator; `once`/`keep` values are constructor
/// arguments, since a worker thread cannot read JS synchronously. Conversions follow Rust types,
/// not the pyo3 converters. `call`, `factory` and repeated sync reads are refused (they would
/// block a worker on the JS thread).
pub(super) fn proxy(cx: &Cx, p: &Proxy, wanted: &mut Vec<String>) -> Result<String, String> {
    linked(&p.trait_path)?;
    if p.host.is_some() {
        return Err("not yet: a bare proxy (`host = ..`: every method calls `_core_<method>`)".to_owned());
    }
    let name = &p.name;
    let fields: Vec<(String, String)> = p.state.to_vec();
    let mut bodies = Vec::new();
    for m in &p.methods {
        bodies.push(proxy_method(cx, m, wanted).map_err(|e| format!("`{}`: {e}", m.name))?);
    }
    let module = snake(name);
    let mut b = String::new();
    b.push_str(&format!("pub mod {module} {{\n    #![allow(unused_imports)]\n"));
    for u in &p.uses {
        b.push_str(&format!("    use {u};\n"));
    }
    b.push_str(&format!(
        "\n    /// `{}` over a JS object (`#[host(proxy)]` in `{}`).\n    pub struct {name} {{\n        pub(crate) bridge: crate::bridge::NodeBridge,\n",
        p.trait_path, p.source
    ));
    for (n, t) in &fields {
        b.push_str(&format!("        pub(crate) {n}: {t},\n"));
    }
    b.push_str("    }\n\n");
    let params: Vec<String> = std::iter::once("bridge: crate::bridge::NodeBridge".to_owned())
        .chain(fields.iter().map(|(n, t)| format!("{n}: {t}")))
        .collect();
    let all: Vec<String> = std::iter::once("bridge".to_owned()).chain(fields.iter().map(|(n, _)| n.clone())).collect();
    b.push_str(&format!(
        "    impl {name} {{\n        pub fn from_bridge({}) -> Self {{\n            {name} {{ {} }}\n        }}\n    }}\n\n",
        params.join(", "),
        all.join(", ")
    ));
    if p.async_trait {
        b.push_str("    #[async_trait::async_trait]\n");
    }
    b.push_str(&format!("    impl {} for {name} {{\n", p.trait_path));
    for (i, (m, body)) in p.methods.iter().zip(&bodies).enumerate() {
        if i > 0 {
            b.push('\n');
        }
        b.push_str(&format!("        {} {{\n", m.sig));
        for l in body.lines() {
            b.push_str(&format!("            {l}\n"));
        }
        b.push_str("        }\n");
    }
    b.push_str(&format!("    }}\n}}\npub use {module}::{name};\n"));
    Ok(b)
}

type SigTypes = (Vec<(String, syn::Type)>, Option<syn::Type>);

/// Read off the verbatim signature.
fn sig_types(sig: &str) -> Result<SigTypes, String> {
    let s: syn::Signature = syn::parse_str(sig).map_err(|e| format!("the signature does not parse: {e}"))?;
    let params = s
        .inputs
        .iter()
        .filter_map(|a| match a {
            syn::FnArg::Typed(t) => match &*t.pat {
                syn::Pat::Ident(i) => Some((i.ident.to_string(), (*t.ty).clone())),
                _ => None,
            },
            syn::FnArg::Receiver(_) => None,
        })
        .collect();
    let ret = match s.output {
        syn::ReturnType::Default => None,
        syn::ReturnType::Type(_, t) => Some(*t),
    };
    Ok((params, ret))
}


fn is_unit(t: &syn::Type) -> bool {
    matches!(t, syn::Type::Tuple(tt) if tt.elems.is_empty())
}


fn js_args(cx: &Cx, params: &[(String, syn::Type)], wanted: &mut Vec<String>) -> Result<String, String> {
    let mut out = Vec::new();
    for (n, t) in params {
        let nt = n_of_syn(cx, t, None).map_err(|e| format!("parameter `{n}`: {e}"))?;
        collect_deps(&nt, wanted);
        out.push(format!("crate::bridge::JsArg::new({})", to_js(cx, &nt, n, false).map_err(|e| format!("parameter `{n}`: {e}"))?));
    }
    Ok(format!("vec![{}]", out.join(", ")))
}


fn proxy_method(cx: &Cx, m: &ProxyMethod, wanted: &mut Vec<String>) -> Result<String, String> {
    let (params, ret) = sig_types(&m.sig)?;
    let err = "crate::bridge::host_err";
    let unit = ret.as_ref().is_none_or(is_unit);
    Ok(match &m.action {
        ProxyAction::Soon { method } => {
            let r = format!("self.bridge.soon({method:?}, {})", js_args(cx, &params, wanted)?);
            if unit { format!("let _ = {r};") } else { format!("{r}.map_err({err})") }
        }
        ProxyAction::Core => return Err("not yet: a bare proxy method".to_owned()),
    })
}
