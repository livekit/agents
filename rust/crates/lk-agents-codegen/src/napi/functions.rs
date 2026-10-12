//! Exported functions, and the call plumbing classes share: parameter conversion ([`P`]), error
//! mapping, the return, the `#[napi]` body; plus the `e_to_node` error converters.

use super::records::collect_deps;
use super::types::{exposed_object, from_js, n_of_rty, napi_ty, to_js, ts_override, N};
use super::{arg, doc_line, linked, push_unique, raw, snake, summary, Cx, ERROR_WITH, NATIVE_INTS};
use crate::ir::*;
use crate::node::camel;

pub(super) struct P {
    pub(super) decl: String,
    pub(super) prelude: Vec<String>,
    pub(super) call: String,
    fallible: bool,
}

/// Each parameter's napi declaration, conversion, and core call argument.
pub(super) fn params(cx: &Cx, sig: &Sig, owned_only: bool, wanted: &mut Vec<String>) -> Result<Vec<P>, String> {
    cx.in_params.set(true);
    let out = params_inner(cx, sig, owned_only, wanted);
    cx.in_params.set(false);
    out
}

fn params_inner(cx: &Cx, sig: &Sig, owned_only: bool, wanted: &mut Vec<String>) -> Result<Vec<P>, String> {
    let mut out = Vec::new();
    for p in &sig.params {
        let label = p.label.as_deref().unwrap_or(&p.name);
        let n = arg(&p.name);
        // `#[host(lenient)]`: a wrong-typed value reads as the default, as pyo3's
        // `extract().unwrap_or_default()`.
        if p.lenient {
            let nt = n_of_rty(cx, &p.ty).map_err(|e| format!("parameter `{label}`: {e}"))?;
            if !matches!(nt, N::Bool | N::Float(_) | N::Str) && !matches!(&nt, N::Int(w) if NATIVE_INTS.contains(&w.as_str()))
                && !matches!(&nt, N::Option(i) if matches!(**i, N::Bool | N::Float(_) | N::Str))
            {
                return Err(format!("parameter `{label}`: #[host(lenient)] reads a plain value (a number, a string, an Option of one)"));
            }
            let napi = napi_ty(cx, &nt)?;
            out.push(P {
                decl: format!("{n}: napi::bindgen_prelude::Unknown<'_>"),
                prelude: vec![format!("let {n}: {napi} = crate::host_error::lenient(&{n});")],
                call: n.clone(),
                fallible: false,
            });
            continue;
        }
        let default = sig.defaults.iter().find(|(k, _)| k == &p.name).map(|(_, v)| v.clone());
        if let RTy::Declared { name, by_ref } = &p.ty
            && let Some(o) = exposed_object(cx, name)
        {
            if owned_only {
                return Err(format!("parameter `{label}` is borrowed from JS, which an async method cannot hold across an await"));
            }
            let class = format!("crate::generated::objects::{}", o.name);
            let (prelude, call) = match (o.cell, by_ref) {
                (Cell::Mutex, true) => (
                    vec![format!("let {n} = {n}.inner.lock().unwrap_or_else(std::sync::PoisonError::into_inner);")],
                    format!("&{n}"),
                ),
                (_, true) => (Vec::new(), format!("&{n}.inner")),
                _ => return Err(format!("parameter `{label}`: an exposed `{name}` is lent, never moved; take `&{name}`")),
            };
            out.push(P { decl: format!("{n}: &{class}"), prelude, call, fallible: false });
            continue;
        }
        let (owned, lend): (RTy, &str) = match &p.ty {
            RTy::StrRef => (RTy::Str, "&"),
            RTy::Bytes => (RTy::Bytes, "bytes"),
            RTy::Slice(i) if **i == RTy::StrRef => (RTy::Vec(Box::new(RTy::Str)), "strs"),
            RTy::Slice(i) => (RTy::Vec(i.clone()), "&"),
            RTy::Option(i) if **i == RTy::StrRef => (RTy::Option(Box::new(RTy::Str)), "as_deref"),
            RTy::Option(i) if matches!(**i, RTy::Json { by_ref: true } | RTy::JsonMap { by_ref: true } | RTy::Declared { by_ref: true, .. }) => {
                (RTy::Option(Box::new(unref(i))), "as_ref")
            }
            t @ (RTy::Json { by_ref: true } | RTy::JsonMap { by_ref: true } | RTy::Declared { by_ref: true, .. }) => (unref(t), "&"),
            t => (t.clone(), ""),
        };
        if owned_only && !lend.is_empty() {
            return Err(format!("parameter `{label}` is borrowed, which an async method cannot hold across an await; take it owned"));
        }
        let nt = if lend == "bytes" {
            None
        } else {
            let nt = n_of_rty(cx, &owned).map_err(|e| format!("parameter `{label}`: {e}"))?;
            if nt.any(&|t| matches!(t, N::Ref(_) | N::Object(_))) {
                return Err(format!("parameter `{label}`: {:?} borrows inside a container", p.ty));
            }
            collect_deps(&nt, wanted);
            Some(nt)
        };
        let (napi, ts) = match &nt {
            None => ("napi::bindgen_prelude::Buffer".to_owned(), None),
            Some(nt) => (napi_ty(cx, nt)?, ts_override(cx, nt)?),
        };
        let conv = |v: &str| -> Result<String, String> {
            match &nt {
                None => Ok(v.to_owned()),
                Some(nt) => from_js(cx, nt, v, label).map_err(|e| format!("parameter `{label}`: {e}")),
            }
        };
        let is_option = matches!(owned, RTy::Option(_) | RTy::Maybe(_));
        let mut prelude = Vec::new();
        let napi = match &default {
            Some(d) if d == "None" && is_option => napi,
            None => napi,
            Some(d) if lend == "&" && owned == RTy::Str => {
                prelude.push(format!("let {n} = {n}.as_deref().unwrap_or({d});"));
                format!("Option<{napi}>")
            }
            Some(d) => {
                if !lend.is_empty() {
                    return Err(format!("not yet: a default on the borrowed parameter `{label}`"));
                }
                let c = conv("v")?;
                prelude.push(if c == "v" {
                    format!("let {n} = {n}.unwrap_or({d});")
                } else {
                    format!("let {n} = match {n} {{ Some(v) => {c}, None => {d} }};")
                });
                format!("Option<{napi}>")
            }
        };
        if prelude.is_empty() {
            let c = conv(&n)?;
            if c != n {
                prelude.push(format!("let {n} = {c};"));
            }
        }
        let decl = match ts {
            Some(t) => format!("#[napi(ts_arg_type = {t:?})] {n}: {napi}"),
            None => format!("{n}: {napi}"),
        };
        let call = match lend {
            "&" | "bytes" => format!("&{n}"),
            "as_deref" => format!("{n}.as_deref()"),
            "as_ref" => format!("{n}.as_ref()"),
            "strs" => {
                prelude.push(format!("let {n}: Vec<&str> = {n}.iter().map(String::as_str).collect();"));
                format!("&{n}")
            }
            _ => n.clone(),
        };
        let fallible = prelude.iter().any(|p| p.contains('?'));
        out.push(P { decl, prelude, call, fallible });
    }
    Ok(out)
}

fn unref(t: &RTy) -> RTy {
    match t {
        RTy::Json { .. } => RTy::Json { by_ref: false },
        RTy::JsonMap { .. } => RTy::JsonMap { by_ref: false },
        RTy::Declared { name, .. } => RTy::Declared { name: name.clone(), by_ref: false },
        other => other.clone(),
    }
}

/// `None` for a call that cannot fail.
pub(super) fn err_map(cx: &Cx, sig: &Sig) -> Result<Option<String>, String> {
    Ok(match &sig.err {
        None => None,
        Some(ErrMap::Kind(k)) => Some(format!("|e| crate::host_error::JsError::new({k:?}, e.to_string())")),
        Some(ErrMap::Auto) => match sig.err_ty.as_deref().and_then(|n| cx.x.errors.iter().find(|e| e.name == n)) {
            Some(e) if e.with.is_some() || ERROR_WITH.iter().any(|(n, _)| *n == e.name) => match ERROR_WITH.iter().find(|(n, _)| *n == e.name) {
                Some((_, f)) => Some((*f).to_owned()),
                None => return Err(format!("`{}` maps through a pyo3 converter with no Node counterpart", e.name)),
            },
            Some(e) => {
                push_unique(&mut cx.errors.borrow_mut(), &e.name);
                Some(format!("crate::generated::errors::{}_to_node", snake(&e.name)))
            }
            // `ValueError(str(e))`, pyo3's default.
            None => Some("|e| crate::host_error::JsError::new(\"ValueError\", e.to_string())".to_owned()),
        },
    })
}

pub(super) fn ret(cx: &Cx, t: &RTy, wanted: &mut Vec<String>) -> Result<(String, Option<String>, String), String> {
    if *t == RTy::Unit {
        return Ok(("()".to_owned(), None, "()".to_owned()));
    }
    let n = n_of_rty(cx, t).map_err(|e| format!("return: {e}"))?;
    collect_deps(&n, wanted);
    Ok((napi_ty(cx, &n)?, ts_override(cx, &n)?, to_js(cx, &n, "out", false).map_err(|e| format!("return: {e}"))?))
}

/// The signature line and body. `call` is the core call, `pre` binds what it names.
#[allow(clippy::too_many_arguments)]
pub(super) fn body(
    cx: &Cx,
    ident: &str,
    recv: Option<&str>,
    ps: &[P],
    pre: &[String],
    call: &str,
    sig: &Sig,
    ret_ty: &str,
    ret_expr: &str,
    pad: &str,
) -> Result<String, String> {
    let err = err_map(cx, sig)?;
    let fallible = err.is_some() || ps.iter().any(|p| p.fallible);
    let mut args: Vec<String> = recv.map(str::to_owned).into_iter().collect();
    if fallible {
        args.push("env: Env".to_owned());
    }
    args.extend(ps.iter().map(|p| p.decl.clone()));
    let ty = if fallible { format!("napi::Result<{ret_ty}>") } else { ret_ty.to_owned() };
    let arrow = if ty == "()" { String::new() } else { format!(" -> {ty}") };
    let mut b = format!("{pad}pub fn {ident}({}){arrow} {{\n", args.join(", "));
    let inner = if fallible { format!("{pad}        ") } else { format!("{pad}    ") };
    if fallible {
        b.push_str(&format!("{pad}    crate::host_error::run(&env, || {{\n"));
    }
    for p in ps.iter().flat_map(|p| &p.prelude).chain(pre) {
        b.push_str(&format!("{inner}{p}\n"));
    }
    let call = match &err {
        Some(m) => format!("{call}.map_err({m})?"),
        None => call.to_owned(),
    };
    if ret_ty == "()" {
        b.push_str(&format!("{inner}{call};\n"));
        if fallible {
            b.push_str(&format!("{inner}Ok(())\n"));
        }
    } else if ret_expr == "out" {
        let e = match (fallible, call.strip_suffix('?')) {
            (true, Some(c)) if !c.contains('?') => c.to_owned(),
            (true, _) => format!("Ok({call})"),
            (false, _) => call,
        };
        b.push_str(&format!("{inner}{e}\n"));
    } else if ret_expr.matches("out").count() == 1 {
        let e = ret_expr.replacen("out", &call, 1);
        b.push_str(&format!("{inner}{}\n", if fallible { format!("Ok({e})") } else { e }));
    } else {
        b.push_str(&format!("{inner}let out = {call};\n"));
        b.push_str(&format!("{inner}{}\n", if fallible { format!("Ok({ret_expr})") } else { ret_expr.to_owned() }));
    }
    if fallible {
        b.push_str(&format!("{pad}    }})\n"));
    }
    b.push_str(&format!("{pad}}}\n"));
    Ok(b)
}

pub(super) fn napi_attr(kind: &str, js: &str, ts_ret: Option<&str>, pad: &str) -> String {
    let mut parts: Vec<String> = Vec::new();
    if !kind.is_empty() {
        parts.push(kind.to_owned());
    }
    parts.push(format!("js_name = \"{js}\""));
    if let Some(t) = ts_ret {
        parts.push(format!("ts_return_type = {t:?}"));
    }
    format!("{pad}#[napi({})]\n", parts.join(", "))
}

pub(super) fn function(cx: &Cx, f: &ExportFn, wanted: &mut Vec<String>) -> Result<String, String> {
    linked(&f.rust_path)?;
    let mut mine = Vec::new();
    let ps = params(cx, &f.sig, false, &mut mine)?;
    let (ret_ty, ts_ret, ret_expr) = ret(cx, &f.sig.ret, &mut mine)?;
    let call = if f.is_const { f.rust_path.clone() } else { format!("{}({})", f.rust_path, ps.iter().map(|p| p.call.as_str()).collect::<Vec<_>>().join(", ")) };
    let mut b = String::new();
    for l in summary(&f.doc) {
        b.push_str(&doc_line(&l, ""));
    }
    b.push_str(&napi_attr("", &camel(&f.name), ts_ret.as_deref(), ""));
    b.push_str(&body(cx, &raw(&f.name), None, &ps, &[], &call, &f.sig, &ret_ty, &ret_expr, "")?);
    wanted.extend(mine);
    Ok(b)
}

/// `pub fn <e>_to_node(e: E) -> HostError`: the kind table `exports.rs` writes for pyo3.
pub(super) fn error_fn(e: &ErrorType) -> String {
    let name = snake(&e.name);
    let mut b = format!("/// `{}` as a host error (`#[host(error)]` in `{}`).\n", e.name, e.source);
    if e.variants.is_empty() {
        b.push_str(&format!("pub fn {name}_to_node(e: {}) -> JsError {{\n    JsError::new({:?}, e.to_string())\n}}\n", e.path, e.kind));
        return b;
    }
    b.push_str(&format!("#[allow(unreachable_patterns)]\npub fn {name}_to_node(e: {}) -> JsError {{\n    let kind = match &e {{\n", e.path));
    for (v, pat, k) in &e.variants {
        b.push_str(&format!("        {}::{v}{pat} => {k:?},\n", e.path));
    }
    b.push_str(&format!("        _ => {:?},\n    }};\n    JsError::new(kind, e.to_string())\n}}\n", e.kind));
    b
}
