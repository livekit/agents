//! [`N`], a Rust type as Node sees it, and the conversions both ways: the napi type, its
//! TypeScript, core -> JS and JS -> core expressions.

use super::records::{ident_path, rec, rec_for, RecKind};
use super::{key, Cx, Dir, HAND_TYPES, JSON_MAP_ALIASES, NATIVE_INTS};
use crate::ir::*;

#[derive(Clone, Debug, PartialEq)]
pub(super) enum N {
    Bool,
    Int(String),
    Float(String),
    Char,
    Str,
    Bytes,
    Option(Box<N>),
    Vec(Box<N>),
    Map(Box<N>),
    Tuple(Vec<N>),
    Json,
    JsonMap,
    Duration,
    SystemTime,
    Maybe(Box<N>),
    /// `true` for a `#[host(wire_enum)]` (has `FromStr`); a declared unit enum only has `as_str`,
    /// so it is read from JS only as a parameter.
    Enum(Vec<String>, bool),
    Record(String),
    Union(String),
    Object(String),
    /// A `()` field that carries a literal.
    Lit(String),
    Ref(Box<N>),
    /// Encoded bytes, decoded by the host with `@livekit/protocol`.
    Proto(String),
    Hand(usize),
    /// Return only: collected.
    Iter(Box<N>),
}

impl N {
    pub(super) fn any(&self, f: &dyn Fn(&N) -> bool) -> bool {
        f(self)
            || match self {
                N::Option(i) | N::Vec(i) | N::Map(i) | N::Maybe(i) | N::Ref(i) | N::Iter(i) => i.any(f),
                N::Tuple(items) => items.iter().any(|i| i.any(f)),
                _ => false,
            }
    }
}

fn prim(p: &str) -> Result<N, String> {
    Ok(match p {
        "bool" => N::Bool,
        "f32" | "f64" => N::Float(p.to_owned()),
        "char" => N::Char,
        "u8" | "i8" | "u16" | "i16" | "u32" | "i32" | "i64" | "u64" | "usize" | "isize" => N::Int(p.to_owned()),
        other => return Err(format!("`{other}` has no Node shape")),
    })
}

pub(super) fn n_of_rty(cx: &Cx, t: &RTy) -> Result<N, String> {
    Ok(match t {
        RTy::Unit => return Err("`()` is not a value".to_owned()),
        RTy::Prim(p) => prim(p)?,
        RTy::Str | RTy::IntoString => N::Str,
        RTy::StrRef => N::Ref(Box::new(N::Str)),
        RTy::Bytes => N::Ref(Box::new(N::Bytes)),
        RTy::Option(i) => N::Option(Box::new(n_of_rty(cx, i)?)),
        RTy::Vec(i) if **i == RTy::Prim("u8".to_owned()) => N::Bytes,
        RTy::Vec(i) | RTy::IntoIter(i) => N::Vec(Box::new(n_of_rty(cx, i)?)),
        RTy::Slice(i) => N::Ref(Box::new(N::Vec(Box::new(n_of_rty(cx, i)?)))),
        RTy::Iter(i) => N::Iter(Box::new(n_of_rty(cx, i)?)),
        RTy::Tuple(items) => N::Tuple(items.iter().map(|i| n_of_rty(cx, i)).collect::<Result<_, _>>()?),
        RTy::Json { by_ref } => wrap_ref(N::Json, *by_ref),
        RTy::JsonMap { by_ref } => wrap_ref(N::JsonMap, *by_ref),
        RTy::Duration => N::Duration,
        RTy::SystemTime => N::SystemTime,
        RTy::Map(i) => N::Map(Box::new(n_of_rty(cx, i)?)),
        RTy::Maybe(i) => N::Maybe(Box::new(n_of_rty(cx, i)?)),
        RTy::Declared { name, by_ref } => {
            if exposed_object(cx, name).is_some() {
                return Ok(wrap_ref(N::Object(name.clone()), *by_ref));
            }
            if cx.x.kwargs.iter().any(|k| &k.name == name) {
                return Err(format!("not yet: `{name}` is a keyword record (spread into keywords in Python; an options object in Node)"));
            }
            wrap_ref(named(cx, name)?, *by_ref)
        }
        RTy::SelfTy => return Err("`Self` outside a constructor".to_owned()),
        RTy::Dyn(tr) => return Err(format!("not yet: `Arc<dyn {tr}>` (a host object behind a proxy)")),
        RTy::Shared(name) => return Err(format!("not yet: `Arc<{name}>` (a shared handle to an exposed class)")),
        RTy::HostRef => return Err("not yet: a `HostRef` (a host object in the side table)".to_owned()),
        RTy::StreamHalf { .. } => return Err("not yet: a stream half returned as a stream class".to_owned()),
    })
}

fn wrap_ref(n: N, by_ref: bool) -> N {
    if by_ref { N::Ref(Box::new(n)) } else { n }
}

fn named(cx: &Cx, name: &str) -> Result<N, String> {
    if let Some(w) = cx.x.wire_enums.iter().find(|w| w.name == name) {
        if w.open {
            return Err(format!("not yet: `{name}` is an open string (#[host(string)])"));
        }
        return Ok(N::Enum(w.values.clone(), true));
    }
    let d = cx
        .decls
        .iter()
        .find(|d| d.rust_name == name)
        .ok_or_else(|| format!("`{name}` is not a #[derive(Host)] type or a #[host(wire_enum)]"))?;
    Ok(match &d.body {
        Body::Unit(vs) => N::Enum(vs.iter().map(|v| v.value.clone()).collect(), false),
        Body::Union(_) => N::Union(key(d)),
        Body::Struct(_) => N::Record(key(d)),
    })
}

pub(super) fn exposed_object<'a>(cx: &Cx<'a>, name: &str) -> Option<&'a Object> {
    cx.x.objects.iter().find(|o| o.core_path.rsplit("::").next() == Some(name))
}

/// A field's Rust spelling, guided by [`Ty`] where the spelling is an alias (`Extra`, `Object`).
pub(super) fn n_of_field(cx: &Cx, owner: &str, f: &Field) -> Result<N, String> {
    if let Some(lit) = &f.literal {
        return Ok(N::Lit(lit.clone()));
    }
    let t: syn::Type =
        syn::parse_str(&f.rust_ty).map_err(|e| format!("{owner}.{}: `{}` does not parse: {e}", f.rust_name, f.rust_ty))?;
    n_of_syn(cx, &t, Some(&f.ty)).map_err(|e| format!("field `{}`: {e}", f.rust_name))
}

fn guide(t: Option<&Ty>) -> Option<&Ty> {
    match t {
        Some(Ty::Foreign(f)) => f.rust.as_deref(),
        other => other,
    }
}

pub(super) fn n_of_syn(cx: &Cx, t: &syn::Type, ty: Option<&Ty>) -> Result<N, String> {
    let ty = guide(ty);
    let spelled = || quote::quote!(#t).to_string().replace(' ', "");
    match t {
        syn::Type::Tuple(tt) if !tt.elems.is_empty() => {
            let items = match ty {
                Some(Ty::Tuple(g)) => tt.elems.iter().zip(g.iter().map(Some).chain(std::iter::repeat(None))).map(|(e, g)| n_of_syn(cx, e, g)).collect(),
                _ => tt.elems.iter().map(|e| n_of_syn(cx, e, None)).collect::<Result<Vec<_>, _>>(),
            };
            Ok(N::Tuple(items?))
        }
        syn::Type::Reference(r) => Ok(N::Ref(Box::new(n_of_syn(cx, &r.elem, ty)?))),
        syn::Type::Path(p) if p.path.segments.first().is_some_and(|s| s.ident == "proto" || s.ident == "livekit_protocol") => {
            Ok(N::Proto(spelled()))
        }
        syn::Type::Path(p) if p.path.segments.last().is_some_and(|s| HAND_TYPES.iter().any(|h| s.ident == h.0)) => {
            let last = p.path.segments.last().expect("checked").ident.to_string();
            Ok(N::Hand(HAND_TYPES.iter().position(|h| h.0 == last).expect("checked")))
        }
        syn::Type::Path(p) => {
            let seg = p.path.segments.last().ok_or("an empty path")?;
            let args: Vec<&syn::Type> = match &seg.arguments {
                syn::PathArguments::AngleBracketed(a) => a
                    .args
                    .iter()
                    .filter_map(|a| if let syn::GenericArgument::Type(t) = a { Some(t) } else { None })
                    .collect(),
                _ => Vec::new(),
            };
            let inner = |g: fn(&Ty) -> Option<&Ty>| ty.and_then(g);
            let name = seg.ident.to_string();
            Ok(match (name.as_str(), args.as_slice()) {
                ("String" | "str", []) => N::Str,
                ("Option", [a]) => N::Option(Box::new(n_of_syn(cx, a, inner(|t| if let Ty::Option(i) = t { Some(i) } else { None }))?)),
                ("Vec", [a]) if quote::quote!(#a).to_string() == "u8" => N::Bytes,
                ("Vec", [a]) => N::Vec(Box::new(n_of_syn(cx, a, inner(|t| if let Ty::List(i) = t { Some(i) } else { None }))?)),
                ("HashMap" | "BTreeMap" | "IndexMap", [k, v]) => {
                    if quote::quote!(#k).to_string() != "String" {
                        return Err(format!("`{}`: a map key that is not a String", spelled()));
                    }
                    N::Map(Box::new(n_of_syn(cx, v, inner(|t| if let Ty::Map(_, v) = t { Some(v) } else { None }))?))
                }
                ("Map", [_, _]) => N::JsonMap,
                ("Value", []) => N::Json,
                ("Duration", []) => N::Duration,
                ("SystemTime", []) => N::SystemTime,
                ("Maybe", [a]) => N::Maybe(Box::new(n_of_syn(cx, a, None)?)),
                (p, []) if prim(p).is_ok() => prim(p)?,
                (other, []) => match (named(cx, other), ty) {
                    (Ok(n), _) => n,
                    (Err(_), Some(Ty::Any)) => N::Json,
                    (Err(_), _) if JSON_MAP_ALIASES.contains(&other) => N::JsonMap,
                    (Err(_), Some(Ty::Map(k, v))) if matches!(**k, Ty::Str) && matches!(**v, Ty::Any) => N::JsonMap,
                    (Err(_), _) => return Err(format!("`{}` has no Node shape (not a declared record, enum or JSON)", spelled())),
                },
                _ => return Err(format!("`{}` has no Node shape", spelled())),
            })
        }
        _ => Err(format!("`{}` has no Node shape", spelled())),
    }
}


pub(super) fn napi_ty(cx: &Cx, n: &N) -> Result<String, String> {
    Ok(match n {
        N::Bool => "bool".to_owned(),
        N::Int(w) if NATIVE_INTS.contains(&w.as_str()) => w.clone(),
        N::Int(_) => "i64".to_owned(),
        N::Float(_) | N::Duration | N::SystemTime => "f64".to_owned(),
        N::Char | N::Str | N::Enum(..) | N::Lit(_) => "String".to_owned(),
        N::Bytes => "napi::bindgen_prelude::Buffer".to_owned(),
        N::Maybe(i) if matches!(**i, N::Option(_)) => {
            let N::Option(inner) = &**i else { unreachable!() };
            format!("crate::host_error::MaybeNull<{}>", napi_ty(cx, inner)?)
        }
        N::Option(i) | N::Maybe(i) => format!("Option<{}>", napi_ty(cx, i)?),
        N::Vec(i) | N::Iter(i) => format!("Vec<{}>", napi_ty(cx, i)?),
        N::Map(i) => format!("std::collections::HashMap<String, {}>", napi_ty(cx, i)?),
        N::Tuple(items) => {
            let parts = items.iter().map(|i| napi_ty(cx, i)).collect::<Result<Vec<_>, _>>()?;
            if parts.len() == 1 { format!("({},)", parts[0]) } else { format!("({})", parts.join(", ")) }
        }
        N::Json => "serde_json::Value".to_owned(),
        N::JsonMap => "serde_json::Map<String, serde_json::Value>".to_owned(),
        N::Record(r) | N::Union(r) => format!("crate::generated::records::{}", rec(cx, r)?.ident),
        N::Object(o) => format!("&crate::generated::objects::{}", exposed_object(cx, o).expect("resolved").name),
        N::Ref(i) => napi_ty(cx, i)?,
        N::Proto(_) => "napi::bindgen_prelude::Buffer".to_owned(),
        N::Hand(i) => HAND_TYPES[*i].1.to_owned(),
    })
}

/// `None` where napi's own rendering of [`napi_ty`] is already right.
pub(super) fn ts_override(cx: &Cx, n: &N) -> Result<Option<String>, String> {
    let special = |n: &N| {
        matches!(n, N::Enum(..) | N::Union(_) | N::Lit(_))
            || matches!(n, N::Hand(i) if HAND_TYPES[*i].2.contains('\''))
            || matches!(n, N::Maybe(i) if matches!(**i, N::Option(_)))
    };
    if !n.any(&special) {
        return Ok(None);
    }
    ts(cx, n).map(Some)
}

pub(super) fn ts(cx: &Cx, n: &N) -> Result<String, String> {
    let paren = |s: String| if s.contains(' ') { format!("({s})") } else { s };
    Ok(match n {
        N::Bool => "boolean".to_owned(),
        N::Int(_) | N::Float(_) | N::Duration | N::SystemTime => "number".to_owned(),
        N::Char | N::Str => "string".to_owned(),
        N::Bytes => "Buffer".to_owned(),
        N::Maybe(i) if matches!(**i, N::Option(_)) => format!("{} | undefined", ts(cx, i)?),
        N::Option(i) | N::Maybe(i) => format!("{} | null", ts(cx, i)?),
        N::Vec(i) | N::Iter(i) => format!("Array<{}>", ts(cx, i)?),
        N::Map(i) => format!("Record<string, {}>", ts(cx, i)?),
        N::Tuple(items) => format!("[{}]", items.iter().map(|i| ts(cx, i)).collect::<Result<Vec<_>, _>>()?.join(", ")),
        N::Json => "any".to_owned(),
        N::JsonMap => "Record<string, any>".to_owned(),
        N::Enum(vs, _) => vs.iter().map(|v| format!("'{v}'")).collect::<Vec<_>>().join(" | "),
        N::Lit(v) => format!("'{v}'"),
        N::Record(r) => rec(cx, r)?.ident.clone(),
        N::Union(r) => match &rec(cx, r)?.kind {
            RecKind::Union(arms) => arms.iter().map(|a| rec(cx, &a.member).map(|m| m.ident.clone())).collect::<Result<Vec<_>, _>>()?.join(" | "),
            RecKind::Struct { .. } => unreachable!("a union plan"),
        },
        N::Object(o) => exposed_object(cx, o).expect("resolved").name.clone(),
        N::Ref(i) => paren(ts(cx, i)?),
        N::Proto(_) => "Buffer".to_owned(),
        N::Hand(i) => HAND_TYPES[*i].2.to_owned(),
    })
}

fn recv(x: &str) -> String {
    x.strip_prefix('&').map(str::to_owned).unwrap_or_else(|| x.to_owned())
}

fn val(x: &str, by_ref: bool) -> String {
    match (by_ref, x.strip_prefix('&')) {
        (false, _) => x.to_owned(),
        (true, Some(place)) => place.to_owned(),
        (true, None) => format!("*{x}"),
    }
}

/// Infallible. `by_ref`: `x` is a `&T`.
pub(super) fn to_js(cx: &Cx, n: &N, x: &str, by_ref: bool) -> Result<String, String> {
    Ok(match n {
        N::Bool => val(x, by_ref),
        N::Int(w) if NATIVE_INTS.contains(&w.as_str()) => val(x, by_ref),
        // Saturating: a count past `i64::MAX` is not a negative count.
        N::Int(_) => format!("i64::try_from({}).unwrap_or(i64::MAX)", val(x, by_ref)),
        N::Float(w) if w == "f32" => format!("f64::from({})", val(x, by_ref)),
        N::Float(_) => val(x, by_ref),
        N::Char => format!("{}.to_string()", recv(x)),
        N::Str if by_ref => format!("{}.to_string()", recv(x)),
        N::Json | N::JsonMap if by_ref => format!("{}.clone()", recv(x)),
        N::Str | N::Json | N::JsonMap => x.to_owned(),
        N::Bytes if by_ref => format!("napi::bindgen_prelude::Buffer::from({}.to_vec())", recv(x)),
        N::Bytes => format!("napi::bindgen_prelude::Buffer::from({x})"),
        N::Duration => format!("{}.as_secs_f64()", recv(x)),
        N::SystemTime => format!("lk_agents_abi::epoch_secs({})", val(x, by_ref)),
        N::Enum(..) => format!("{}.as_str().to_owned()", recv(x)),
        N::Lit(v) => format!("{v:?}.to_owned()"),
        N::Record(r) | N::Union(r) => {
            let r = rec_for(cx, r, Dir::ToJs)?;
            if by_ref { format!("{}::from({x})", ident_path(&r)) } else { format!("{}::from(&{x})", ident_path(&r)) }
        }
        N::Option(i) => {
            let e = to_js(cx, i, "v", by_ref)?;
            let src = if by_ref { format!("{}.as_ref()", recv(x)) } else { x.to_owned() };
            match () {
                _ if e == "v" && !by_ref => x.to_owned(),
                _ if e == "*v" => val(x, true),
                _ => format!("{src}.map(|v| {e})"),
            }
        }
        N::Vec(i) => {
            let e = to_js(cx, i, "v", by_ref)?;
            match () {
                _ if e == "v" && !by_ref => x.to_owned(),
                _ if by_ref => format!("{}.iter().map(|v| {e}).collect::<Vec<_>>()", recv(x)),
                _ => format!("{x}.into_iter().map(|v| {e}).collect::<Vec<_>>()"),
            }
        }
        N::Iter(i) => {
            let e = to_js(cx, i, "v", false)?;
            if e == "v" { format!("{x}.collect::<Vec<_>>()") } else { format!("{x}.map(|v| {e}).collect::<Vec<_>>()") }
        }
        N::Map(i) => {
            let e = to_js(cx, i, "v", by_ref)?;
            if by_ref {
                format!("{}.iter().map(|(k, v)| (k.clone(), {e})).collect::<std::collections::HashMap<_, _>>()", recv(x))
            } else {
                format!("{x}.into_iter().map(|(k, v)| (k, {e})).collect::<std::collections::HashMap<_, _>>()")
            }
        }
        N::Tuple(items) => {
            let names: Vec<String> = (0..items.len()).map(|i| format!("t{i}")).collect();
            let parts = items.iter().zip(&names).map(|(t, n)| to_js(cx, t, n, by_ref)).collect::<Result<Vec<_>, _>>()?;
            if !by_ref && parts.iter().zip(&names).all(|(p, n)| p == n) {
                x.to_owned()
            } else {
                let tuple = if parts.len() == 1 { format!("({},)", parts[0]) } else { format!("({})", parts.join(", ")) };
                format!("{{ let ({},) = {x}; {tuple} }}", names.join(", "))
            }
        }
        N::Ref(i) => to_js(cx, i, x, true)?,
        N::Proto(_) => format!(
            "napi::bindgen_prelude::Buffer::from(prost::Message::encode_to_vec({}))",
            if by_ref { x.to_owned() } else { format!("&{x}") }
        ),
        N::Hand(i) => HAND_TYPES[*i].3.replace("{x}", &if by_ref { x.to_owned() } else { format!("&{x}") }),
        N::Maybe(_) => return Err("not yet: a `Maybe` return".to_owned()),
        N::Object(_) => return Err("not yet: returning an exposed class".to_owned()),
    })
}

/// With `?` on a `HostError` where it can fail; `label` names the argument, as pyo3 does.
pub(super) fn from_js(cx: &Cx, n: &N, v: &str, label: &str) -> Result<String, String> {
    Ok(match n {
        N::Bool | N::Str | N::Json | N::JsonMap => v.to_owned(),
        N::Int(w) if NATIVE_INTS.contains(&w.as_str()) => v.to_owned(),
        // `OverflowError`, as pyo3.
        N::Int(w) => format!(
            "<{w}>::try_from({v}).map_err(|e| crate::host_error::JsError::new(\"OverflowError\", format!(\"{label}: {{e}}\")))?"
        ),
        N::Float(w) if w == "f32" => format!("{v} as f32"),
        N::Float(_) => v.to_owned(),
        N::Char => return Err("not yet: a `char` argument".to_owned()),
        N::Bytes => format!("{v}.to_vec()"),
        N::Duration => format!(
            "std::time::Duration::try_from_secs_f64({v}).map_err(|e| crate::host_error::JsError::new(\"ValueError\", format!(\"{label}: {{e}}\")))?"
        ),
        N::SystemTime => format!(
            "lk_agents_abi::wall_from_secs({v}).ok_or_else(|| crate::host_error::JsError::new(\"ValueError\", format!(\"{label}={{}} is not a usable wall clock\", {v})))?"
        ),
        N::Enum(_, wire) if !*wire && !cx.in_params.get() => {
            return Err("a declared enum is read from JS only as a parameter (it has no `FromStr` to rely on)".to_owned());
        }
        N::Enum(..) => format!(
            "{v}.parse().map_err(|e| crate::host_error::JsError::new(\"ValueError\", format!(\"{label}: {{e}}\")))?"
        ),
        N::Lit(_) => "()".to_owned(),
        N::Record(r) | N::Union(r) => {
            rec_for(cx, r, Dir::FromJs)?;
            format!("{v}.try_into()?")
        }
        N::Option(i) => {
            let e = from_js(cx, i, "v", label)?;
            match () {
                _ if e == "v" => v.to_owned(),
                _ if e.contains('?') => format!("{v}.map(|v| Ok::<_, crate::host_error::JsError>({e})).transpose()?"),
                _ => format!("{v}.map(|v| {e})"),
            }
        }
        N::Vec(i) => {
            let e = from_js(cx, i, "v", label)?;
            match () {
                _ if e == "v" => v.to_owned(),
                _ if e.contains('?') => format!(
                    "{v}.into_iter().map(|v| Ok::<_, crate::host_error::JsError>({e})).collect::<Result<Vec<_>, _>>()?"
                ),
                _ => format!("{v}.into_iter().map(|v| {e}).collect::<Vec<_>>()"),
            }
        }
        N::Map(i) => {
            let e = from_js(cx, i, "v", label)?;
            if e.contains('?') {
                format!("{v}.into_iter().map(|(k, v)| Ok::<_, crate::host_error::JsError>((k, {e}))).collect::<Result<_, _>>()?")
            } else {
                format!("{v}.into_iter().map(|(k, v)| (k, {e})).collect()")
            }
        }
        N::Tuple(items) => {
            let names: Vec<String> = (0..items.len()).map(|i| format!("t{i}")).collect();
            let parts = items.iter().zip(&names).map(|(t, n)| from_js(cx, t, n, label)).collect::<Result<Vec<_>, _>>()?;
            if parts.iter().zip(&names).all(|(p, n)| p == n) {
                v.to_owned()
            } else {
                let tuple = if parts.len() == 1 { format!("({},)", parts[0]) } else { format!("({})", parts.join(", ")) };
                format!("{{ let ({},) = {v}; {tuple} }}", names.join(", "))
            }
        }
        // An omitted (or null) argument is `NotGiven`, as pyo3's `Maybe` folds `None`.
        N::Maybe(i) => {
            // `Maybe<Option<T>>`: `undefined`, `null` and a value stay apart via `MaybeNull`.
            if let N::Option(inner) = &**i {
                let e = from_js(cx, inner, "v", label)?;
                return Ok(if e == "v" {
                    format!("{v}.into_core()")
                } else {
                    format!("{v}.try_map(|v| Ok::<_, crate::host_error::JsError>({e}))?.into_core()")
                });
            }
            let e = from_js(cx, i, "v", label)?;
            format!("match {v} {{ Some(v) => lk_agents_abi::Maybe::Given({e}), None => lk_agents_abi::Maybe::NotGiven }}")
        }
        N::Ref(i) => from_js(cx, i, v, label)?,
        N::Iter(_) => return Err("an iterator is a return type".to_owned()),
        N::Proto(path) => format!(
            "<{path} as prost::Message>::decode(&{v}[..]).map_err(|e| crate::host_error::JsError::new(\"ValueError\", format!(\"{label}: {{e}}\")))?"
        ),
        N::Hand(i) => HAND_TYPES[*i].4.replace("{v}", v),
        N::Object(_) => return Err("an exposed class is lent, not converted".to_owned()),
    })
}
