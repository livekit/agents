//! A fn signature and its Rust types as the bindings see them ([`Sig`], [`RTy`]); a type with no
//! host shape is refused by name.

use super::exposed::Aliases;
use super::meta::{host_attr_strict, meta_rust, meta_str, Meta};
use super::{quote, Res};
use crate::ir::*;

/// A fn signature under the `head` group's keys, plus per-parameter `#[host(..)]`.
pub(super) fn signature(at: &str, m: &[Meta], head: &str, s: &syn::Signature, al: &Aliases) -> Res<(Receiver, Sig)> {
    if !s.generics.params.is_empty() {
        return Err(format!("{at}: a generic fn has no single host signature; wrap it in a concrete one"));
    }
    let mut receiver = Receiver::None;
    let mut params = Vec::new();
    for input in &s.inputs {
        match input {
            // `self: &Arc<Self>`: the class holds the `Arc`, so it borrows like `&self`.
            syn::FnArg::Receiver(r) if r.reference.is_none() && quote(&r.ty).replace(' ', "") == "&Arc<Self>" => {
                receiver = Receiver::Ref;
            }
            syn::FnArg::Receiver(r) => {
                if r.reference.is_none() {
                    return Err(format!("{at}: a `self`-consuming method cannot be called on a shared host object"));
                }
                receiver = if r.mutability.is_some() { Receiver::Mut } else { Receiver::Ref };
            }
            syn::FnArg::Typed(t) => {
                let syn::Pat::Ident(id) = &*t.pat else {
                    return Err(format!("{at}: a destructuring parameter has no host name"));
                };
                // The host-side name: `r#ref` is `ref` to Python and JS. Each emitter re-escapes
                // it where it writes a Rust identifier (`napi::arg`), never a host one.
                let name = syn::ext::IdentExt::unraw(&id.ident).to_string();
                let pm = host_attr_strict(at, &t.attrs)?.unwrap_or_default();
                if let Some(bad) = pm.iter().find(|x| !lk_agents_bindgen::PARAM_MARKERS.contains(&x.key.as_str())) {
                    return Err(format!("{at}({name}): #[host({})] -- a parameter takes {}", bad.key, lk_agents_bindgen::PARAM_MARKERS.join(", ")));
                }
                let required = pm.iter().any(|x| x.key == "required");
                let ambient = pm.iter().any(|x| x.key == "ambient").then(|| quote(&t.ty).replace(' ', ""));
                let label = meta_str(&pm, "label");
                let lenient = pm.iter().any(|x| x.key == "lenient");
                let ty = match rty(&format!("{at}({name})"), &t.ty, al) {
                    Err(_) if ambient.is_some() => {
                        RTy::Declared { name: quote(&t.ty).replace(' ', ""), by_ref: false }
                    }
                    ty => ty?,
                };
                params.push((Param { ty, name, lenient, label, ambient }, required));
            }
        }
    }
    let ret_ty = |label: String, t: &syn::Type| -> Res<RTy> { rty(&label, t, al) };
    let (ret, is_result, err_ty) = match &s.output {
        syn::ReturnType::Default => (RTy::Unit, false, None),
        syn::ReturnType::Type(_, t) => match result_ok(t) {
            Some((ok, e)) => (ret_ty(format!("{at} -> Ok"), ok)?, true, e),
            None => (ret_ty(format!("{at} ->"), t)?, false, None),
        },
    };
    let err = match meta_str(m, &format!("{head}.err")) {
        Some(_) if !is_result => return Err(format!("{at}: `err` on a fn that does not return a Result")),
        Some(e) => Some(ErrMap::Kind(e)),
        None if is_result => Some(ErrMap::Auto),
        None => None,
    };
    let prefix = format!("{head}.defaults.");
    let mut defaults = Vec::new();
    for x in m {
        let Some(p) = x.key.strip_prefix(&prefix) else { continue };
        if !params.iter().any(|(q, _): &(Param, bool)| q.name == p) {
            return Err(format!("{at}: defaults({p} = ..) names no parameter"));
        }
        defaults.push((p.to_owned(), x.rust().unwrap_or_default()));
    }
    // An `Option<T>` whose followers all default is `None` unwritten: the expansion's rule.
    let host: Vec<lk_agents_bindgen::Shape> = params
        .iter()
        .filter(|(p, _)| p.ambient.is_none())
        .map(|(p, required)| lk_agents_bindgen::Shape {
            name: &p.name,
            option: matches!(p.ty, RTy::Option(_)),
            maybe: matches!(p.ty, RTy::Maybe(_)),
            required: *required,
            written: defaults.iter().any(|(d, _)| *d == p.name),
        })
        .collect();
    let implied: Vec<String> = lk_agents_bindgen::implied_none(&host).into_iter().map(str::to_owned).collect();
    defaults.extend(implied.into_iter().map(|n| (n, "None".to_owned())));
    let params: Vec<Param> = params.into_iter().map(|(p, _)| p).collect();
    let key = format!("{head}.kw_only");
    let kw_only = match m.iter().find(|x| x.key == key) {
        None => KwOnly::None,
        Some(Meta { value: None, .. }) => KwOnly::All,
        Some(Meta { value: Some(p), .. }) => {
            if !params.iter().any(|q| &q.name == p) {
                return Err(format!("{at}: kw_only = \"{p}\" names no parameter"));
            }
            KwOnly::From(p.clone())
        }
    };
    let stream = meta_rust(m, &format!("{head}.stream")).map(|p| p.rsplit("::").next().unwrap_or(&p).to_owned());
    Ok((receiver, Sig { params, ret, err, err_ty, defaults, kw_only, stream }))
}

/// `Result<T, E>` -> `(Result<(), E>, T)`.
fn unit_ok(at: &str, s: &syn::Signature) -> Res<(syn::Signature, syn::Type)> {
    let syn::ReturnType::Type(arrow, t) = &s.output else {
        return Err(format!("{at}: a command that answers returns a value"));
    };
    let unit: syn::Type = syn::parse_quote!(());
    let mut out = s.clone();
    let value = match result_ok(t) {
        Some((ok, _)) => {
            let value = ok.clone();
            let mut rt = (**t).clone();
            if let syn::Type::Path(rp) = &mut rt
                && let Some(last) = rp.path.segments.last_mut()
                && let syn::PathArguments::AngleBracketed(ra) = &mut last.arguments
                && let Some(syn::GenericArgument::Type(first)) = ra.args.first_mut()
            {
                *first = unit;
            }
            out.output = syn::ReturnType::Type(*arrow, Box::new(rt));
            value
        }
        None => {
            out.output = syn::ReturnType::Type(*arrow, Box::new(unit));
            (**t).clone()
        }
    };
    Ok((out, value))
}

/// `Reply<T>` (or `Result<Reply<T>, E>`) split into the `()`-returning signature and the reply
/// type; `None` for any other return.
pub(super) fn reply_split(at: &str, s: &syn::Signature, al: &Aliases) -> Res<Option<(syn::Signature, ReplySig)>> {
    let is_reply = |t: &syn::Type| matches!(t, syn::Type::Path(p) if p.path.segments.last().is_some_and(|s| s.ident == "Reply"));
    let syn::ReturnType::Type(_, t) = &s.output else { return Ok(None) };
    if !is_reply(t) && !result_ok(t).is_some_and(|(ok, _)| is_reply(ok)) {
        return Ok(None);
    }
    let (out, reply_ty) = unit_ok(at, s)?;
    let syn::Type::Path(p) = &reply_ty else { unreachable!("checked above") };
    let seg = p.path.segments.last().expect("checked above");
    let bad = || format!("{at}: `Reply<T>` takes the awaited type");
    let syn::PathArguments::AngleBracketed(a) = &seg.arguments else { return Err(bad()) };
    let Some(syn::GenericArgument::Type(inner)) = a.args.first() else { return Err(bad()) };
    let reply = match result_ok(inner) {
        Some((ok, e)) => ReplySig { ret: rty(&format!("{at} -> Reply"), ok, al)?, err: Some(ErrMap::Auto), err_ty: e },
        None => ReplySig { ret: rty(&format!("{at} -> Reply"), inner, al)?, err: None, err_ty: None },
    };
    Ok(Some((out, reply)))
}

/// `(T, E's last segment)` of `Result<T, E>`; `E` is `None` for an alias's `Result<T>`.
fn result_ok(t: &syn::Type) -> Option<(&syn::Type, Option<String>)> {
    let syn::Type::Path(p) = t else { return None };
    let seg = p.path.segments.last()?;
    if seg.ident != "Result" {
        return None;
    }
    let syn::PathArguments::AngleBracketed(a) = &seg.arguments else { return None };
    let mut tys = a.args.iter().filter_map(|g| match g {
        syn::GenericArgument::Type(t) => Some(t),
        _ => None,
    });
    let ok = tys.next()?;
    let err = tys.next().and_then(|e| match e {
        syn::Type::Path(p) => p.path.segments.last().map(|s| s.ident.to_string()),
        _ => None,
    });
    Some((ok, err))
}

const PRIMS: [&str; 15] =
    ["bool", "i8", "i16", "i32", "i64", "isize", "u8", "u16", "u32", "u64", "usize", "f32", "f64", "char", "u128"];

/// Anything unsupported is refused by name.
pub(super) fn rty(at: &str, t: &syn::Type, al: &Aliases) -> Res<RTy> {
    let refuse = |why: &str| Err(format!("{at}: `{}` {why}", quote(t).replace(' ', "")));
    let args = |seg: &syn::PathSegment| -> Vec<syn::Type> {
        match &seg.arguments {
            syn::PathArguments::AngleBracketed(a) => a
                .args
                .iter()
                .filter_map(|g| match g {
                    syn::GenericArgument::Type(t) => Some(t.clone()),
                    _ => None,
                })
                .collect(),
            _ => Vec::new(),
        }
    };
    let assoc_item = |seg: &syn::PathSegment| -> Option<syn::Type> {
        let syn::PathArguments::AngleBracketed(a) = &seg.arguments else { return None };
        a.args.iter().find_map(|g| match g {
            syn::GenericArgument::AssocType(at) if at.ident == "Item" => Some(at.ty.clone()),
            _ => None,
        })
    };
    match t {
        syn::Type::Paren(p) => rty(at, &p.elem, al),
        syn::Type::Array(arr) => Ok(RTy::Slice(Box::new(rty(at, &arr.elem, al)?))),
        syn::Type::Tuple(tu) if tu.elems.is_empty() => Ok(RTy::Unit),
        syn::Type::Tuple(tu) => Ok(RTy::Tuple(tu.elems.iter().map(|e| rty(at, e, al)).collect::<Res<_>>()?)),
        syn::Type::Reference(r) => match &*r.elem {
            syn::Type::Slice(s) if quote(&s.elem) == "u8" => Ok(RTy::Bytes),
            syn::Type::Slice(s) => Ok(RTy::Slice(Box::new(rty(at, &s.elem, al)?))),
            inner => match rty(at, inner, al)? {
                RTy::Str if quote(inner) == "str" => Ok(RTy::StrRef),
                RTy::Json { .. } => Ok(RTy::Json { by_ref: true }),
                RTy::JsonMap { .. } => Ok(RTy::JsonMap { by_ref: true }),
                RTy::Declared { name, .. } => Ok(RTy::Declared { name, by_ref: true }),
                RTy::HostRef => Ok(RTy::HostRef),
                _ => refuse("is a reference the binding has no host shape for (only &str, &[T], &[u8], &Value, &Map and &<declared>)"),
            },
        },
        syn::Type::ImplTrait(it) => {
            for b in &it.bounds {
                let syn::TypeParamBound::Trait(tb) = b else { continue };
                let Some(seg) = tb.path.segments.last() else { continue };
                if seg.ident == "Into" && args(seg).first().is_some_and(|a| quote(a) == "String") {
                    return Ok(RTy::IntoString);
                }
                if seg.ident == "Iterator"
                    && let Some(item) = assoc_item(seg)
                {
                    return Ok(RTy::Iter(Box::new(rty(at, &item, al)?)));
                }
                if seg.ident == "IntoIterator"
                    && let Some(item) = assoc_item(seg)
                {
                    let inner = rty(at, &item, al)?;
                    return Ok(RTy::IntoIter(Box::new(match inner {
                        RTy::IntoString => RTy::Str,
                        other => other,
                    })));
                }
            }
            refuse("is an `impl Trait` the binding cannot name (only impl Into<String>, impl Iterator<Item = T> and impl IntoIterator<Item = T>)")
        }
        syn::Type::Path(p) => {
            let seg = p.path.segments.last().expect("a path");
            let head = seg.ident.to_string();
            let a = args(seg);
            let one = |i: usize| -> Res<RTy> {
                a.get(i).map(|t| rty(at, t, al)).unwrap_or_else(|| Err(format!("{at}: {head} with too few type arguments")))
            };
            Ok(match head.as_str() {
                "String" | "str" => RTy::Str,
                h if PRIMS.contains(&h) => RTy::Prim(head),
                "Value" => RTy::Json { by_ref: false },
                "Map" => RTy::JsonMap { by_ref: false },
                "Duration" => RTy::Duration,
                "SystemTime" => RTy::SystemTime,
                "Option" => RTy::Option(Box::new(one(0)?)),
                "Vec" => RTy::Vec(Box::new(one(0)?)),
                "Maybe" => RTy::Maybe(Box::new(one(0)?)),
                "HashMap" => {
                    if one(0)? != RTy::Str {
                        return refuse("is a map whose key is not a String; a host dict here is keyed by str");
                    }
                    RTy::Map(Box::new(one(1)?))
                }
                "Self" => RTy::SelfTy,
                "HostRef" => RTy::HostRef,
                "Arc" | "Box" if a.len() == 1 && matches!(a[0], syn::Type::TraitObject(_)) => {
                    let syn::Type::TraitObject(o) = &a[0] else { unreachable!() };
                    let tr = o.bounds.iter().find_map(|b| match b {
                        syn::TypeParamBound::Trait(tb) => tb.path.segments.last().map(|s| s.ident.to_string()),
                        _ => None,
                    });
                    match (head.as_str(), tr.as_deref()) {
                        ("Arc", Some(tr)) => RTy::Dyn(tr.to_owned()),
                        ("Box", Some("StreamSink")) => RTy::StreamHalf { sink: true },
                        ("Box", Some("StreamSource")) => RTy::StreamHalf { sink: false },
                        _ => return refuse("is a trait object the binding has no host shape for (`Arc<dyn ProxiedTrait>`, `Box<dyn StreamSink/StreamSource<T>>`)"),
                    }
                }
                "Arc" if a.len() == 1 && matches!(&a[0], syn::Type::Path(tp) if tp.qself.is_none()) => {
                    let syn::Type::Path(tp) = &a[0] else { unreachable!() };
                    RTy::Shared(tp.path.segments.last().expect("a path").ident.to_string())
                }
                "Result" => return refuse("is a nested Result; only the return type may be one"),
                _ if a.is_empty() && p.path.segments.len() == 1 && al.get(&head).is_some() => {
                    return rty(at, al.get(&head).expect("checked"), al);
                }
                // A generic (`Fallbacks<T>`) is named by its head.
                _ if !matches!(head.as_str(), "Box" | "Arc") && head.chars().next().is_some_and(char::is_uppercase) => {
                    RTy::Declared { name: head, by_ref: false }
                }
                _ => return refuse("has no host shape (Arc, Box ... are not exportable)"),
            })
        }
        _ => refuse("has no host shape"),
    }
}

pub(super) fn python_gated(attrs: &[syn::Attribute]) -> bool {
    attrs.iter().any(|a| a.path().is_ident("cfg") && quote::ToTokens::to_token_stream(&a.meta).to_string().replace(' ', "").contains("feature=\"python\""))
}
