//! Records: a declaration planned into a [`Rec`] (fields or union arms, both directions), then
//! written as a `#[napi(object)]` struct or a tagged enum with its `From`/`TryFrom` impls.

use std::rc::Rc;

use super::types::{from_js, n_of_field, napi_ty, to_js, ts_override, N};
use super::{by_key, doc_line, host_name, js_name, key, push_unique, raw, summary, Cx, Dir, HAND_WRITTEN, LINKED};
use crate::ir::*;
use crate::node::{camel, field_ts};

pub(super) struct Rec {
    pub(super) decl: String,
    pub(super) ident: String,
    core: String,
    pub(super) kind: RecKind,
    no_from_js: Option<String>,
    pub(super) deps: Vec<String>,
    doc: Vec<String>,
}

pub(super) enum RecKind {
    Struct { tag: Option<(String, String)>, fields: Vec<FieldPlan> },
    Union(Vec<Arm>),
}

pub(super) struct FieldPlan {
    ident: String,
    js: Option<String>,
    napi_ty: String,
    ts: Option<String>,
    core_field: String,
    to_js: String,
    from_js: Option<String>,
    doc: Vec<String>,
}

pub(super) struct Arm {
    variant: String,
    pub(super) member: String,
    /// `(js key, value)`: how `FromNapiValue` tells this arm apart.
    tag: Option<(String, String)>,
}

pub(super) fn ident_path(r: &Rec) -> String {
    format!("crate::generated::records::{}", r.ident)
}

pub(super) fn rec_for(cx: &Cx, name: &str, dir: Dir) -> Result<Rc<Rec>, String> {
    let r = rec(cx, name)?;
    if dir == Dir::FromJs
        && let Some(why) = &r.no_from_js {
            return Err(format!("`{}` cannot be built from JS: {why}", r.decl));
        }
    Ok(r)
}

pub(super) fn rec(cx: &Cx, name: &str) -> Result<Rc<Rec>, String> {
    if let Some((_, state)) = cx.recs.borrow().iter().find(|(n, _)| n == name) {
        return match state {
            Some(r) => r.clone(),
            None => Err(format!("`{name}` contains itself")),
        };
    }
    cx.recs.borrow_mut().push((name.to_owned(), None));
    let was = cx.in_params.replace(false);
    let plan = plan_rec(cx, name).map(Rc::new);
    cx.in_params.set(was);
    let mut recs = cx.recs.borrow_mut();
    let slot = recs.iter_mut().find(|(n, _)| n == name).expect("pushed above");
    slot.1 = Some(plan.clone());
    plan
}

/// `SOURCES` files whose host-only declarations stay unbound although their path is known: the rule
/// below once keyed on the `host/` and `provider/` directories, which these files were not in.
/// Binding them (`RawFunctionDescription`, and through it the `TaskGroup*` arms) is its own change.
const UNBOUND: &[&str] = &["llm/chat_context", "llm/provider_format/tools", "telemetry/report", "worker/job"];

/// A projection's `from`; a binding-target record is itself; a host-only declaration only where
/// the path cannot be guessed wrong: its `SOURCES` path is its file's module, or the root.
fn core_path(d: &Decl) -> Option<String> {
    if let Some(p) = d.target(Host::Node).and_then(|t| t.into_type.clone()) {
        return Some(p);
    }
    if let Some(p) = d.targets_for(Host::Binding).find_map(|t| t.into_type.clone()) {
        return Some(p);
    }
    if d.targets_for(Host::Binding).next().is_some() {
        return Some(d.core_path.clone());
    }
    let file = d.source.strip_prefix("rust/crates/lk-agents-abi/src/")?.strip_suffix(".rs")?;
    let file = file.strip_suffix("/mod").unwrap_or(file);
    if UNBOUND.contains(&file) {
        return None;
    }
    let inner: String = d.rust_module.iter().map(|m| format!("{m}::")).collect();
    let module = d.core_path.strip_suffix(&format!("::{inner}{}", d.rust_name))?;
    if module == format!("lk_agents_abi::{}", file.replace('/', "::")) {
        return Some(d.core_path.clone());
    }
    // A module whose top-level types are re-exported at the root.
    (module == "lk_agents_abi" && inner.is_empty()).then(|| d.core_path.clone())
}

fn plan_rec(cx: &Cx, name: &str) -> Result<Rec, String> {
    let d = by_key(cx.decls, name).ok_or_else(|| format!("`{name}` is not declared"))?;
    let core = core_path(d).ok_or_else(|| {
        "no core path: the IR names none (give the declaration #[node(from = \"..\")] or a binding target)".to_owned()
    })?;
    let krate = core.split("::").next().unwrap_or_default();
    if !LINKED.contains(&krate) {
        return Err(format!("`{krate}` is not linked by lk-agents-node"));
    }
    let ident = js_name(cx.decls, d);
    if HAND_WRITTEN.contains(&ident.as_str()) {
        return Err(format!("`{ident}` is a hand-written napi name"));
    }
    let node = d.target(Host::Node);
    let mut deps = Vec::new();
    let mut no_from_js = None;
    let kind = match &d.body {
        Body::Unit(_) => return Err("a string enum crosses as its string".to_owned()),
        Body::Struct(fields) => {
            let tag = match node {
                Some(t) => t.tag.clone(),
                None => d.targets.iter().find_map(|t| t.tag.clone()),
            };
            let mut plans = Vec::new();
            for f in fields {
                if f.private {
                    no_from_js.get_or_insert(format!("`{}` is host-private", f.rust_name));
                    continue;
                }
                let n = n_of_field(cx, name, f)?;
                collect_deps(&n, &mut deps);
                let label = format!("{}.{}", host_name(d), f.name);
                let (optional, ts) = match node {
                    Some(_) => match field_ts(cx.decls, d, f, true)? {
                        Some(ft) => (ft.optional, Some(ft.ty)),
                        None => continue,
                    },
                    None => (false, ts_override(cx, &n)?),
                };
                let is_option = matches!(n, N::Option(_));
                let base = napi_ty(cx, &n)?;
                let napi_ty = if optional && !is_option { format!("Option<{base}>") } else { base };
                let place = format!("&v.{}", f.rust_name);
                let mut to = to_js(cx, &n, &place, true)?;
                if optional && !is_option {
                    to = format!("Some({to})");
                }
                // `#[binding(skip)]`: host-supplied; Node applies pydantic's default (`created_at`).
                if f.emit_skip {
                    to = match (default_rust(f, &n), optional && !is_option) {
                        (Some(def), true) => format!("Some({})", to_js(cx, &n, &def, false)?),
                        (Some(def), false) => to_js(cx, &n, &def, false)?,
                        (None, _) if optional => "None".to_owned(),
                        (None, _) => to,
                    };
                }
                let field_in = format!("v.{}", raw(&f.name));
                let from = if optional && !is_option {
                    match (from_js(cx, &n, "v", &label), default_rust(f, &n)) {
                        (Ok(e), Some(def)) => Some(format!("match {field_in} {{ Some(v) => {e}, None => {def} }}")),
                        (Ok(_), None) => {
                            no_from_js.get_or_insert(format!("`{}`'s Python default has no Rust spelling here", f.rust_name));
                            None
                        }
                        (Err(why), _) => {
                            no_from_js.get_or_insert(format!("`{}`: {why}", f.rust_name));
                            None
                        }
                    }
                } else {
                    match from_js(cx, &n, &field_in, &label) {
                        Ok(e) => Some(e),
                        Err(why) => {
                            no_from_js.get_or_insert(format!("`{}`: {why}", f.rust_name));
                            None
                        }
                    }
                };
                plans.push(FieldPlan {
                    ident: raw(&f.name),
                    js: explicit_js(&f.name),
                    napi_ty,
                    ts,
                    core_field: f.rust_name.clone(),
                    to_js: to,
                    from_js: from,
                    doc: f.doc.clone(),
                });
            }
            RecKind::Struct { tag, fields: plans }
        }
        Body::Union(members) => {
            let mut arms = Vec::new();
            for m in members {
                let Ty::Declared(member) = &m.ty else {
                    return Err(format!("arm `{}` is not a declared record ({:?})", m.rust_name, m.ty));
                };
                let member = &by_key(cx.decls, member).map(key).or_else(|| cx.decls.iter().find(|d| &d.rust_name == member).map(key)).unwrap_or_else(|| member.clone());
                let r = rec(cx, member).map_err(|e| format!("arm `{}`: {e}", m.rust_name))?;
                let tag = match &r.kind {
                    RecKind::Struct { tag: Some((k, v)), .. } => Some((camel(k), v.clone())),
                    RecKind::Struct { tag: None, .. } => {
                        let md = by_key(cx.decls, member).expect("planned");
                        md.fields().iter().find_map(|f| f.literal.as_ref().map(|l| (camel(&f.name), l.clone())))
                    }
                    RecKind::Union(_) => return Err(format!("arm `{}` is itself a union", m.rust_name)),
                };
                if tag.is_none() {
                    no_from_js.get_or_insert(format!("arm `{}` has no discriminant to dispatch on", m.rust_name));
                }
                if let Some(why) = &r.no_from_js {
                    no_from_js.get_or_insert(format!("arm `{}`: {why}", m.rust_name));
                }
                deps.push(member.clone());
                arms.push(Arm { variant: m.rust_name.clone(), member: member.clone(), tag });
            }
            let keys: Vec<&String> = arms.iter().filter_map(|a| a.tag.as_ref().map(|t| &t.0)).collect();
            if keys.windows(2).any(|w| w[0] != w[1]) {
                no_from_js.get_or_insert("the arms discriminate on different keys".to_owned());
            }
            RecKind::Union(arms)
        }
    };
    Ok(Rec { decl: name.to_owned(), ident, core, kind, no_from_js, deps, doc: d.doc.clone() })
}

pub(super) fn collect_deps(n: &N, out: &mut Vec<String>) {
    match n {
        N::Record(r) | N::Union(r) => push_unique(out, r),
        N::Option(i) | N::Vec(i) | N::Map(i) | N::Maybe(i) | N::Ref(i) | N::Iter(i) => collect_deps(i, out),
        N::Tuple(items) => items.iter().for_each(|i| collect_deps(i, out)),
        _ => {}
    }
}

fn explicit_js(name: &str) -> Option<String> {
    let simple = !name.starts_with('_') && name.chars().all(|c| c.is_ascii_lowercase() || c == '_');
    (!simple).then(|| camel(name))
}

/// A record key's Python default as the Rust value, for a key JS may omit.
fn default_rust(f: &Field, n: &N) -> Option<String> {
    if let Some(fac) = &f.default_factory {
        return match (fac.as_str(), n) {
            ("list" | "dict", _) => Some("Default::default()".to_owned()),
            ("time.time", N::Float(_)) => Some("lk_agents_abi::epoch_secs(std::time::SystemTime::now())".to_owned()),
            _ => None,
        };
    }
    let d = f.default.as_deref()?;
    match (d, n) {
        ("True", N::Bool) => Some("true".to_owned()),
        ("False", N::Bool) => Some("false".to_owned()),
        (s, N::Str) if s.len() >= 2 && (s.starts_with('"') || s.starts_with('\'')) => {
            Some(format!("{:?}.to_owned()", &s[1..s.len() - 1]))
        }
        (s, N::Int(w)) if s.parse::<i64>().is_ok() => Some(format!("{s}{w}")),
        (s, N::Float(w)) if s.parse::<f64>().is_ok() => {
            Some(if s.contains('.') { format!("{s}_{w}") } else { format!("{s}.0_{w}") })
        }
        (s, N::Duration) if s.parse::<f64>().is_ok() => Some(format!("std::time::Duration::from_secs_f64({})", if s.contains('.') { s.to_owned() } else { format!("{s}.0") })),
        _ => None,
    }
}

pub(super) fn emit_rec(cx: &Cx, r: &Rec) -> String {
    let mut b = String::new();
    for l in summary(&r.doc) {
        b.push_str(&doc_line(&l, ""));
    }
    let id = &r.ident;
    let core = &r.core;
    match &r.kind {
        RecKind::Struct { tag, fields } => {
            b.push_str(&format!("#[napi(object, js_name = \"{id}\")]\npub struct {id} {{\n"));
            if let Some((k, v)) = tag {
                b.push_str(&format!("    #[napi(js_name = \"{}\", ts_type = \"'{v}'\")]\n    pub {}: String,\n", camel(k), tag_ident(k)));
            }
            for f in fields {
                for l in summary(&f.doc) {
                    b.push_str(&doc_line(&l, "    "));
                }
                let mut attrs = Vec::new();
                if let Some(js) = &f.js {
                    attrs.push(format!("js_name = \"{js}\""));
                }
                if let Some(t) = &f.ts {
                    attrs.push(format!("ts_type = {t:?}"));
                }
                if !attrs.is_empty() {
                    b.push_str(&format!("    #[napi({})]\n", attrs.join(", ")));
                }
                b.push_str(&format!("    pub {}: {},\n", f.ident, f.napi_ty));
            }
            b.push_str("}\n\n");
            b.push_str(&format!("impl From<&{core}> for {id} {{\n    fn from(v: &{core}) -> Self {{\n        {id} {{\n"));
            if let Some((k, v)) = tag {
                b.push_str(&format!("            {}: {v:?}.to_owned(),\n", tag_ident(k)));
            }
            for f in fields {
                b.push_str(&format!("            {}: {},\n", f.ident, f.to_js));
            }
            b.push_str("        }\n    }\n}\n");
            if r.no_from_js.is_none() {
                // `From` when nothing can fail, else `TryFrom` with a `HostError`.
                let fallible = fields.iter().any(|f| f.from_js.as_deref().is_some_and(|e| e.contains('?')));
                if fallible {
                    b.push_str(&format!(
                        "\nimpl TryFrom<{id}> for {core} {{\n    type Error = crate::host_error::JsError;\n\n    fn try_from(v: {id}) -> Result<Self, Self::Error> {{\n        Ok({core} {{\n"
                    ));
                } else {
                    b.push_str(&format!("\nimpl From<{id}> for {core} {{\n    fn from(v: {id}) -> Self {{\n        {core} {{\n"));
                }
                for f in fields {
                    b.push_str(&format!("            {}: {},\n", f.core_field, f.from_js.as_deref().expect("checked")));
                }
                b.push_str(if fallible { "        })\n    }\n}\n" } else { "        }\n    }\n}\n" });
            }
        }
        RecKind::Union(arms) => {
            let members: Vec<Rc<Rec>> = arms.iter().map(|a| rec(cx, &a.member).expect("planned")).collect();
            b.push_str(&format!("pub enum {id} {{\n"));
            for (a, m) in arms.iter().zip(&members) {
                b.push_str(&format!("    {}({}),\n", a.variant, m.ident));
            }
            b.push_str("}\n\n");
            b.push_str(&format!(
                "impl napi::bindgen_prelude::TypeName for {id} {{\n    fn type_name() -> &'static str {{\n        \"{id}\"\n    }}\n\n    fn value_type() -> napi::ValueType {{\n        napi::ValueType::Object\n    }}\n}}\n\n"
            ));
            b.push_str(&format!(
                "impl napi::bindgen_prelude::ToNapiValue for {id} {{\n    unsafe fn to_napi_value(env: napi::sys::napi_env, val: Self) -> napi::Result<napi::sys::napi_value> {{\n        match val {{\n"
            ));
            for a in arms {
                b.push_str(&format!(
                    "            {id}::{}(v) => unsafe {{ napi::bindgen_prelude::ToNapiValue::to_napi_value(env, v) }},\n",
                    a.variant
                ));
            }
            b.push_str("        }\n    }\n}\n\n");
            b.push_str(&format!("impl From<&{core}> for {id} {{\n    fn from(v: &{core}) -> Self {{\n        match v {{\n"));
            for (a, m) in arms.iter().zip(&members) {
                b.push_str(&format!("            {core}::{}(x) => {id}::{}({}::from(x)),\n", a.variant, a.variant, m.ident));
            }
            b.push_str("        }\n    }\n}\n");
            if r.no_from_js.is_none() {
                let key = &arms[0].tag.as_ref().expect("checked").0;
                b.push_str(&format!(
                    "\nimpl napi::bindgen_prelude::FromNapiValue for {id} {{\n    unsafe fn from_napi_value(env: napi::sys::napi_env, value: napi::sys::napi_value) -> napi::Result<Self> {{\n        let obj = unsafe {{ <napi::bindgen_prelude::Object as napi::bindgen_prelude::FromNapiValue>::from_napi_value(env, value)? }};\n        let tag: String = obj.get_named_property({key:?})?;\n        match tag.as_str() {{\n"
                ));
                for a in arms {
                    let v = &a.tag.as_ref().expect("checked").1;
                    b.push_str(&format!(
                        "            {v:?} => Ok({id}::{}(unsafe {{ napi::bindgen_prelude::FromNapiValue::from_napi_value(env, value)? }})),\n",
                        a.variant
                    ));
                }
                b.push_str(&format!(
                    "            other => Err(napi::Error::new(napi::Status::InvalidArg, format!(\"{id}: unknown {key} `{{other}}`\"))),\n        }}\n    }}\n}}\n\nimpl napi::bindgen_prelude::ValidateNapiValue for {id} {{}}\n"
                ));
                b.push_str(&format!(
                    "\nimpl TryFrom<{id}> for {core} {{\n    type Error = crate::host_error::JsError;\n\n    fn try_from(v: {id}) -> Result<Self, Self::Error> {{\n        Ok(match v {{\n"
                ));
                for a in arms {
                    b.push_str(&format!("            {id}::{}(x) => {core}::{}(x.try_into()?),\n", a.variant, a.variant));
                }
                b.push_str("        })\n    }\n}\n");
            }
        }
    }
    b
}

fn tag_ident(k: &str) -> String {
    raw(k)
}
