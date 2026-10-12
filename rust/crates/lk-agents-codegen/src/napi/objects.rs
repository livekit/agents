//! The `#[napi]` classes: `#[host(object)]` impls with their methods (sync, async, reply), and
//! `#[host(stream)]` push/recv classes.

use super::functions::{body, err_map, napi_attr, params, ret, P};
use super::records::collect_deps;
use super::types::{from_js, n_of_rty, napi_ty, to_js, ts};
use super::{arg, doc_line, linked, on_hand_written, push_unique, raw, snake, summary, Cx, ERROR_WITH, HAND_WRITTEN, NATIVE_INTS};
use crate::ir::*;
use crate::node::camel;

pub(super) fn object(cx: &Cx, o: &Object, wanted: &mut Vec<String>, gaps: &mut Vec<(String, String)>) -> Result<String, String> {
    linked(&o.core_path)?;
    if !o.consts.is_empty() {
        return Err("not yet: class constants".to_owned());
    }
    if on_hand_written(o).is_none() && HAND_WRITTEN.contains(&o.name.as_str()) {
        return Err(format!("`{}` is a hand-written napi name", o.name));
    }
    let mutates = o.methods.iter().any(|m| m.receiver == Receiver::Mut);
    if o.cell == Cell::Arc && mutates {
        return Err("a `&mut self` method in an `arc` cell".to_owned());
    }
    let core = &o.core_path;
    let field = match o.cell {
        Cell::Plain => core.clone(),
        Cell::Mutex => format!("std::sync::Mutex<{core}>"),
        Cell::Arc => format!("std::sync::Arc<{core}>"),
    };
    let mut b = String::new();
    for l in summary(&o.doc) {
        b.push_str(&doc_line(&l, ""));
    }
    let name = &o.name;
    match on_hand_written(o) {
        // Another `#[napi] impl` on the hand-written class; napi keys it by bare ident, so import it.
        Some(class) => {
            let ident = class.rsplit("::").next().unwrap_or(class);
            b.push_str(&format!("use {class};\n\n#[napi]\nimpl {ident} {{\n"));
        }
        None => b.push_str(&format!("#[napi(js_name = \"{name}\")]\npub struct {name} {{\n    pub(crate) inner: {field},\n}}\n\n#[napi]\nimpl {name} {{\n")),
    }
    let mut first = true;
    for m in &o.methods {
        let at = format!("{core}::{}", m.rust_name);
        let mut mine = Vec::new();
        match method(cx, o, m, &mut mine) {
            Ok(text) => {
                if !first {
                    b.push('\n');
                }
                first = false;
                b.push_str(&text);
                wanted.extend(mine);
            }
            Err(e) => gaps.push((at, e)),
        }
    }
    b.push_str("}\n");
    Ok(b)
}

fn method(cx: &Cx, o: &Object, m: &Method, wanted: &mut Vec<String>) -> Result<String, String> {
    if m.name.starts_with("__") {
        return Err(format!("not yet: `{}` is a Python protocol method", m.name));
    }
    let is_async = m.kind == MethodKind::Async;
    if is_async && o.cell != Cell::Arc {
        return Err("#[host(async)] needs cell = \"arc\"".to_owned());
    }
    let ps = params(cx, &m.sig, is_async, wanted)?;
    let list = ps.iter().map(|p| p.call.as_str()).collect::<Vec<_>>().join(", ");
    let core = &o.core_path;
    let pad = "    ";
    let mut b = String::new();
    let js = camel(&m.name);
    match m.kind {
        MethodKind::New => {
            let wrap = match o.cell {
                Cell::Plain => "out",
                Cell::Mutex => "std::sync::Mutex::new(out)",
                Cell::Arc => "std::sync::Arc::new(out)",
            };
            b.push_str(&format!("{pad}#[napi(constructor)]\n"));
            let expr = format!("{} {{ inner: {wrap} }}", o.name);
            b.push_str(&body(cx, "new", None, &ps, &[], &format!("{core}::{}({list})", m.rust_name), &m.sig, "Self", &expr, pad)?);
            return Ok(b);
        }
        MethodKind::Static => {
            let (ret_ty, ts_ret, ret_expr) = ret(cx, &m.sig.ret, wanted)?;
            b.push_str(&napi_attr("", &js, ts_ret.as_deref(), pad));
            b.push_str(&body(cx, &raw(&m.name), None, &ps, &[], &format!("{core}::{}({list})", m.rust_name), &m.sig, &ret_ty, &ret_expr, pad)?);
            return Ok(b);
        }
        MethodKind::Setter if m.sig.params.len() != 1 || m.sig.ret != RTy::Unit => {
            return Err("#[host(setter)] takes one value and returns nothing".to_owned());
        }
        MethodKind::Getter if !m.sig.params.is_empty() => return Err("#[host(getter)] takes no arguments".to_owned()),
        _ => {}
    }
    let recv = if o.cell == Cell::Plain && m.receiver == Receiver::Mut { "&mut self" } else { "&self" };
    if let Some(reply) = &m.reply {
        let (ret_ty, ts_ret, ret_expr) = ret(cx, &reply.ret, wanted)?;
        return reply_method(cx, o, m, &ps, &ret_ty, ts_ret.as_deref(), &ret_expr);
    }
    let (ret_ty, ts_ret, ret_expr) = ret(cx, &m.sig.ret, wanted)?;
    if is_async {
        return async_method(cx, m, &ps, &ret_ty, ts_ret.as_deref(), &ret_expr);
    }
    // napi derives C symbols from the ident, which `r#type` cannot be; fall back to the Rust name.
    let ident = |name: &str| if raw(name).starts_with("r#") { m.rust_name.clone() } else { raw(name) };
    let (kind, ident) = match m.kind {
        MethodKind::Getter => ("getter", ident(&m.name)),
        MethodKind::Setter => ("setter", format!("set_{}", m.name)),
        _ => ("", ident(&m.name)),
    };
    b.push_str(&napi_attr(kind, &js, ts_ret.as_deref(), pad));
    let (pre, target) = match o.cell {
        Cell::Mutex => (
            vec![format!(
                "let {}inner = self.inner.lock().unwrap_or_else(std::sync::PoisonError::into_inner);",
                if m.receiver == Receiver::Mut { "mut " } else { "" },
            )],
            format!("inner.{}({list})", m.rust_name),
        ),
        Cell::Plain | Cell::Arc => (Vec::new(), format!("self.inner.{}({list})", m.rust_name)),
    };
    b.push_str(&body(cx, &ident, Some(recv), &ps, &pre, &target, &m.sig, &ret_ty, &ret_expr, pad)?);
    Ok(b)
}

/// `command(reply)`: send on the JS thread (a refusal throws there), await on napi's runtime,
/// convert back on the JS thread.
fn reply_method(cx: &Cx, o: &Object, m: &Method, ps: &[P], ret_ty: &str, ts_ret: Option<&str>, ret_expr: &str) -> Result<String, String> {
    let pad = "    ";
    let reply = m.reply.as_ref().expect("a reply method");
    let sent_err = err_map(cx, &m.sig)?;
    let reply_err = err_map(cx, &Sig { err: reply.err.clone(), err_ty: reply.err_ty.clone(), ..m.sig.clone() })?;
    let mut b = String::new();
    let inner_ts = match (ts_ret, &reply.ret) {
        (Some(t), _) => t.to_owned(),
        (None, RTy::Unit) => "void".to_owned(),
        (None, t) => ts(cx, &n_of_rty(cx, t)?)?,
    };
    let ts = format!("Promise<{inner_ts}>");
    b.push_str(&napi_attr("", &camel(&m.name), Some(&ts), pad));
    let mut args = vec!["&self".to_owned(), "env: &'env Env".to_owned()];
    args.extend(ps.iter().map(|p| p.decl.clone()));
    b.push_str(&format!(
        "{pad}pub fn {}<'env>({}) -> napi::Result<napi::bindgen_prelude::PromiseRaw<'env, {ret_ty}>> {{\n",
        raw(&m.name),
        args.join(", ")
    ));
    let list = ps.iter().map(|p| p.call.as_str()).collect::<Vec<_>>().join(", ");
    let this = match o.cell {
        Cell::Mutex => "self.inner.lock().unwrap_or_else(std::sync::PoisonError::into_inner)".to_owned(),
        Cell::Plain | Cell::Arc => "self.inner".to_owned(),
    };
    b.push_str(&format!("{pad}    let reply = crate::host_error::run(env, || {{\n"));
    for p in ps.iter().flat_map(|p| &p.prelude) {
        b.push_str(&format!("{pad}        {p}\n"));
    }
    let call = format!("{this}.{}({list})", m.rust_name);
    match sent_err {
        Some(map) => b.push_str(&format!("{pad}        {call}.map_err({map})\n")),
        None => b.push_str(&format!("{pad}        Ok({call})\n")),
    }
    b.push_str(&format!("{pad}    }})?;\n"));
    let result = match reply_err {
        Some(map) => format!("reply.await.map_err({map})"),
        None => "Ok::<_, crate::host_error::JsError>(reply.await)".to_owned(),
    };
    b.push_str(&format!("{pad}    env.spawn_future_with_callback(async move {{ Ok({result}) }}, |env, out| match out {{\n"));
    b.push_str(&format!("{pad}        Ok(out) => Ok({ret_expr}),\n{pad}        Err(e) => Err(e.into_napi(env)),\n{pad}    }})\n{pad}}}\n"));
    Ok(b)
}

/// `async` on an `arc` cell: convert on the JS thread, run on napi's tokio runtime, convert back on
/// the JS thread, where a `HostError` can become a JS `Error` (it needs an `Env`).
fn async_method(cx: &Cx, m: &Method, ps: &[P], ret_ty: &str, ts_ret: Option<&str>, ret_expr: &str) -> Result<String, String> {
    let pad = "    ";
    let err = err_map(cx, &m.sig)?;
    let mut b = String::new();
    let ts = format!("Promise<{}>", ts_ret.map(str::to_owned).unwrap_or_else(|| ts_of_napi(ret_ty)));
    b.push_str(&napi_attr("", &camel(&m.name), Some(&ts), pad));
    let js = if m.sig.params.iter().any(|p| p.name == "env") { "js_env" } else { "env" };
    let mut args = vec!["&self".to_owned(), format!("{js}: &'env Env")];
    args.extend(ps.iter().map(|p| p.decl.clone()));
    b.push_str(&format!(
        "{pad}pub fn {}<'env>({}) -> napi::Result<napi::bindgen_prelude::PromiseRaw<'env, {ret_ty}>> {{\n",
        raw(&m.name),
        args.join(", ")
    ));
    let names: Vec<String> = m.sig.params.iter().map(|p| arg(&p.name)).collect();
    if !names.is_empty() {
        b.push_str(&format!("{pad}    let ({},) = crate::host_error::run({js}, || {{\n", names.join(", ")));
        for p in ps.iter().flat_map(|p| &p.prelude) {
            b.push_str(&format!("{pad}        {p}\n"));
        }
        b.push_str(&format!("{pad}        Ok(({},))\n{pad}    }})?;\n", names.join(", ")));
    }
    b.push_str(&format!("{pad}    let inner = std::sync::Arc::clone(&self.inner);\n"));
    let list = ps.iter().map(|p| p.call.as_str()).collect::<Vec<_>>().join(", ");
    let call = format!("inner.{}({list}).await", m.rust_name);
    let result = match err {
        Some(map) => format!("{call}.map_err({map})"),
        None => format!("Ok::<_, crate::host_error::JsError>({call})"),
    };
    b.push_str(&format!("{pad}    {js}.spawn_future_with_callback(async move {{ Ok({result}) }}, |env, out| match out {{\n"));
    b.push_str(&format!("{pad}        Ok(out) => Ok({ret_expr}),\n{pad}        Err(e) => Err(e.into_napi(env)),\n{pad}    }})\n{pad}}}\n"));
    Ok(b)
}

fn ts_of_napi(t: &str) -> String {
    match t {
        "()" => "void".to_owned(),
        "bool" => "boolean".to_owned(),
        "String" => "string".to_owned(),
        t if NATIVE_INTS.contains(&t) || t == "f64" => "number".to_owned(),
        _ => "unknown".to_owned(),
    }
}

fn stream_err(cx: &Cx, name: &str) -> Result<String, String> {
    let e = cx.x.errors.iter().find(|e| e.name == name).ok_or_else(|| format!("error = \"{name}\" is not a #[host(error)] type"))?;
    if e.with.is_some() || ERROR_WITH.iter().any(|(n, _)| *n == e.name) {
        return ERROR_WITH
            .iter()
            .find(|(n, _)| *n == e.name)
            .map(|(_, f)| (*f).to_owned())
            .ok_or_else(|| format!("`{}` maps through a pyo3 converter with no Node counterpart", e.name));
    }
    push_unique(&mut cx.errors.borrow_mut(), &e.name);
    Ok(format!("crate::generated::errors::{}_to_node", snake(&e.name)))
}

/// `#[host(stream)]`, Python's shape: a synchronous `push`/`flush`/`endInput`, and Promise
/// `recv()` (`null` at the end) / `aclose()`. Items convert by Rust types, not pyo3 converters.
pub(super) fn stream(cx: &Cx, s: &Stream, wanted: &mut Vec<String>) -> Result<String, String> {
    let n = &s.name;
    if s.deadline != "crate::inference::NATIVE_CLOSE_DEADLINE" {
        return Err(format!("deadline = \"{}\" is binding code", s.deadline));
    }
    let err = stream_err(cx, &s.err_with)?;
    let out = n_of_rty(cx, &s.output).map_err(|e| format!("output: {e}"))?;
    collect_deps(&out, wanted);
    let out_ty = napi_ty(cx, &out)?;
    let out_ts = ts(cx, &out)?;
    let conv = to_js(cx, &out, "v", false).map_err(|e| format!("output: {e}"))?;
    let mut b = String::new();
    for l in summary(&s.doc) {
        b.push_str(&doc_line(&l, ""));
    }
    let sink_field = s.sink_ty.as_ref().map(|t| {
        if s.optional_sink.is_some() { format!("Option<std::sync::Arc<{t}>>") } else { format!("std::sync::Arc<{t}>") }
    });
    b.push_str(&format!("#[napi(js_name = \"{n}\")]\npub struct {n} {{\n"));
    if let Some(f) = &sink_field {
        b.push_str(&format!("    pub(crate) sink: {f},\n"));
    }
    b.push_str(&format!("    pub(crate) source: std::sync::Arc<tokio::sync::Mutex<Option<{}>>>,\n}}\n\n", s.source_ty));
    let source_new = "source: std::sync::Arc::new(tokio::sync::Mutex::new(Some(source)))";
    b.push_str(&format!("impl {n} {{\n"));
    match (&s.sink_ty, &s.optional_sink) {
        (Some(t), None) => b.push_str(&format!(
            "    pub fn new(sink: impl Into<std::sync::Arc<{t}>>, source: {}) -> Self {{\n        {n} {{ sink: sink.into(), {source_new} }}\n    }}\n",
            s.source_ty
        )),
        (Some(t), Some(_)) => b.push_str(&format!(
            "    pub fn new(sink: Option<impl Into<std::sync::Arc<{t}>>>, source: {}) -> Self {{\n        {n} {{ sink: sink.map(Into::into), {source_new} }}\n    }}\n",
            s.source_ty
        )),
        (None, _) => b.push_str(&format!("    pub fn new(source: {}) -> Self {{\n        {n} {{ {source_new} }}\n    }}\n", s.source_ty)),
    }
    if let Some(msg) = &s.optional_sink {
        b.push_str(&format!(
            "\n    fn sink(&self) -> Result<&std::sync::Arc<{}>, crate::host_error::JsError> {{\n        self.sink.as_ref().ok_or_else(|| crate::host_error::JsError::new(\"RuntimeError\", {msg:?}))\n    }}\n",
            s.sink_ty.as_deref().unwrap_or_default()
        ));
    }
    b.push_str(&format!("}}\n\n#[napi]\nimpl {n} {{\n"));
    let sink = if s.optional_sink.is_some() { "self.sink()?" } else { "self.sink" };
    if s.sink_ty.is_some() {
        let input = n_of_rty(cx, &s.input).map_err(|e| format!("input: {e}"))?;
        collect_deps(&input, wanted);
        let a = raw(&s.arg);
        let push_ty = napi_ty(cx, &input)?;
        let push_conv = from_js(cx, &input, &a, &s.arg).map_err(|e| format!("input: {e}"))?;
        b.push_str(&format!(
            "    #[napi(js_name = \"{}\")]\n    pub fn {}(&self, env: Env, {a}: {push_ty}) -> napi::Result<()> {{\n        crate::host_error::run(&env, || {{\n            let {a} = {push_conv};\n            {sink}.push({a}).map_err({err})\n        }})\n    }}\n\n",
            camel(&s.push),
            raw(&s.push)
        ));
        b.push_str(&format!(
            "    #[napi(js_name = \"flush\")]\n    pub fn flush(&self, env: Env) -> napi::Result<()> {{\n        crate::host_error::run(&env, || {sink}.flush().map_err({err}))\n    }}\n\n"
        ));
        if s.end_input {
            b.push_str(&format!(
                "    #[napi(js_name = \"endInput\")]\n    pub fn end_input(&self, env: Env) -> napi::Result<()> {{\n        crate::host_error::run(&env, || {{\n            {sink}.end_input();\n            Ok(())\n        }})\n    }}\n\n"
            ));
        }
    }
    b.push_str(&format!(
        "    /// The next item, or `null` at the end of the stream.\n    #[napi(js_name = \"recv\", ts_return_type = \"Promise<{out_ts} | null>\")]\n    pub fn recv<'env>(&self, env: &'env Env) -> napi::Result<napi::bindgen_prelude::PromiseRaw<'env, Option<{out_ty}>>> {{\n        let source = std::sync::Arc::clone(&self.source);\n        env.spawn_future_with_callback(\n            async move {{\n                let mut guard = source.lock().await;\n                let Some(src) = guard.as_mut() else {{ return Ok(Ok(None)) }};\n                Ok(match src.next().await {{\n                    Ok(Some(v)) => Ok(Some({conv})),\n                    Ok(None) => Ok(None),\n                    Err(e) => Err({err}(e)),\n                }})\n            }},\n            |env, out| out.map_err(|e| e.into_napi(env)),\n        )\n    }}\n\n"
    ));
    let end = match (&s.sink_ty, s.close_ends_input, &s.optional_sink) {
        (Some(_), true, None) => "        self.sink.end_input();\n".to_owned(),
        (Some(_), true, Some(_)) => "        if let Some(sink) = &self.sink {\n            sink.end_input();\n        }\n".to_owned(),
        _ => String::new(),
    };
    let close = if s.boxed { "src.close" } else { "Box::new(src).close" };
    b.push_str(&format!(
        "    #[napi(js_name = \"aclose\")]\n    pub async fn aclose(&self) -> napi::Result<()> {{\n{end}        if let Some(src) = self.source.lock().await.take() {{\n            {close}(lk_agents_abi::Ts::from_nanos(u64::MAX)).await;\n        }}\n        Ok(())\n    }}\n}}\n"
    ));
    Ok(b)
}
