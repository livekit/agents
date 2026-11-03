//! The declared proxies (`#[host(proxy)]` traits whose methods declare their calls), written into
//! their subject's adapter module: `<Trait>Proxy` implements the ABI trait over a host object.
//!
//! Each call dispatches through the runtime's HostInvoker (`lk_agents_runtime::host`), entered
//! by `crate::host` in a copy of a captured host context: the one its `#[host(context)]`
//! parameter carries (refused when it lacks a scope the call `requires`), else the one the proxy
//! captured when it was taken from the host. Generation refuses a host call emitted any other way. Every parameter crosses by its declared kind and the result is read by its
//! declared kind; a host failure becomes the declared error. `read_once` values are read when the
//! proxy is taken; a `bind` method returns a copy whose awaited calls pass the bound value last,
//! handed out through its declared native wrapper when it names one.

use crate::adapter::from_object;
use crate::decl::{HostForm, Param, Proxy, ProxyCall, ProxyMethod, Record, Ty};

pub fn proxy(p: &Proxy) -> Result<String, String> {
    let name = format!("{}Proxy", p.rust_name);
    let tr = &p.trait_path;
    let mut s = format!("// Declared in {}.\n\n", p.source);
    let mut fields = vec!["    host: ::std::sync::Arc<Py<PyAny>>,\n".to_owned()];
    let mut inits = vec!["host: ::std::sync::Arc::new(obj.clone().unbind())".to_owned()];
    let mut reads = match &p.view {
        Some((module, class)) => format!(
            "        let view = crate::py_calls::call(&obj.py().import({module:?})?.getattr({class:?})?, (obj,), None)?;\n        let obj = &view;\n"
        ),
        None => String::new(),
    };
    if p.handed_back {
        // The object the host passed, before any view over it.
        fields.push("    /// The host object it was taken from, handed back as itself.\n    source: ::std::sync::Arc<Py<PyAny>>,\n".into());
        inits.push("source".into());
        reads =
            format!("        let source = ::std::sync::Arc::new(obj.clone().unbind());\n{reads}");
    }
    let context_arg = "context: ::std::option::Option<&::lk_agents_abi::utils::host_ref::HostRef>";
    if p.captures_context() {
        reads.push_str("        let context = match context {\n            Some(context) => ::std::clone::Clone::clone(context),\n            None => crate::py_calls::capture_context(obj.py())?,\n        };\n");
        fields.push(
            "    /// The host's running loop and a copy of its context, captured when it was taken.\n    context: ::lk_agents_abi::utils::host_ref::HostRef,\n".into(),
        );
        inits.push("context".into());
    } else {
        reads.push_str("        let _ = context;\n");
    }
    if p.methods.iter().any(|m| m.kind == ProxyCall::Notify) {
        fields.push(
            "    /// The `notify` calls, delivered in order.\n    notifier: crate::py_calls::Notifier,\n".into(),
        );
        inits.push("notifier: ::std::default::Default::default()".into());
    }
    for m in p.methods.iter().filter(|m| m.kind == ProxyCall::ReadOnce) {
        let at = format!("{}::{}", tr, m.rust_name);
        fields.push(format!("    {}: {},\n", m.rust_name, m.ret.ok.rust()));
        let read = from_object("v", &m.ret.ok, &format!("\"{}.{}\"", p.rust_name, m.name), &at)?;
        reads.push_str(&format!(
            "        let {} = {{\n            let v = crate::py_calls::read(obj, {:?})?;\n            {read}?\n        }};\n",
            m.rust_name, m.name
        ));
        inits.push(m.rust_name.clone());
    }
    if let Some(bound) = p.bound() {
        fields.push(format!(
            "    /// Appended to every awaited call once bound.\n    {}: ::std::option::Option<{}>,\n",
            bound.name,
            bound.ty.rust()
        ));
        inits.push(format!("{}: None", bound.name));
    }
    s.push_str(&format!(
        "/// [`{tr}`] over a host `{host}`.\n#[derive(Clone)]\npub struct {name} {{\n{}}}\n\nimpl {name} {{\n    /// A registered native class as itself, any other host object behind this proxy.\n    pub fn from_host(obj: &Bound<'_, PyAny>) -> PyResult<::std::sync::Arc<dyn {tr}>> {{\n        if let Some(native) = crate::py_objects::native::<dyn {tr}>(obj) {{\n            return Ok(native);\n        }}\n        Ok(::std::sync::Arc::new(Self::take(obj, None)?))\n    }}\n\n    /// A host object a call returned, its calls run in a copy of the context that call ran in.\n    pub fn in_context(obj: &Bound<'_, PyAny>, context: &::lk_agents_abi::utils::host_ref::HostRef) -> PyResult<::std::sync::Arc<dyn {tr}>> {{\n        if let Some(native) = crate::py_objects::native::<dyn {tr}>(obj) {{\n            return Ok(native);\n        }}\n        Ok(::std::sync::Arc::new(Self::take(obj, Some(context))?))\n    }}\n\n    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.\n    pub fn owned(obj: &Bound<'_, PyAny>) -> PyResult<::std::boxed::Box<dyn {tr}>> {{\n        Ok(::std::boxed::Box::new(Self::take(obj, None)?))\n    }}\n\n    /// A host object a call returned for the core to own, run in a copy of that call's context.\n    pub fn owned_in_context(obj: &Bound<'_, PyAny>, context: &::lk_agents_abi::utils::host_ref::HostRef) -> PyResult<::std::boxed::Box<dyn {tr}>> {{\n        Ok(::std::boxed::Box::new(Self::take(obj, Some(context))?))\n    }}\n\n    fn take(obj: &Bound<'_, PyAny>, {context_arg}) -> PyResult<Self> {{\n{reads}        Ok({name} {{ {} }})\n    }}\n}}\n\n",
        fields.concat(),
        inits.join(", "),
        host = p.host,
    ));
    if p.handed_back {
        s.push_str(&format!(
            "impl {name} {{\n    /// The host object behind a proxy of `{tr}`; a native implementation has none.\n    pub fn to_host<'py>(py: Python<'py>, value: ::std::sync::Arc<dyn {tr}>) -> PyResult<Bound<'py, PyAny>> {{\n        let any: ::std::sync::Arc<dyn ::std::any::Any + Send + Sync> = value;\n        match any.downcast::<Self>() {{\n            Ok(proxy) => Ok(proxy.source.bind(py).clone()),\n            Err(_) => Err(::pyo3::exceptions::PyTypeError::new_err(\"a native `{}` has no host object to hand back\")),\n        }}\n    }}\n}}\n\n",
            p.rust_name
        ));
    }
    if p.async_trait {
        s.push_str("#[::async_trait::async_trait]\n");
    }
    s.push_str(&format!("impl {tr} for {name} {{\n"));
    let methods = p
        .methods
        .iter()
        .map(|m| method(p, m).and_then(|text| dispatched(p, m, text)))
        .collect::<Result<Vec<_>, _>>()?;
    s.push_str(&methods.join("\n"));
    s.push_str("}\n");
    Ok(s)
}

fn method(p: &Proxy, m: &ProxyMethod) -> Result<String, String> {
    let at = format!("{}::{}", p.trait_path, m.rust_name);
    let label = format!("{}.{}", p.rust_name, m.name);
    let params: Vec<String> =
        m.params.iter().map(|q| format!("{}: {}", q.name, param_type(q))).collect();
    // A future named by the method's own lifetime (`fn m<'a>(&'a self, ..) -> BoxFuture<'a, _>`)
    // keeps it: the proxy's future borrows nothing, so its parameters need not carry it.
    let named = m.boxed.as_deref().filter(|l| *l != "'_" && *l != "'static");
    let receiver = match (m.consumes, named) {
        (true, _) => "self: ::std::boxed::Box<Self>".to_owned(),
        (false, Some(l)) => format!("&{l} self"),
        (false, None) => "&self".to_owned(),
    };
    let sig = format!(
        "fn {}{}({})",
        m.rust_name,
        named.map(|l| format!("<{l}>")).unwrap_or_default(),
        std::iter::once(receiver).chain(params).collect::<Vec<_>>().join(", ")
    );
    match m.kind {
        ProxyCall::ReadOnce if m.lends => {
            let lent = match &m.ret.ok {
                Ty::Str => "&str".to_owned(),
                t => format!("&{}", t.rust()),
            };
            return Ok(format!("    {sig} -> {lent} {{\n        &self.{}\n    }}\n", m.rust_name));
        }
        ProxyCall::ReadOnce => {
            return Ok(format!(
                "    {sig} -> {} {{\n        ::std::clone::Clone::clone(&self.{})\n    }}\n",
                m.ret.ok.rust(),
                m.rust_name
            ));
        }
        ProxyCall::Bind => {
            let q = &m.params[0];
            let copy = format!(
                "::std::sync::Arc::new(Self {{ {}: Some({}), ..::std::clone::Clone::clone(self) }})",
                q.name,
                owned(q)
            );
            let out = match &m.wrap {
                None => copy,
                Some(wrap) => format!(
                    "{wrap}::bound({copy}, {}, ::lk_agents_runtime::process::host_clock())",
                    q.name
                ),
            };
            return Ok(format!(
                "    {sig} -> {} {{\n        Some({out})\n    }}\n",
                m.ret.ok.rust()
            ));
        }
        ProxyCall::Notify | ProxyCall::Await | ProxyCall::Call => {}
    }
    // Owned copies move into the call, which runs on the host's loop.
    let mut moved = String::new();
    let mut args = Vec::new();
    // A `CallContext` is what the call runs in, never an argument.
    for q in m.params.iter().filter(|q| !q.ty.hidden()) {
        moved.push_str(&format!("        let {}_arg = {};\n", q.name, owned(q)));
        let value = to_host(&format!("{}_arg", q.name), &q.ty, &at)?;
        if m.kind == ProxyCall::Notify
            && matches!(&q.ty, Ty::Optional(t) if matches!(**t, Ty::Error(_)))
        {
            return Err(format!("{at}: a notice takes its error, not an optional one"));
        }
        if m.kind == ProxyCall::Notify && matches!(q.ty, Ty::Error(_)) {
            // A later raise may consume the captured host exception before this notice drains.
            moved.push_str(&format!(
                "        let {name}_arg = match ::pyo3::Python::attach(|py| -> ::pyo3::PyResult<::pyo3::Py<::pyo3::PyAny>> {{ Ok(({value}).unbind()) }}) {{\n            Ok(value) => value,\n            Err(error) => {{\n                log::warn!(\"{label} failed: {{error}}\");\n                return;\n            }}\n        }};\n",
                name = q.name,
            ));
            args.push(format!("{}_arg.into_bound(py)", q.name));
        } else {
            args.push(value);
        }
    }
    let bound = match (m.kind, p.bound()) {
        (ProxyCall::Await, Some(b)) => {
            moved.push_str(&format!(
                "        let bound = ::std::clone::Clone::clone(&self.{});\n",
                b.name
            ));
            let arg = to_host("bound", &b.ty, &at)?;
            format!(
                "                let mut args = args.to_vec();\n                if let Some(bound) = bound {{\n                    args.push({arg});\n                }}\n"
            )
        }
        _ => String::new(),
    };
    // Every call dispatches through the runtime's HostInvoker, entered by `crate::host`.
    let host_call = format!(
        "::std::boxed::Box::new(move |py| {{\n                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; {}] = [{}];\n{bound}                Ok((host.bind(py).getattr({:?})?, ::pyo3::types::PyTuple::new(py, args)?))\n            }})",
        args.len(),
        args.join(", "),
        m.name
    );
    // What the call runs in: its own context parameter, else the context the proxy captured.
    let scope = match m.context {
        Some(i) if m.params[i].ty == Ty::TraceContext => {
            format!("&::lk_agents_abi::invoke::Traced(&self.context, {}.span)", m.params[i].name)
        }
        Some(i) if m.params[i].by_ref => m.params[i].name.clone(),
        Some(i) => format!("&{}", m.params[i].name),
        None => "&self.context".to_owned(),
    };
    let requires = m
        .requires
        .iter()
        .map(|s| format!("::lk_agents_abi::invoke::Scope::{s}"))
        .collect::<Vec<_>>()
        .join(", ");
    // A host stream it returns is pulled in a copy of the context its call ran in.
    let stream_context = |err: &str| {
        format!(
            "                let Some(context) = context.as_ref() else {{\n                    return Err(<{err} as crate::py_errors::FromHostErr>::from_host(py, &::pyo3::exceptions::PyRuntimeError::new_err(\"{label}: the call captured no host context\")));\n                }};\n"
        )
    };
    // Returned capabilities inherit the operation's context, including inside containers.
    let returns_proxy = (p.captures_context() || m.context.is_some()) && m.ret.ok.holds_proxy();
    let context = match m.ret.ok.holds_host_stream() || returns_proxy {
        true => format!(
            "        let context = ::lk_agents_abi::agent::HostContext::host_context({scope}).cloned();\n"
        ),
        false => String::new(),
    };
    let host = "        let host = ::std::sync::Arc::clone(&self.host);\n";
    if m.kind == ProxyCall::Call {
        let (ret, refused, wrap, err) = result_shape(m, &at)?;
        let binding = if m.ret.ok == Ty::Unit { "()" } else { "out" };
        let read = if returns_proxy {
            returned_value("o", &m.ret.ok, &label, &at)?
        } else {
            from_host("o", &m.ret.ok, &label, &at)?
        };
        let unpack = if context.is_empty() { String::new() } else { stream_context(&err) };
        return Ok(format!(
            "    {sig} -> {ret} {{\n{host}{moved}{context}        let out = ::lk_agents_runtime::host::call_now(&crate::py_calls::Entry, {host_call});\n        ::pyo3::Python::attach(|py| {{\n            let out = match out {{\n                Ok(out) => out,\n                Err(e) => {{\n{refused}                    return Err(<{err} as crate::py_errors::FromHostErr>::from_host(py, &e));\n                }}\n            }};\n{unpack}            let o = out.bind(py);\n            let {binding} = ({read}).map_err(|e| <{err} as crate::py_errors::FromHostErr>::from_host(py, &e))?;\n            Ok({wrap})\n        }})\n    }}\n"
        ));
    }
    if m.kind == ProxyCall::Notify {
        return Ok(format!(
            "    {sig} {{\n{host}{moved}        ::lk_agents_runtime::host::notify(&self.notifier, \"{label}\", {scope}, &[{requires}], {}, {host_call});\n    }}\n",
            m.lossy
        ));
    }
    let dispatch = match m.params.iter().find(|p| p.ty == Ty::Cancellation) {
        Some(cancel) => format!(
            "::lk_agents_runtime::host::call_cancellable(&crate::py_calls::Entry, \"{label}\", {scope}, &[{requires}], {}, {host_call})",
            cancel.name
        ),
        None => format!(
            "::lk_agents_runtime::host::call(&crate::py_calls::Entry, \"{label}\", {scope}, &[{requires}], {host_call})"
        ),
    };
    let head = format!("{host}{moved}{context}        let started = {dispatch};\n");
    match m.kind {
        ProxyCall::Await if m.ret.err.is_none() => Ok(awaited(
            m,
            &sig,
            "()",
            &head,
            &format!(
                "        if let Err(e) = started.await {{\n            log::warn!(\"{label} failed: {{e}}\");\n        }}\n"
            ),
        )),
        ProxyCall::Await => {
            let (ret, refused, wrap, err) = result_shape(m, &at)?;
            let binding = if m.ret.ok == Ty::Unit { "()" } else { "out" };
            let read = returned_value("o", &m.ret.ok, &label, &at)?;
            let unpack = if context.is_empty() { String::new() } else { stream_context(&err) };
            Ok(awaited(
                m,
                &sig,
                &ret,
                &head,
                &format!(
                    "        let out = started.await;\n        ::pyo3::Python::attach(|py| {{\n            let out = match out {{\n                Ok(out) => out,\n                Err(e) => {{\n{refused}                    return Err(<{err} as crate::py_errors::FromHostErr>::from_host(py, &e));\n                }}\n            }};\n{unpack}            let o = out.bind(py);\n            let {binding} = ({read}).map_err(|e| <{err} as crate::py_errors::FromHostErr>::from_host(py, &e))?;\n            Ok({wrap})\n        }})\n"
                ),
            ))
        }
        ProxyCall::ReadOnce | ProxyCall::Bind | ProxyCall::Call | ProxyCall::Notify => {
            unreachable!("answered above")
        }
    }
}

/// `text`, once its host call (if any) dispatches through the runtime's HostInvoker and nothing
/// else in it calls into the host.
fn dispatched(p: &Proxy, m: &ProxyMethod, text: String) -> Result<String, String> {
    let calls_host = matches!(m.kind, ProxyCall::Await | ProxyCall::Call | ProxyCall::Notify);
    let bypass = ["call_method", "call_in(", "call_soon_threadsafe", "into_future", ".call1("]
        .into_iter()
        .find(|api| text.contains(api));
    match (calls_host && !text.contains("::lk_agents_runtime::host::"), bypass) {
        (false, None) => Ok(text),
        (true, _) => Err(format!(
            "{}::{}: a proxy's host call dispatches through `lk_agents_runtime::host`",
            p.trait_path, m.rust_name
        )),
        (false, Some(api)) => Err(format!(
            "{}::{}: `{api}` calls into the host outside HostInvoker",
            p.trait_path, m.rust_name
        )),
    }
}

/// An awaited call: an `async fn` under `#[async_trait]`, else a `fn` returning the boxed future
/// of `body`, whose owned values (`head`) are taken before it so it borrows nothing of the call.
fn awaited(m: &ProxyMethod, sig: &str, ret: &str, head: &str, body: &str) -> String {
    match &m.boxed {
        None if ret == "()" => format!("    async {sig} {{\n{head}{body}    }}\n"),
        None => format!("    async {sig} -> {ret} {{\n{head}{body}    }}\n"),
        Some(lifetime) => format!(
            "    {sig} -> ::lk_agents_abi::BoxFuture<{lifetime}, {ret}> {{\n{head}        ::std::boxed::Box::pin(async move {{\n{body}        }})\n    }}\n"
        ),
    }
}

/// An awaited or called method's return type, the refusal check run on a host failure, the
/// success value and the declared error.
fn result_shape(
    m: &ProxyMethod,
    at: &str,
) -> Result<(String, String, &'static str, String), String> {
    let err = m.ret.err.as_ref().ok_or_else(|| format!("{at}: the call declares its error"))?;
    let ok = m.ret.ok.rust();
    Ok(match &m.ret.refusal {
        None => (
            format!("::std::result::Result<{ok}, {err}>"),
            String::new(),
            if m.ret.ok == Ty::Unit { "()" } else { "out" },
            err.clone(),
        ),
        Some(r) => (
            format!("::std::result::Result<::std::result::Result<{ok}, {}>, {err}>", r.rust),
            format!(
                "                    if e.is_instance(py, &py.import({:?}).and_then(|m| m.getattr({:?})).map_err(|e| <{err} as crate::py_errors::FromHostErr>::from_host(py, &e))?) {{\n                        return Ok(Err(<{} as ::std::convert::From<String>>::from(e.value(py).to_string())));\n                    }}\n",
                r.module, r.class, r.rust
            ),
            "Ok(out)",
            err.clone(),
        ),
    })
}

/// The trait's parameter type.
fn param_type(q: &Param) -> String {
    match (&q.ty, q.by_ref) {
        (Ty::Str, true) => "&str".into(),
        (Ty::List(t), true) => format!("&[{}]", t.rust()),
        (Ty::Optional(t), true) if **t == Ty::Str => "Option<&str>".into(),
        (Ty::Optional(t), true) => format!("Option<&{}>", t.rust()),
        (t, true) => format!("&{}", t.rust()),
        (t, false) => t.rust(),
    }
}

/// The parameter as an owned value.
fn owned(q: &Param) -> String {
    match (&q.ty, q.by_ref) {
        (Ty::Str, true) => format!("{}.to_owned()", q.name),
        (Ty::List(_), true) => format!("{}.to_vec()", q.name),
        (Ty::Optional(_), true) => format!("{}.map(::std::borrow::ToOwned::to_owned)", q.name),
        (_, true) => format!("::std::clone::Clone::clone({})", q.name),
        (_, false) => q.name.clone(),
    }
}

/// `v` (owned) as a host object, a `PyResult<Bound<PyAny>>` unwrapped with `?`.
fn to_host(v: &str, ty: &Ty, at: &str) -> Result<String, String> {
    Ok(match ty {
        Ty::Json => format!("crate::py_values::value_to_py(py, &{v})?"),
        Ty::JsonObject => format!("crate::py_values::extra_to_py(py, &{v})?.into_any()"),
        // The host object a declared proxy was taken from.
        Ty::Proxy { declared: Some((family, name)), boxed: false, .. } => {
            format!("super::{family}::{name}Proxy::to_host(py, ::std::sync::Arc::clone(&{v}))?")
        }
        Ty::Optional(t) if matches!(**t, Ty::Proxy { .. }) => format!(
            "match &{v} {{ None => py.None().into_bound(py), Some(x) => {} }}",
            to_host("x", t, at)?
        ),
        // The failure an observer records: the host's own exception while it is retained.
        Ty::Optional(t) if matches!(**t, Ty::Error(_)) => format!(
            "match {v} {{ None => py.None().into_bound(py), Some(x) => crate::py_errors::ExceptionValue::to_exception(x, py)? }}"
        ),
        Ty::Error(_) => format!("crate::py_errors::ExceptionValue::to_exception({v}, py)?"),
        // A native implementation, handed over as the declared class it is registered as.
        Ty::Native { rust, .. } => {
            format!("<dyn {rust} as super::NativeToHost>::to_host(py, {v})?")
        }
        // A core stream, handed over as its generated stream class (this family's).
        Ty::Source { class, .. } => format!(
            "::pyo3::Bound::new(py, {class}(::lk_agents_runtime::net::source::SharedStream::new({v}, ::lk_agents_runtime::process::host_clock())))?.into_any()",
            class = crate::adapter::class_ident(class),
        ),
        Ty::Unit
        | Ty::This
        | Ty::Proxy { .. }
        | Ty::Sink { .. }
        | Ty::Duplex { .. }
        | Ty::HostSource { .. }
        | Ty::HostDuplex { .. }
        | Ty::Handle { .. }
        | Ty::Instance { .. } => {
            return Err(format!("{at}: `{}` is not passed to the host", ty.rust()));
        }
        Ty::Shared { class, family, value: false, .. } => {
            format!(
                "::pyo3::Bound::new(py, crate::generated::{family}::{class}({v}))?.into_any()",
                class = crate::adapter::class_ident(class)
            )
        }
        Ty::Shared { class, family, value: true, .. } => format!(
            "::pyo3::Bound::new(py, crate::generated::{family}::{class}(::std::sync::Arc::new({v})))?.into_any()"
        ),
        // Seconds, as the host's timeouts and delays are.
        Ty::Duration => {
            format!("::pyo3::IntoPyObjectExt::into_bound_py_any({v}.as_secs_f64(), py)?")
        }
        Ty::Optional(t) if **t == Ty::Duration => {
            format!("::pyo3::IntoPyObjectExt::into_bound_py_any({v}.map(|d| d.as_secs_f64()), py)?")
        }
        Ty::Optional(t) | Ty::List(t) if matches!(**t, Ty::Json | Ty::JsonObject) => {
            return Err(format!("{at}: `{}` has no host conversion yet", ty.rust()));
        }
        Ty::List(t) if matches!(**t, Ty::Error(_)) => {
            return Err(format!("{at}: `{}` has no host conversion yet", ty.rust()));
        }
        ty if crate::records::holds_record(ty) => crate::records::emit(ty, v)?,
        _ => format!("::pyo3::IntoPyObjectExt::into_bound_py_any({v}, py)?"),
    })
}

/// Recursively extract returned capabilities using the returning operation's captured context.
fn returned_value(o: &str, ty: &Ty, label: &str, at: &str) -> Result<String, String> {
    if !ty.holds_proxy() {
        return from_host(o, ty, label, at);
    }
    Ok(match ty {
        Ty::Proxy { declared: Some((family, name)), boxed: false, .. } => {
            format!("super::{family}::{name}Proxy::in_context({o}.as_any(), context)")
        }
        Ty::Proxy { declared: Some((family, name)), boxed: true, .. } => {
            format!("super::{family}::{name}Proxy::owned_in_context({o}.as_any(), context)")
        }
        Ty::Optional(inner) => format!(
            "if {o}.is_none() {{ ::pyo3::PyResult::Ok(None) }} else {{ ({}).map(Some) }}",
            returned_value(o, inner, label, at)?
        ),
        Ty::List(inner) => format!(
            "{o}.try_iter().and_then(|items| items.map(|item| {{ let item = item?; {} }}).collect::<::pyo3::PyResult<Vec<_>>>())",
            returned_value("item", inner, label, at)?
        ),
        Ty::Tuple(items) => {
            let mut reads = String::new();
            let mut fields = Vec::new();
            for (i, item) in items.iter().enumerate() {
                let name = format!("item_{i}");
                reads.push_str(&format!(" let {name} = tuple.get_item({i})?;"));
                fields.push(format!("({})?", returned_value(&name, item, label, at)?));
            }
            format!(
                "(|| -> ::pyo3::PyResult<_> {{ let tuple = {o}.cast::<::pyo3::types::PyTuple>()?; if tuple.len() != {len} {{ return Err(::pyo3::exceptions::PyTypeError::new_err(\"{label}: expected a tuple of {len} items\")); }}{reads} Ok(({},)) }})()",
                fields.join(", "),
                len = items.len(),
            )
        }
        Ty::Map(key, value) => format!(
            "{o}.cast::<::pyo3::types::PyDict>().and_then(|items| items.iter().map(|(key, value)| Ok((({})?, ({})?))).collect::<::pyo3::PyResult<_>>())",
            returned_value("key", key, label, at)?,
            returned_value("value", value, label, at)?,
        ),
        _ => return Err(format!("{at}: unsupported returned capability container")),
    })
}

/// `PyResult<T>` of the host's value `o: &Bound<PyAny>`.
fn from_host(o: &str, ty: &Ty, label: &str, at: &str) -> Result<String, String> {
    match ty {
        Ty::Unit => Ok(format!("{{ let _ = {o}; ::pyo3::PyResult::Ok(()) }}")),
        Ty::HostSource { item, .. } => {
            let read = from_object("o", item, &format!("{label:?}"), at)?;
            // The stream pulls in a copy of the context its call ran in.
            Ok(format!(
                "crate::py_streams::HostStream::open(py, {o}, context, |o| {read}, ::lk_agents_runtime::process::host_clock()).map(|s| ::std::boxed::Box::new(s) as {})",
                ty.rust()
            ))
        }
        Ty::HostDuplex { input, output, provider } => {
            let read = from_object("o", output, &format!("{label:?}"), at)?;
            let write = match input.as_ref() {
                Ty::Record(Record { host: HostForm::Record { module, name }, .. }) => {
                    format!("{}_to_py", crate::records::conversion(module, name))
                }
                Ty::Record(record @ Record { host: HostForm::Converted(..), .. }) => {
                    format!("{}_to_py", crate::records::converted(record))
                }
                _ => format!("|py, v| {}", crate::records::as_result(&to_host("v", input, at)?)),
            };
            Ok(format!(
                "crate::py_streams::HostStream::duplex(py, {o}, context, {write}, |o| {read}, ::lk_agents_runtime::process::host_clock()).map(|stream| {{ let (sink, source) = stream.split(); ::lk_agents_abi::DuplexStream::new(::std::boxed::Box::new(sink) as ::std::boxed::Box<dyn ::lk_agents_abi::StreamInput<{}>>, ::std::boxed::Box::new(source) as ::std::boxed::Box<dyn ::lk_agents_abi::{}<{}>>) }})",
                input.rust(),
                if *provider { "ProviderStream" } else { "StreamOutput" },
                output.rust()
            ))
        }
        // `None` for no stream.
        Ty::Optional(t) if matches!(**t, Ty::HostSource { .. } | Ty::HostDuplex { .. }) => {
            Ok(format!(
                "if {o}.is_none() {{ ::pyo3::PyResult::Ok(None) }} else {{ ({}).map(Some) }}",
                from_host(o, t, label, at)?
            ))
        }
        _ => from_object(o, ty, &format!("{label:?}"), at),
    }
}

#[cfg(test)]
mod tests {
    use crate::decl::tests::{proxy, read};

    #[test]
    fn cancellable_proxy_operations_settle_without_passing_the_signal_to_python() {
        let source = proxy(
            "#[host(await)] async fn setup(&self, #[host(cancel)] cancel: std::sync::Arc<dyn crate::invoke::Cancellation>) -> Result<(), HostError>;",
        );
        let decls = read(&source).expect("native cancellation resolves");
        let p = &decls.proxies[0];
        let emitted = super::method(p, &p.methods[0]).expect("emits");
        assert!(emitted.contains("::host::call_cancellable("));
        assert!(emitted.contains("&self.context, &[], cancel,"));
        assert!(emitted.contains("PyAny>; 0] = []"));
        for invalid in [
            "#[host(call)] fn setup(&self, #[host(cancel)] cancel: std::sync::Arc<dyn crate::invoke::Cancellation>) -> Result<(), HostError>;",
            "#[host(await)] async fn setup(&self, #[host(cancel)] cancel: String) -> Result<(), HostError>;",
            "#[host(await)] async fn setup(&self, #[host(cancel)] a: std::sync::Arc<dyn crate::invoke::Cancellation>, #[host(cancel)] b: std::sync::Arc<dyn crate::invoke::Cancellation>) -> Result<(), HostError>;",
        ] {
            assert!(read(&proxy(invalid)).is_err());
        }
    }

    #[test]
    fn returned_capabilities_inherit_context_inside_containers() {
        for call in ["#[host(await)] async", "#[host(call)]"] {
            let source = proxy(&format!(
                "{call} fn acquire(&self) -> Result<(Option<std::sync::Arc<dyn Callbacks>>, Vec<std::sync::Arc<dyn Callbacks>>), HostError>; #[host(await)] async fn ping(&self) -> Result<(), HostError>;",
            ));
            let decls = read(&source).expect("nested capabilities resolve");
            let p = &decls.proxies[0];
            let emitted = super::method(p, &p.methods[0]).expect("emits");
            assert_eq!(emitted.matches("CallbacksProxy::in_context").count(), 2);
            assert!(!emitted.contains("CallbacksProxy::from_host"));
            assert!(emitted.contains("host_context(&self.context).cloned()"));
        }
    }

    #[test]
    fn factory_context_follows_async_capabilities_but_pure_factories_need_no_loop() {
        let mut source = proxy("#[host(await)] async fn ping(&self) -> Result<(), HostError>;");
        source.push_str(r#"
            #[crate::host(proxy(host = "livekit.agents.Upstream"))]
            pub trait Upstream: Send + Sync {
                #[host(call)] fn factory(&self) -> Result<Option<std::sync::Arc<dyn Factory>>, HostError>;
            }
            #[crate::host(proxy(host = "livekit.agents.Factory"))]
            pub trait Factory: Send + Sync {
                #[host(call)] fn create(&self) -> Result<Vec<std::sync::Arc<dyn Callbacks>>, HostError>;
            }
            #[crate::host(proxy(host = "livekit.agents.Cpu"))]
            pub trait Cpu: Send + Sync {
                #[host(call)] fn compute(&self) -> Result<u64, HostError>;
            }
            #[crate::host(proxy(host = "livekit.agents.PureFactory"))]
            pub trait PureFactory: Send + Sync {
                #[host(call)] fn create(&self) -> Result<std::sync::Arc<dyn Cpu>, HostError>;
            }
        "#);
        let decls = read(&source).expect("factories resolve");
        for name in ["Callbacks", "Factory", "Upstream"] {
            let p = decls.proxies.iter().find(|p| p.rust_name == name).unwrap();
            assert!(p.captures_context(), "{name} must carry the async operation's context");
            let generated = super::proxy(p).expect("factory emits");
            assert!(generated.contains("context: ::lk_agents_abi::utils::host_ref::HostRef"));
        }
        for name in ["Cpu", "PureFactory"] {
            let p = decls.proxies.iter().find(|p| p.rust_name == name).unwrap();
            assert!(!p.captures_context(), "{name} must construct without a running loop");
            let generated = super::proxy(p).expect("pure factory emits");
            assert!(!generated.contains("host_context(&self.context)"));
        }
    }

    // Catches: an emitted host call that bypasses HostInvoker (calling the host object in place,
    // or entering a context itself) generated instead of refused.
    #[test]
    fn a_host_call_outside_host_invoker_is_refused() {
        let decls =
            read(&proxy("#[host(call)] fn state(&self, id: &str) -> Result<String, HostError>;"))
                .expect("resolves");
        let (p, m) = (&decls.proxies[0], &decls.proxies[0].methods[0]);
        assert!(super::dispatched(p, m, super::method(p, m).expect("emits")).is_ok());
        let direct =
            "    fn state(&self) { self.host.bind(py).call_method1(\"state\", ()); }".to_owned();
        let error = super::dispatched(p, m, direct).expect_err("refused");
        assert!(error.contains("dispatches through `lk_agents_runtime::host`"), "{error}");
        let bypass =
            "::lk_agents_runtime::host::call_now(..); crate::py_calls::call_in(c, f)".to_owned();
        let error = super::dispatched(p, m, bypass).expect_err("refused");
        assert!(error.contains("`call_in(` calls into the host outside HostInvoker"), "{error}");
    }
}
