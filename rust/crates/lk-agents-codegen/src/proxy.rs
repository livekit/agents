//! The declared proxies (`#[host(proxy)]` traits whose methods declare their calls), written into
//! their subject's adapter module: `<Trait>Proxy` implements the ABI trait over a host object.
//!
//! Each call enters a copy of a captured host context (`lk_agents_runtime::host::python::call_in`):
//! the one its `#[host(context)]` parameter carries, else the one the proxy captured when it was
//! taken from the host. Every parameter crosses by its declared kind and the result is read by its
//! declared kind; a host failure becomes the declared error. `read_once` values are read when the
//! proxy is taken; a `bind` method returns a copy whose awaited calls pass the bound value last,
//! handed out through its declared native wrapper when it names one.
//!
//! The ABI's `HostProxy for dyn Trait` (the in-place exports' conversion) is written into the ABI
//! crate by [`legacy_bridge`]; it resolves to this proxy through the `DeclaredProxy` registration
//! the adapter submits.

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
            "        let view = obj.py().import({module:?})?.getattr({class:?})?.call1((obj,))?;\n        let obj = &view;\n"
        ),
        None => String::new(),
    };
    if p.captures_context() {
        fields.push(
            "    /// The host's running loop and a copy of its context, captured when it was taken.\n    context: ::lk_agents_abi::utils::host_ref::HostRef,\n".into(),
        );
        inits.push("context: ::lk_agents_runtime::host::python::capture_context(obj.py())?".into());
    }
    if p.methods.iter().any(|m| m.kind == ProxyCall::Notify) {
        fields.push(
            "    /// The `notify` calls, delivered in order.\n    notifier: ::lk_agents_runtime::host::python::Notifier,\n".into(),
        );
        inits.push("notifier: ::lk_agents_runtime::host::python::Notifier::new(::lk_agents_runtime::host::python::capture_context(obj.py())?)".into());
    }
    for m in p.methods.iter().filter(|m| m.kind == ProxyCall::ReadOnce) {
        let at = format!("{}::{}", tr, m.rust_name);
        fields.push(format!("    {}: {},\n", m.rust_name, m.ret.ok.rust()));
        let read = from_object("v", &m.ret.ok, &format!("\"{}.{}\"", p.rust_name, m.name), &at)?;
        reads.push_str(&format!(
            "        let {} = {{\n            let v = obj.call_method0({:?})?;\n            {read}?\n        }};\n",
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
        "/// [`{tr}`] over a host `{host}`.\n#[derive(Clone)]\npub struct {name} {{\n{}}}\n\nimpl {name} {{\n    /// A registered native class as itself, any other host object behind this proxy.\n    pub fn from_host(obj: &Bound<'_, PyAny>) -> PyResult<::std::sync::Arc<dyn {tr}>> {{\n        if let Some(native) = ::lk_agents_abi::python::native::<dyn {tr}>(obj) {{\n            return Ok(native);\n        }}\n        Ok(::std::sync::Arc::new(Self::take(obj)?))\n    }}\n\n    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.\n    pub fn owned(obj: &Bound<'_, PyAny>) -> PyResult<::std::boxed::Box<dyn {tr}>> {{\n        Ok(::std::boxed::Box::new(Self::take(obj)?))\n    }}\n\n    fn take(obj: &Bound<'_, PyAny>) -> PyResult<Self> {{\n{reads}        Ok({name} {{ {} }})\n    }}\n}}\n\n",
        fields.concat(),
        inits.join(", "),
        host = p.host,
    ));
    s.push_str(&format!(
        "::lk_agents_abi::python::inventory::submit! {{\n    ::lk_agents_abi::python::DeclaredProxy::of::<dyn {tr}>(|obj| {name}::from_host(obj).map(|p| ::std::boxed::Box::new(p) as ::std::boxed::Box<dyn ::std::any::Any>))\n}}\n\n"
    ));
    if p.async_trait {
        s.push_str("#[::async_trait::async_trait]\n");
    }
    s.push_str(&format!("impl {tr} for {name} {{\n"));
    let methods = p.methods.iter().map(|m| method(p, m)).collect::<Result<Vec<_>, _>>()?;
    s.push_str(&methods.join("\n"));
    s.push_str("}\n");
    Ok(s)
}

/// `impl HostProxy for dyn Trait` in the ABI crate: the in-place exports take a declared proxy
/// through it, registered by the adapter that generates it.
pub fn legacy_bridge(p: &Proxy) -> String {
    let tr = &p.trait_path;
    format!(
        "impl crate::python::HostProxy for dyn {tr} {{\n    fn from_host(obj: &::pyo3::Bound<'_, ::pyo3::PyAny>, _clock: ::std::option::Option<::std::sync::Arc<dyn crate::Clock>>) -> ::pyo3::PyResult<::std::sync::Arc<Self>> {{\n        crate::python::declared_proxy::<dyn {tr}>(obj)\n    }}\n\n    fn view(handle: crate::python::HostHandle, _clock: ::std::option::Option<::std::sync::Arc<dyn crate::Clock>>) -> ::pyo3::PyResult<::std::boxed::Box<Self>> {{\n        let _ = handle;\n        Err(::pyo3::exceptions::PyTypeError::new_err(\"`{}` is a declared proxy: it is taken from the host object, not a view\"))\n    }}\n}}\n",
        p.rust_name
    )
}

fn method(p: &Proxy, m: &ProxyMethod) -> Result<String, String> {
    let at = format!("{}::{}", p.trait_path, m.rust_name);
    let label = format!("{}.{}", p.rust_name, m.name);
    let params: Vec<String> =
        m.params.iter().map(|q| format!("{}: {}", q.name, param_type(q))).collect();
    let receiver = if m.consumes { "self: ::std::boxed::Box<Self>" } else { "&self" };
    let sig = format!(
        "fn {}({})",
        m.rust_name,
        std::iter::once(receiver.to_owned()).chain(params).collect::<Vec<_>>().join(", ")
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
    for q in &m.params {
        moved.push_str(&format!("        let {}_arg = {};\n", q.name, owned(q)));
        args.push(to_host(&format!("{}_arg", q.name), &q.ty, &at)?);
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
    let call = format!(
        "::lk_agents_runtime::host::python::call_in(context, move |py| {{\n                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; {}] = [{}];\n{bound}                Ok((host.bind(py).getattr({:?})?, ::pyo3::types::PyTuple::new(py, args)?))\n            }})",
        args.len(),
        args.join(", "),
        m.name
    );
    let no_context = format!(
        "::pyo3::exceptions::PyRuntimeError::new_err(\"{label}: the call captured no host context\")"
    );
    let missing = match m.kind {
        ProxyCall::Notify => format!(
            "::lk_agents_abi::python::log::warn!(\"{label} failed: {{}}\", {no_context});\n            return;"
        ),
        _ => format!(
            "return Err(::pyo3::Python::attach(|py| <{} as ::lk_agents_abi::python::FromHostErr>::from_host(py, &{no_context})));",
            m.ret.err.as_deref().unwrap_or_default()
        ),
    };
    if m.kind == ProxyCall::Call {
        let (ret, refused, wrap, err) = result_shape(m, &at)?;
        let read = from_host("o", &m.ret.ok, &label, &at)?;
        return Ok(format!(
            "    {sig} -> {ret} {{\n{moved}        ::pyo3::Python::attach(|py| {{\n            // One place converts the call's failures to its declared error.\n            #[allow(clippy::redundant_closure_call)]\n            let out = (|| {{\n                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; {}] = [{}];\n                self.host.bind(py).call_method1({:?}, ::pyo3::types::PyTuple::new(py, args)?)\n            }})();\n            let out = match out {{\n                Ok(out) => out,\n                Err(e) => {{\n{refused}                    return Err(<{err} as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));\n                }}\n            }};\n            let o = &out;\n            let out = ({read}).map_err(|e| <{err} as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;\n            Ok({wrap})\n        }})\n    }}\n",
            args.len(),
            args.join(", "),
            m.name
        ));
    }
    // The call's own context, else the one the proxy captured.
    let context = match m.context {
        Some(i) => format!(
            "        let Some(context) = ::lk_agents_abi::agent::HostContext::host_context({}).cloned() else {{\n            {missing}\n        }};\n",
            m.params[i].name
        ),
        None => "        let context = ::std::clone::Clone::clone(&self.context);\n".into(),
    };
    let head = format!("        let host = ::std::sync::Arc::clone(&self.host);\n{moved}{context}");
    match m.kind {
        ProxyCall::Notify => Ok(format!(
            "    {sig} {{\n{head}        self.notifier.{}(context, \"{label}\", move |py| {{\n            let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; {}] = [{}];\n            Ok((host.bind(py).getattr({:?})?, ::pyo3::types::PyTuple::new(py, args)?))\n        }});\n    }}\n",
            if m.lossy { "notify_lossy" } else { "notify" },
            args.len(),
            args.join(", "),
            m.name
        )),
        ProxyCall::Await if m.ret.err.is_none() => Ok(format!(
            "    async {sig} {{\n{head}        let context = &context;\n        if let Err(e) = {call}.await {{\n            ::lk_agents_abi::python::log::warn!(\"{label} failed: {{e}}\");\n        }}\n    }}\n",
        )),
        ProxyCall::Await => {
            let (ret, refused, wrap, err) = result_shape(m, &at)?;
            let read = from_host("o", &m.ret.ok, &label, &at)?;
            Ok(format!(
                "    async {sig} -> {ret} {{\n{head}        let context = &context;\n        let out = {call}.await;\n        ::pyo3::Python::attach(|py| {{\n            let out = match out {{\n                Ok(out) => out,\n                Err(e) => {{\n{refused}                    return Err(<{err} as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));\n                }}\n            }};\n            let o = out.bind(py);\n            let out = ({read}).map_err(|e| <{err} as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;\n            Ok({wrap})\n        }})\n    }}\n",
            ))
        }
        ProxyCall::ReadOnce | ProxyCall::Bind | ProxyCall::Call => unreachable!("answered above"),
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
        None => (format!("::std::result::Result<{ok}, {err}>"), String::new(), "out", err.clone()),
        Some(r) => (
            format!("::std::result::Result<::std::result::Result<{ok}, {}>, {err}>", r.rust),
            format!(
                "                    if e.is_instance(py, &py.import({:?}).and_then(|m| m.getattr({:?})).map_err(|e| <{err} as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?) {{\n                        return Ok(Err(<{} as ::std::convert::From<String>>::from(e.value(py).to_string())));\n                    }}\n",
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
        Ty::Json => format!("::lk_agents_abi::python::value_to_py(py, &{v})?"),
        Ty::JsonObject => format!("::lk_agents_abi::python::extra_to_py(py, &{v})?.into_any()"),
        Ty::Error(e) => format!("({})({v}).into_value(py).into_bound(py).into_any()", e.with),
        // A host class the framework does not generate: built from the record's fields.
        Ty::Record(Record { host: HostForm::External { module, name }, .. }) => format!(
            "py.import({module:?})?.getattr({name:?})?.call((), Some(&::pyo3::IntoPyObject::into_pyobject({v}, py)?.cast_into::<::pyo3::types::PyDict>()?))?"
        ),
        Ty::Unit
        | Ty::This
        | Ty::Proxy { .. }
        | Ty::Source { .. }
        | Ty::HostSource(_)
        | Ty::HostDuplex { .. }
        | Ty::Handle { .. }
        | Ty::Instance { .. } => {
            return Err(format!("{at}: `{}` is not passed to the host", ty.rust()));
        }
        Ty::Shared { rust, .. } => {
            format!("<{rust} as ::lk_agents_abi::python::ArcClass>::wrap(py, {v})?")
        }
        // Seconds, as the host's timeouts and delays are.
        Ty::Duration => {
            format!("::pyo3::IntoPyObjectExt::into_bound_py_any({v}.as_secs_f64(), py)?")
        }
        Ty::Optional(t) if **t == Ty::Duration => {
            format!("::pyo3::IntoPyObjectExt::into_bound_py_any({v}.map(|d| d.as_secs_f64()), py)?")
        }
        Ty::Optional(t) | Ty::List(t)
            if matches!(**t, Ty::Json | Ty::JsonObject | Ty::Error(_)) =>
        {
            return Err(format!("{at}: `{}` has no host conversion yet", ty.rust()));
        }
        _ => format!("::pyo3::IntoPyObjectExt::into_bound_py_any({v}, py)?"),
    })
}

/// `PyResult<T>` of the host's value `o: &Bound<PyAny>`.
fn from_host(o: &str, ty: &Ty, label: &str, at: &str) -> Result<String, String> {
    match ty {
        Ty::Unit => Ok(format!("{{ let _ = {o}; ::pyo3::PyResult::Ok(()) }}")),
        Ty::HostSource(item) => {
            let read = from_object("o", item, &format!("{label:?}"), at)?;
            // The stream pulls in a copy of the context its call ran in.
            Ok(format!(
                "::lk_agents_runtime::host::stream::HostStream::open(py, {o}, context, |o| Ok({read}?), ::lk_agents_runtime::process::host_clock()).map(|s| ::std::boxed::Box::new(s) as {})",
                ty.rust()
            ))
        }
        Ty::HostDuplex { input, output } => {
            let read = from_object("o", output, &format!("{label:?}"), at)?;
            let write = to_host("v", input, at)?;
            Ok(format!(
                "::lk_agents_runtime::host::stream::HostStream::duplex(py, {o}, context, |py, v| Ok({write}), |o| Ok({read}?), ::lk_agents_runtime::process::host_clock()).map(|(sink, source)| (::std::boxed::Box::new(sink) as ::std::boxed::Box<dyn ::lk_agents_abi::StreamSink<{}>>, ::std::boxed::Box::new(source) as ::std::boxed::Box<dyn ::lk_agents_abi::StreamSource<{}>>))",
                input.rust(),
                output.rust()
            ))
        }
        // `None` for no stream.
        Ty::Optional(t) if matches!(**t, Ty::HostSource(_) | Ty::HostDuplex { .. }) => Ok(format!(
            "if {o}.is_none() {{ ::pyo3::PyResult::Ok(None) }} else {{ ({}).map(Some) }}",
            from_host(o, t, label, at)?
        )),
        _ => from_object(o, ty, &format!("{label:?}"), at),
    }
}
