//! The declared proxies (`#[host(proxy)]` traits whose methods declare their calls), written into
//! their subject's adapter module: `<Trait>Proxy` implements the ABI trait over a host object.
//! Each call enters a copy of the host context its `#[host(context)]` parameter carries
//! (`lk_agents_runtime::host::python::call_in`), passes every parameter by its declared kind and
//! reads the result by its declared kind; a host failure becomes the declared error.

use crate::adapter::from_object;
use crate::decl::{Param, Proxy, ProxyCall, ProxyMethod, Ty};

pub fn proxy(p: &Proxy) -> Result<String, String> {
    let name = format!("{}Proxy", p.rust_name);
    let tr = &p.trait_path;
    let mut s = format!("// Declared in {}.\n\n", p.source);
    s.push_str(&format!(
        "/// [`{tr}`] over a host `{host}`:\n/// each call runs in a copy of the host context its `#[host(context)]` parameter captured.\npub struct {name}(::std::sync::Arc<Py<PyAny>>);\n\nimpl {name} {{\n    /// A registered native class as itself, any other host object behind this proxy.\n    pub fn from_host(obj: &Bound<'_, PyAny>) -> PyResult<::std::sync::Arc<dyn {tr}>> {{\n        if let Some(native) = ::lk_agents_abi::python::native::<dyn {tr}>(obj) {{\n            return Ok(native);\n        }}\n        Ok(::std::sync::Arc::new({name}(::std::sync::Arc::new(obj.clone().unbind()))))\n    }}\n}}\n\n",
        host = p.host
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

fn method(p: &Proxy, m: &ProxyMethod) -> Result<String, String> {
    let at = format!("{}::{}", p.trait_path, m.rust_name);
    let label = format!("{}.{}", p.rust_name, m.name);
    let params: Vec<String> =
        m.params.iter().map(|q| format!("{}: {}", q.name, param_type(q))).collect();
    let ctx = &m.params[m.context].name;
    // Owned copies move into the call, which runs on the host's loop.
    let mut moved = String::new();
    let mut args = Vec::new();
    for q in &m.params {
        moved.push_str(&format!("        let {}_arg = {};\n", q.name, owned(q)));
        args.push(to_host(&format!("{}_arg", q.name), &q.ty, &at)?);
    }
    let call = format!(
        "::lk_agents_runtime::host::python::call_in(context, move |py| {{\n                let args = [{}];\n                Ok((host.bind(py).getattr({:?})?, ::pyo3::types::PyTuple::new(py, args)?))\n            }})",
        args.join(", "),
        m.name
    );
    let no_context = format!(
        "::pyo3::exceptions::PyRuntimeError::new_err(\"{label}: the call captured no host context\")"
    );
    let head = format!(
        "        let host = ::std::sync::Arc::clone(&self.0);\n{moved}        let context = ::lk_agents_abi::agent::HostContext::host_context({ctx});\n"
    );
    match m.kind {
        ProxyCall::Notify => Ok(format!(
            "    fn {}(&self, {}) {{\n{head}        let Some(context) = context.cloned() else {{\n            ::lk_agents_abi::python::log::warn!(\"{label} failed: {{}}\", {no_context});\n            return;\n        }};\n        let runtime = match ::lk_agents_runtime::process::handle() {{\n            Ok(runtime) => runtime,\n            Err(e) => {{\n                ::lk_agents_abi::python::log::warn!(\"{label} failed: {{e}}\");\n                return;\n            }}\n        }};\n        runtime.spawn(async move {{\n            let context = &context;\n            if let Err(e) = {call}.await {{\n                ::lk_agents_abi::python::log::warn!(\"{label} failed: {{e}}\");\n            }}\n        }});\n    }}\n",
            m.rust_name,
            params.join(", "),
        )),
        ProxyCall::Await => {
            let err = m.ret.err.as_ref().ok_or_else(|| format!("{at}: an awaited call declares its error"))?;
            let ok = m.ret.ok.rust();
            let (ret, refused, wrap) = match &m.ret.refusal {
                None => (format!("::std::result::Result<{ok}, {err}>"), String::new(), "out"),
                Some(r) => (
                    format!("::std::result::Result<::std::result::Result<{ok}, {}>, {err}>", r.rust),
                    format!(
                        "                    if e.is_instance(py, &py.import({:?}).and_then(|m| m.getattr({:?})).map_err(|e| <{err} as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?) {{\n                        return Ok(Err(<{} as ::std::convert::From<String>>::from(e.value(py).to_string())));\n                    }}\n",
                        r.module, r.class, r.rust
                    ),
                    "Ok(out)",
                ),
            };
            let read = from_host("o", &m.ret.ok, &label, &at)?;
            Ok(format!(
                "    async fn {}(&self, {}) -> {ret} {{\n{head}        let out = match context {{\n            Some(context) => {call}\n            .await,\n            None => Err({no_context}),\n        }};\n        ::pyo3::Python::attach(|py| {{\n            let out = match out {{\n                Ok(out) => out,\n                Err(e) => {{\n{refused}                    return Err(<{err} as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));\n                }}\n            }};\n            let o = out.bind(py);\n            let out = ({read}).map_err(|e| <{err} as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;\n            Ok({wrap})\n        }})\n    }}\n",
                m.rust_name,
                params.join(", "),
            ))
        }
    }
}

/// The trait's parameter type.
fn param_type(q: &Param) -> String {
    match (&q.ty, q.by_ref) {
        (Ty::Str, true) => "&str".into(),
        (Ty::List(t), true) => format!("&[{}]", t.rust()),
        (t, true) => format!("&{}", t.rust()),
        (t, false) => t.rust(),
    }
}

/// The parameter as an owned value.
fn owned(q: &Param) -> String {
    match (&q.ty, q.by_ref) {
        (Ty::Str, true) => format!("{}.to_owned()", q.name),
        (Ty::List(_), true) => format!("{}.to_vec()", q.name),
        (_, true) => format!("::std::clone::Clone::clone({})", q.name),
        (_, false) => q.name.clone(),
    }
}

/// `v` (owned) as a host object, a `PyResult<Bound<PyAny>>` unwrapped with `?`.
fn to_host(v: &str, ty: &Ty, at: &str) -> Result<String, String> {
    Ok(match ty {
        Ty::Json => format!("::lk_agents_abi::python::value_to_py(py, &{v})?"),
        Ty::JsonObject => format!("::lk_agents_abi::python::extra_to_py(py, &{v})?.into_any()"),
        Ty::Unit | Ty::This | Ty::Proxy { .. } | Ty::Source { .. } => {
            return Err(format!("{at}: `{}` is not passed to the host", ty.rust()));
        }
        Ty::Optional(t) | Ty::List(t) if matches!(**t, Ty::Json | Ty::JsonObject) => {
            return Err(format!("{at}: `{}` has no host conversion yet", ty.rust()));
        }
        _ => format!("::pyo3::IntoPyObjectExt::into_bound_py_any({v}, py)?"),
    })
}

/// `PyResult<T>` of the host's value `o: &Bound<PyAny>`.
fn from_host(o: &str, ty: &Ty, label: &str, at: &str) -> Result<String, String> {
    if *ty == Ty::Unit {
        return Ok(format!("{{ let _ = {o}; ::pyo3::PyResult::Ok(()) }}"));
    }
    from_object(o, ty, &format!("{label:?}"), at)
}
