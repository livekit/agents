//! Host-value emission, shared by record fields, class results and proxy arguments.

use super::conversion;
use crate::decl::{HostForm, Record, Ty};

/// A `Bound<'py, PyAny>` expression of the owned value `x` (fallible steps use `?`, `py` in scope):
/// what a declared record field, an adapter's result or a proxy's argument crosses as.
pub(crate) fn emit(ty: &Ty, x: &str) -> Result<String, String> {
    Ok(match ty {
        Ty::Record(Record { host: HostForm::WireStrs, .. }) => {
            emit(&Ty::Str, &format!("{x}.as_str()"))?
        }
        Ty::Record(Record { host: HostForm::Record { module, name }, .. }) => {
            format!("{}_to_py(py, {x})?", conversion(module, name))
        }
        Ty::Record(r @ Record { host: HostForm::Converted(..), .. }) => {
            format!("{}_to_py(py, {x})?", super::converted(r))
        }
        Ty::Record(Record { host: HostForm::Via { host, via, attribute }, .. }) => {
            let emitted = emit(via, "scalar")?;
            let emitted = if attribute.is_some() {
                let (module, name) =
                    host.rsplit_once('.').ok_or("attribute representation needs a host class")?;
                format!(
                    "crate::py_calls::call(&py.import({module:?})?.getattr({name:?})?, ({emitted},), None)?"
                )
            } else {
                emitted
            };
            format!("{{ let scalar: {} = {x}.into(); {emitted} }}", via.rust())
        }
        Ty::Record(Record { host: HostForm::Str | HostForm::Wire, .. }) => {
            format!("crate::py_values::str_to_py(py, &{x})")
        }
        Ty::Tuple(ts) => {
            let names: Vec<String> = (0..ts.len()).map(|i| format!("x{i}")).collect();
            let items =
                ts.iter().zip(&names).map(|(t, n)| emit(t, n)).collect::<Result<Vec<_>, _>>()?;
            format!(
                "{{ let ({},) = {x}; ::pyo3::types::PyTuple::new(py, [{}])?.into_any() }}",
                names.join(", "),
                items.join(", ")
            )
        }
        Ty::Boxed(t) => format!("{{ let value = *{x}; {} }}", emit(t, "value")?),
        Ty::Json => format!("crate::py_values::value_to_py(py, &{x})?"),
        Ty::JsonObject => format!("crate::py_values::extra_to_py(py, &{x})?.into_any()"),
        Ty::Optional(t) => format!(
            "match {x} {{ None => py.None().into_bound(py), Some(x) => {} }}",
            emit(t, "x")?
        ),
        Ty::List(t) if **t == Ty::Int(crate::decl::Int::U8) => {
            format!("::pyo3::types::PyBytes::new(py, &{x}).into_any()")
        }
        Ty::List(t) => format!(
            "::pyo3::types::PyList::new(py, {x}.into_iter().map(|x| {}).collect::<PyResult<Vec<_>>>()?)?.into_any()",
            as_result(&emit(t, "x")?)
        ),
        Ty::Map(k, t) | Ty::OrderedMap(k, t) => format!(
            "{{ let d = ::pyo3::types::PyDict::new(py); for (k, x) in {x} {{ d.set_item({}, {})?; }} d.into_any() }}",
            emit(k, "k")?,
            emit(t, "x")?
        ),
        Ty::Set(t) => format!(
            "::pyo3::types::PyFrozenSet::new(py, {x}.into_iter().map(|x| {}).collect::<PyResult<Vec<_>>>()?)?.into_any()",
            as_result(&emit(t, "x")?)
        ),
        Ty::Bool | Ty::Int(_) | Ty::Float | Ty::Str | Ty::Record(_) => {
            format!("::pyo3::IntoPyObjectExt::into_bound_py_any({x}, py)?")
        }
        Ty::Error(_) => format!("crate::py_errors::ExceptionValue::to_exception({x}, py)?"),
        Ty::Timestamp => {
            format!(
                "::pyo3::IntoPyObjectExt::into_bound_py_any(::lk_agents_abi::unix_secs({x}), py)?"
            )
        }
        Ty::Duration => {
            format!("::pyo3::IntoPyObjectExt::into_bound_py_any({x}.as_secs_f64(), py)?")
        }
        // A native implementation, as the declared class it is.
        Ty::Native { rust, .. } => {
            format!("<dyn {rust} as crate::generated::NativeToHost>::to_host(py, {x})?")
        }
        // The host object a declared proxy was taken from.
        Ty::Proxy { declared: Some((family, name)), boxed: false, .. } => {
            format!("crate::generated::{family}::{name}Proxy::to_host(py, {x})?")
        }
        other => return Err(format!("`{}` is not a record field", other.rust())),
    })
}

/// Return an emitted host value, retaining a fallible call directly when possible.
pub(crate) fn as_result(value: &str) -> String {
    value.strip_suffix('?').map_or_else(|| format!("::pyo3::PyResult::Ok({value})"), str::to_owned)
}
