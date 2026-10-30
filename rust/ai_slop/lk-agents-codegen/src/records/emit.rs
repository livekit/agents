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
        Ty::Record(r @ Record { host: HostForm::Converted(_), .. }) => {
            format!("{}_to_py(py, {x})?", super::converted(r))
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
        Ty::Boxed(t) => emit(t, &format!("(*{x})"))?,
        Ty::Json => format!("::lk_agents_abi::python::value_to_py(py, &{x})?"),
        Ty::JsonObject => format!("::lk_agents_abi::python::extra_to_py(py, &{x})?.into_any()"),
        Ty::Optional(t) => format!(
            "match {x} {{ None => py.None().into_bound(py), Some(x) => {} }}",
            emit(t, "x")?
        ),
        Ty::List(t) if **t == Ty::Int(crate::decl::Int::U8) => {
            format!("::pyo3::types::PyBytes::new(py, &{x}).into_any()")
        }
        Ty::List(t) => format!(
            "::pyo3::types::PyList::new(py, {x}.into_iter().map(|x| PyResult::Ok({})).collect::<PyResult<Vec<_>>>()?)?.into_any()",
            emit(t, "x")?
        ),
        Ty::Map(k, t) | Ty::OrderedMap(k, t) => format!(
            "{{ let d = ::pyo3::types::PyDict::new(py); for (k, x) in {x} {{ d.set_item({}, {})?; }} d.into_any() }}",
            emit(k, "k")?,
            emit(t, "x")?
        ),
        Ty::Set(t) => format!(
            "::pyo3::types::PyFrozenSet::new(py, {x}.into_iter().map(|x| PyResult::Ok({})).collect::<PyResult<Vec<_>>>()?)?.into_any()",
            emit(t, "x")?
        ),
        Ty::Bool | Ty::Int(_) | Ty::Float | Ty::Str | Ty::Record(_) => {
            format!("::pyo3::IntoPyObjectExt::into_bound_py_any({x}, py)?")
        }
        Ty::Error(_) => format!("::lk_agents_abi::python::ExceptionValue::to_exception({x}, py)?"),
        Ty::Timestamp => {
            format!("::pyo3::IntoPyObjectExt::into_bound_py_any(::lk_agents_abi::unix_secs({x}), py)?")
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
