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
        Ty::Record(Record { host: HostForm::Record { module, name }, in_place: false, .. }) => {
            format!("{}_to_py(py, {x})?", conversion(module, name))
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
        Ty::Json => format!("::lk_agents_abi::python::value_to_py(py, &{x})?"),
        Ty::JsonObject => format!("::lk_agents_abi::python::extra_to_py(py, &{x})?.into_any()"),
        Ty::Optional(t) => format!(
            "match {x} {{ None => py.None().into_bound(py), Some(x) => {} }}",
            emit(t, "x")?
        ),
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
        Ty::Duration => {
            format!("::pyo3::IntoPyObjectExt::into_bound_py_any({x}.as_secs_f64(), py)?")
        }
        other => return Err(format!("`{}` is not a record field", other.rust())),
    })
}
