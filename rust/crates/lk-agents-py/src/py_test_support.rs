//! Loads ordinary Python fixtures for native boundary tests, preserving real traceback paths.

#[cfg(test)]
pub use support::*;

#[cfg(test)]
mod support {
    use pyo3::prelude::*;
    use pyo3::sync::PyOnceLock;
    use std::ffi::CString;

    pub fn module(py: Python<'_>) -> Bound<'_, PyModule> {
        static MODULE: PyOnceLock<Py<PyModule>> = PyOnceLock::new();
        MODULE
            .get_or_init(py, || {
                PyModule::from_code(
                    py,
                    &CString::new(include_str!("py_test_fixtures.py")).unwrap(),
                    &CString::new(concat!(env!("CARGO_MANIFEST_DIR"), "/src/py_test_fixtures.py"))
                        .unwrap(),
                    c"_lk_agents_py_test_fixtures",
                )
                .unwrap()
                .unbind()
            })
            .bind(py)
            .clone()
    }

    pub fn call<'py, A: pyo3::call::PyCallArgs<'py>>(
        py: Python<'py>,
        name: &str,
        args: A,
    ) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::call(&module(py).getattr(name)?, args, None)
    }

    pub fn fixture<'py>(py: Python<'py>, name: &str) -> Bound<'py, PyAny> {
        call(py, name, ()).unwrap()
    }

    /// Initialize the public package against the linked Cargo build, once per test process.
    /// Call before attaching: concurrent imports must never wait for this initialization with the GIL.
    pub fn install_package(
        register: fn(&Bound<'_, PyModule>) -> PyResult<()>,
    ) -> &'static Py<PyModule> {
        static CORE: std::sync::OnceLock<Py<PyModule>> = std::sync::OnceLock::new();
        Python::initialize();
        CORE.get_or_init(|| {
            Python::attach(|py| -> PyResult<Py<PyModule>> {
                let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../../..")
                    .canonicalize()?;
                let sys = py.import("sys")?;
                let site = py.import("site")?;
                for path in sys.getattr("path")?.extract::<Vec<String>>()? {
                    if path.ends_with("site-packages") {
                        crate::py_calls::call(&site.getattr("addsitedir")?, (path,), None)?;
                    }
                }
                let paths = sys.getattr("path")?;
                crate::py_calls::call(
                    &paths.getattr("insert")?,
                    (0, root.join("livekit-agents-next").to_string_lossy().into_owned()),
                    None,
                )?;
                crate::py_calls::call(
                    &paths.getattr("insert")?,
                    (0, root.to_string_lossy().into_owned()),
                    None,
                )?;
                let core = PyModule::new(py, "livekit.agents._core")?;
                register(&core)?;
                let modules = sys.getattr("modules")?.cast_into::<pyo3::types::PyDict>()?;
                assert!(
                    !modules.contains("livekit.agents")?,
                    "bootstrap must precede public-package imports"
                );
                modules.set_item("livekit.agents._core", &core)?;
                let package = py.import("livekit.agents")?;
                let package_path: String = package.getattr("__file__")?.extract()?;
                assert!(
                    std::path::Path::new(&package_path)
                        .starts_with(root.join("livekit-agents-next")),
                    "binding contracts must load this checkout's Python package: {package_path}"
                );
                package.setattr("_core", &core)?;
                assert!(modules.get_item("livekit.agents._core")?.unwrap().is(&core));
                Ok(core.unbind())
            })
            .unwrap()
        })
    }
}
