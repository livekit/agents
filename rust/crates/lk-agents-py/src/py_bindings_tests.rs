//! Public Python contracts against this Cargo build, in a dedicated interpreter process.

#[path = "py_test_support.rs"]
pub mod py_test_support;

use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::path::Path;

#[test]
fn public_python_binding_contracts() {
    let core = py_test_support::install_package(_core::register);
    Python::attach(|py| -> PyResult<()> {
        assert!(
            core.bind(py)
                .getattr("Role")?
                .is(py.get_type::<_core::generated::native_enums::Role>())
        );
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..").canonicalize()?;
        let sys = py.import("sys")?;
        let modules = sys.getattr("modules")?.cast_into::<PyDict>()?;

        let mut files: Vec<_> =
            std::fs::read_dir(Path::new(env!("CARGO_MANIFEST_DIR")).join("src"))?
                .map(|entry| entry.map(|entry| entry.path()))
                .collect::<Result<_, _>>()?;
        files.retain(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.ends_with("_tests.py"))
        });
        files.sort();
        assert!(!files.is_empty(), "no colocated Python binding tests found");
        let mut args = vec![
            "-c".to_owned(),
            root.join("pyproject.toml").to_string_lossy().into_owned(),
            "-o".into(),
            "addopts=".into(),
            "-q".into(),
        ];
        args.extend(files.iter().map(|path| path.to_string_lossy().into_owned()));
        let exit: i32 = py.import("pytest")?.call_method1("main", (args,))?.extract()?;
        // Python buffers stdout in Cargo's pipe; flush before Rust reports a failing exit code.
        sys.getattr("stdout")?.call_method0("flush")?;
        sys.getattr("stderr")?.call_method0("flush")?;
        assert_eq!(exit, 0, "Python binding contracts failed; see pytest output above");
        assert!(modules.get_item("livekit.agents._core")?.unwrap().is(core.bind(py)));
        Ok(())
    })
    .unwrap();
}
