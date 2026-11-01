//! Give Cargo's embedded-interpreter tests the same Python home and dependencies as PyO3.

fn main() {
    println!("cargo:rerun-if-env-changed=PYTHONPATH");
    if std::env::var_os("HOST") != std::env::var_os("TARGET") {
        return;
    }
    let Some(python) = pyo3_build_config::get().executable() else { return };
    let output = std::process::Command::new(python)
        .args([
            "-c",
            "import sys, sysconfig; print(sys.base_prefix); print(sysconfig.get_path('purelib'))",
        ])
        .output()
        .expect("query the Python interpreter selected by PyO3");
    assert!(output.status.success(), "Python configuration query failed: {output:?}");
    let output = String::from_utf8(output.stdout).expect("Python configuration is UTF-8");
    let mut lines = output.lines();
    let home = lines.next().expect("Python base prefix");
    let dependencies = lines.next().expect("Python dependency directory");
    let mut paths = vec![std::path::PathBuf::from(dependencies)];
    if let Some(existing) = std::env::var_os("PYTHONPATH") {
        paths.extend(std::env::split_paths(&existing));
    }
    let paths = std::env::join_paths(paths).expect("valid Python dependency paths");
    // Cargo supplies these to test executables as well as rustc. No process-wide env mutation.
    println!("cargo:rustc-env=PYTHONHOME={home}");
    println!("cargo:rustc-env=PYTHONPATH={}", paths.to_str().expect("UTF-8 Python paths"));
}
