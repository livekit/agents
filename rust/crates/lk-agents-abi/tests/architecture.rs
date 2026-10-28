//! Run repository boundary checks from the ordinary ABI suite and the launch gate.

// The launch gate runs ABI tests both without Python and against the interpreter. One pass
// is sufficient; the enforcement step itself selects the required Clippy feature sets.
#[cfg(not(feature = "python"))]
#[test]
fn repository_architecture_contracts() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..");
    let status = std::process::Command::new("bash")
        .arg(root.join("rust/architecture/check.sh"))
        .current_dir(&root)
        .status()
        .expect("start repository architecture checks");
    assert!(status.success(), "repository architecture checks failed: {status}");
}
