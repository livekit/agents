//! Checks test-source reachability from the explicit Cargo and regression targets.

use std::{collections::BTreeSet, fs, path::Path};

/// Root test files are either isolated Cargo targets or included by the regression target.
/// Keep registration explicit: silently leaving a new test file out must fail the next build.
pub(crate) fn check_test_modules(root: &Path) {
    let manifest = fs::read_to_string(root.join("Cargo.toml")).expect("Cargo.toml exists");
    let regression_path = root.join("src/regressions_tests.rs");
    println!("cargo:rerun-if-changed={}", root.join("Cargo.toml").display());
    println!("cargo:rerun-if-changed={}", regression_path.display());
    println!("cargo:rerun-if-changed={}", root.join("src").display());
    let mut tests = Vec::new();
    let mut pending = vec![regression_path];
    for entry in fs::read_dir(root.join("src")).expect("src exists") {
        let path = entry.expect("read src entry").path();
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else { continue };
        if !name.ends_with("_tests.rs") {
            continue;
        }
        let target = format!("path = \"src/{name}\"");
        if manifest.contains(&target) {
            pending.push(path.clone());
        }
        tests.push(path);
    }
    let mut registered = BTreeSet::new();
    while let Some(path) = pending.pop() {
        let path = path.canonicalize().expect("registered test source exists");
        if !registered.insert(path.clone()) {
            continue;
        }
        println!("cargo:rerun-if-changed={}", path.display());
        let source = fs::read_to_string(&path).expect("registered test source is readable");
        for line in source.lines() {
            if let Some(relative) =
                line.trim().strip_prefix("#[path = \"").and_then(|value| value.strip_suffix("\"]"))
            {
                pending.push(path.parent().unwrap().join(relative));
            }
        }
    }
    for path in tests {
        let name = path.file_name().unwrap().to_str().unwrap();
        assert!(
            registered.contains(&path.canonicalize().unwrap()),
            "src/{name} is not registered: declare a Cargo test target or include it in regressions_tests.rs"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    struct Fixture(PathBuf);

    impl Fixture {
        fn new(files: &[(&str, &str)]) -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let root = env::temp_dir().join(format!(
                "lk-test-registration-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir_all(root.join("src")).unwrap();
            fs::write(root.join("Cargo.toml"), "").unwrap();
            fs::write(root.join("src/regressions_tests.rs"), "").unwrap();
            for (name, source) in files {
                fs::write(root.join(name), source).unwrap();
            }
            Self(root)
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn nested_test_modules_are_registered_through_their_parent() {
        let fixture = Fixture::new(&[
            ("src/regressions_tests.rs", "#[path = \"outer_tests.rs\"]\nmod outer;"),
            ("src/outer_tests.rs", "#[path = \"inner_tests.rs\"]\nmod inner;"),
            ("src/inner_tests.rs", ""),
        ]);
        check_test_modules(&fixture.0);
    }

    #[test]
    fn explicit_cargo_targets_are_registered() {
        let fixture = Fixture::new(&[
            ("Cargo.toml", "[[test]]\nname = \"isolated\"\npath = \"src/isolated_tests.rs\""),
            ("src/isolated_tests.rs", ""),
        ]);
        check_test_modules(&fixture.0);
    }

    #[test]
    fn an_unreferenced_test_module_is_rejected() {
        let fixture = Fixture::new(&[("src/orphan_tests.rs", "")]);
        assert!(std::panic::catch_unwind(|| check_test_modules(&fixture.0)).is_err());
    }

    #[test]
    fn mutually_referencing_orphans_are_still_rejected() {
        let fixture = Fixture::new(&[
            ("src/first_tests.rs", "#[path = \"second_tests.rs\"]\nmod second;"),
            ("src/second_tests.rs", "#[path = \"first_tests.rs\"]\nmod first;"),
        ]);
        assert!(std::panic::catch_unwind(|| check_test_modules(&fixture.0)).is_err());
    }
}
