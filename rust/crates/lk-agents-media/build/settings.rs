//! Native toolchain identity and the user-controlled persistent cache location.

use std::path::PathBuf;

use super::cache::Key;

pub fn native_debug() -> bool {
    match std::env::var("LK_AGENTS_NATIVE_DEBUG").as_deref() {
        Ok("1") => true,
        Ok("0") | Err(std::env::VarError::NotPresent) => false,
        _ => panic!("LK_AGENTS_NATIVE_DEBUG must be 0 or 1"),
    }
}

/// Keep native symbol policy consistent between cache keys and actual compilation.
pub fn compiler(cpp: bool) -> cc::Build {
    let mut build = cc::Build::new();
    build.cpp(cpp).debug(native_debug());
    build
}

pub fn root() -> PathBuf {
    for name in ["LK_AGENTS_NATIVE_CACHE_DIR", "HOME", "LOCALAPPDATA", "XDG_CACHE_HOME"] {
        println!("cargo:rerun-if-env-changed={name}");
    }
    let path = |name| std::env::var_os(name).map(PathBuf::from);
    let explicit = path("LK_AGENTS_NATIVE_CACHE_DIR");
    let preferred = super::cache::default_directory(
        std::env::consts::OS,
        path("HOME"),
        path("XDG_CACHE_HOME"),
        path("LOCALAPPDATA"),
    );
    let fallback = PathBuf::from(std::env::var_os("OUT_DIR").expect("OUT_DIR")).join("native");
    super::cache::directory(explicit.as_deref(), preferred.as_deref(), &fallback)
        .expect("select a writable native artifact cache")
}

/// Include the compiler's effective flags, not Rust-only flags or checkout paths.
pub fn key(cpp: bool) -> Key {
    let mut key = Key::new();
    // Debug symbols name the source checkout. Fast artifacts can move between
    // worktrees; debug artifacts must keep their original source paths valid.
    if native_debug() {
        let manifest = std::env::var_os("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR");
        key.input("debug-source-root", manifest.as_encoded_bytes());
    }
    let compiler = compiler(cpp).cargo_metadata(false).get_compiler();
    key.input("compiler", compiler.path().as_os_str().as_encoded_bytes());
    for arg in compiler.args() {
        key.input("compiler-argument", arg.as_encoded_bytes());
    }
    for (name, value) in compiler.env() {
        key.input("compiler-environment-name", name.as_encoded_bytes());
        key.input("compiler-environment-value", value.as_encoded_bytes());
    }
    let version = compiler
        .to_command()
        .arg(if compiler.is_like_msvc() { "/Bv" } else { "--version" })
        .output()
        .expect("query native compiler version");
    key.input("compiler-version-stdout", &version.stdout);
    key.input("compiler-version-stderr", &version.stderr);
    for name in [
        "HOST",
        "TARGET",
        "SDKROOT",
        "MACOSX_DEPLOYMENT_TARGET",
        "AR",
        "ARFLAGS",
        "LDFLAGS",
        "CPATH",
        "LIBRARY_PATH",
        "PKG_CONFIG_PATH",
        "PKG_CONFIG_LIBDIR",
        "PKG_CONFIG_SYSROOT_DIR",
    ] {
        println!("cargo:rerun-if-env-changed={name}");
        if let Some(value) = std::env::var_os(name) {
            key.input(name, value.as_encoded_bytes());
        }
    }
    let target = std::env::var("TARGET").expect("TARGET");
    for prefix in ["CC", "CXX", "CFLAGS", "CXXFLAGS", "CPPFLAGS", "AR", "ARFLAGS"] {
        for name in [
            prefix.to_owned(),
            format!("HOST_{prefix}"),
            format!("TARGET_{prefix}"),
            format!("{prefix}_{target}"),
            format!("{prefix}_{}", target.replace('-', "_")),
        ] {
            println!("cargo:rerun-if-env-changed={name}");
            if let Some(value) = std::env::var_os(&name) {
                key.input(&name, value.as_encoded_bytes());
            }
        }
    }
    key
}
