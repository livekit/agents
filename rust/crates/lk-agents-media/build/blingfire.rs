//! Compiles the vendored BlingFire client library, tokeniser (models compiled in as C arrays), our
//! C++ wrapper and the C shim with `cc`: no CMake, no network. The defines are upstream's UNIX
//! `CMakeLists.txt` ones; without them the client reaches for `windows.h` and SAL annotations.

use std::path::{Path, PathBuf};

fn cpp_sources(dir: &Path, exts: &[&str]) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", dir.display()))
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| {
            p.extension().and_then(|e| e.to_str()).is_some_and(|e| exts.contains(&e))
        })
        .collect();
    // Sorted, so the archive's member order (and hash) is independent of the filesystem.
    out.sort();
    out
}

pub fn build() {
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let pin = std::fs::read_to_string(manifest.join("vendor/blingfire.pin"))
        .expect("vendor/blingfire.pin");
    let vendor = manifest.join(format!("vendor/blingfire-{}", &pin.trim()[..12]));
    assert!(
        vendor.join("blingfiretools/blingfiretokdll/blingfiretokdll.cpp").exists(),
        "the vendored BlingFire tree is missing from {}",
        vendor.display()
    );

    let client = vendor.join("blingfireclient.library");
    let tokdll = vendor.join("blingfiretools/blingfiretokdll");
    // Headers only: `blingfiretokdll.cpp` includes a template header from the compile library.
    let compile_inc = vendor.join("blingfirecompile.library/inc");

    println!("cargo:rerun-if-changed=csrc/blingfire");
    println!("cargo:rerun-if-changed=vendor/blingfire.pin");
    println!("cargo:rerun-if-changed={}", vendor.display());

    let mut build = cc::Build::new();
    build
        .cpp(true)
        .warnings(false)
        .include(client.join("inc"))
        .include(client.join("src"))
        .include(&compile_inc)
        .include(&vendor)
        .include(manifest.join("csrc/blingfire"))
        // Upstream's UNIX defines, verbatim.
        .define("NDEBUG", None)
        .define("HAVE_NO_SPECSTRINGS", None)
        .define("_VERBOSE", None)
        .define("BLING_FIRE_NOAP", None)
        .define("BLING_FIRE_NOWINDOWS", None);

    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        build.define("BLING_FIRE_MAC", None);
    }

    // Upstream compiles as C++11, a strict subset, so one standard covers both.
    build.std("c++17");

    for f in cpp_sources(&client.join("src"), &["cpp", "cxx"]) {
        build.file(f);
    }
    for f in cpp_sources(&tokdll, &["cpp", "cxx"]) {
        build.file(f);
    }
    build.file(manifest.join("csrc/blingfire/blingfire_wrap.cpp"));
    build.file(manifest.join("csrc/blingfire/shim.cpp"));

    build.compile("lk_agents_blingfire");
}
