//! Regenerates `pprof.rs`, the pprof message types a job profile encodes, from the
//! `perftools.profiles` `pprof/profile.proto` backend-common vendors (and Cloud reads profiles
//! with), at the pinned commit [`COMMIT`]. Needs `protoc`:
//!
//! ```text
//! git -C ~/livekit/backend-common checkout 'a9bfd4360b6b69afbbe3bba221b0e542548f2e30'
//! BACKEND_COMMON_DIR=~/livekit/backend-common \
//!     cargo test -p lk-agents-net --lib export::profile::pprof_tests -- --ignored
//! ```

use std::path::PathBuf;

const COMMIT: &str = "a9bfd4360b6b69afbbe3bba221b0e542548f2e30";
const SOURCE: &str = "pprof/profile.proto";

#[test]
#[ignore = "needs protoc and BACKEND_COMMON_DIR (a livekit/backend-common checkout at COMMIT)"]
fn regenerate_pprof_proto() {
    let root = PathBuf::from(std::env::var("BACKEND_COMMON_DIR").expect("BACKEND_COMMON_DIR"));
    let out = std::env::temp_dir().join(format!("lk-pprof-{}", std::process::id()));
    std::fs::create_dir_all(&out).unwrap();
    prost_build::Config::new()
        .out_dir(&out)
        .disable_comments(["."])
        .compile_protos(&[root.join(SOURCE)], &[&root])
        .unwrap();
    let code = std::fs::read_to_string(out.join("perftools.profiles.rs")).unwrap();
    std::fs::remove_dir_all(&out).unwrap();
    let mut text = String::from("//! GENERATED FILE -- DO NOT EDIT.\n//!\n");
    text.push_str("//! Written by the ignored test `profile::pprof_tests` (prost-build) from\n");
    text.push_str(&format!("//! livekit/backend-common {COMMIT}, `{SOURCE}`.\n"));
    text.push_str(&code);
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let target = manifest.join("src/export/profile/pprof.rs");
    std::fs::write(&target, text).unwrap();
    let config = manifest.join("../../rustfmt.toml");
    let status = std::process::Command::new("rustfmt")
        .args(["--edition", "2024", "--config-path"])
        .arg(config)
        .arg(&target)
        .status()
        .unwrap();
    assert!(status.success());
}
