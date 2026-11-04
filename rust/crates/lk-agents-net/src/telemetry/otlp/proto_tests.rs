//! Regenerates `proto.rs`, the OTLP message types the exporter encodes, from the official
//! `open-telemetry/opentelemetry-proto` files at the pinned tag [`TAG`]. Needs `protoc`:
//!
//! ```text
//! curl -sSL \
//!     https://github.com/open-telemetry/opentelemetry-proto/archive/refs/tags/v1.5.0.tar.gz \
//!     | tar xz -C /tmp
//! OTLP_PROTO_DIR=/tmp/opentelemetry-proto-1.5.0 \
//!     cargo test -p lk-agents-net --lib otlp::proto_tests -- --ignored
//! ```

use std::collections::BTreeMap;
use std::path::PathBuf;

const TAG: &str = "v1.5.0";

const SOURCES: [&str; 3] = [
    "opentelemetry/proto/collector/trace/v1/trace_service.proto",
    "opentelemetry/proto/collector/logs/v1/logs_service.proto",
    "opentelemetry/proto/collector/metrics/v1/metrics_service.proto",
];

#[derive(Default)]
struct Module {
    code: String,
    children: BTreeMap<String, Module>,
}

impl Module {
    fn write(&self, out: &mut String) {
        out.push_str(&self.code);
        for (name, child) in &self.children {
            out.push_str(&format!("pub mod {name} {{\n"));
            child.write(out);
            out.push_str("}\n");
        }
    }
}

#[test]
#[ignore = "needs protoc and OTLP_PROTO_DIR (an opentelemetry-proto checkout at TAG)"]
fn regenerate_otlp_proto() {
    let root = PathBuf::from(std::env::var("OTLP_PROTO_DIR").expect("OTLP_PROTO_DIR"));
    let out = std::env::temp_dir().join(format!("lk-otlp-proto-{}", std::process::id()));
    std::fs::create_dir_all(&out).unwrap();
    let sources: Vec<PathBuf> = SOURCES.iter().map(|s| root.join(s)).collect();
    prost_build::Config::new()
        .out_dir(&out)
        .disable_comments(["."])
        .compile_protos(&sources, &[&root])
        .unwrap();
    // Packages `opentelemetry.proto.<path>` become nested modules under one root, so the
    // generated `super::` paths between packages resolve.
    let mut tree = Module::default();
    for entry in std::fs::read_dir(&out).unwrap() {
        let path = entry.unwrap().path();
        let name = path.file_stem().unwrap().to_str().unwrap().to_owned();
        let package = name.strip_prefix("opentelemetry.proto.").expect("an OTLP package");
        let module = package
            .split('.')
            .fold(&mut tree, |m, segment| m.children.entry(segment.to_owned()).or_default());
        module.code = std::fs::read_to_string(&path).unwrap();
    }
    std::fs::remove_dir_all(&out).unwrap();
    let mut text = String::from("//! GENERATED FILE -- DO NOT EDIT.\n//!\n");
    text.push_str("//! Written by the ignored test `otlp::proto_tests` (prost-build) from\n");
    text.push_str(&format!("//! open-telemetry/opentelemetry-proto {TAG}, the messages of:\n"));
    for source in SOURCES {
        text.push_str(&format!("//! - `{source}`\n"));
    }
    tree.write(&mut text);
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let target = manifest.join("src/telemetry/otlp/proto.rs");
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
