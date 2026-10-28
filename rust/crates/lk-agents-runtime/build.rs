use std::{env, fs, path::PathBuf};

use serde::Deserialize;

#[derive(Deserialize)]
struct Budgets {
    session_events: EventLimits,
}

#[derive(Deserialize)]
struct EventLimits {
    worker_items: usize,
    worker_bytes: usize,
}

fn main() {
    let manifest = "src/telemetry/budgets.json";
    println!("cargo:rerun-if-changed={manifest}");
    let limits = serde_json::from_slice::<Budgets>(&fs::read(manifest).expect("telemetry budgets"))
        .expect("typed telemetry event limits")
        .session_events;
    assert!(limits.worker_items > 0 && limits.worker_bytes > 0, "positive event retention limits");
    let output = PathBuf::from(env::var_os("OUT_DIR").expect("Cargo output directory"));
    fs::write(
        output.join("telemetry_event_limits.rs"),
        format!(
            "// GENERATED FILE -- DO NOT EDIT.\nconst PROCESS_EVENT_ITEMS: usize = {};\nconst PROCESS_EVENT_BYTES: usize = {};\n",
            limits.worker_items, limits.worker_bytes,
        ),
    )
    .expect("write telemetry event limits");
}
