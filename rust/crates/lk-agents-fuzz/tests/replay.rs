//! The corpus: every saved case in `replay/`, replayed against today's product.
//!
//! Not ignored. A finding is only kept once it runs in the ordinary `cargo test`.
use lk_agents_fuzz::replay;

#[test]
fn the_corpus_still_reproduces() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("replay");
    let mut files: Vec<_> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
        .map(|entry| entry.expect("a readable directory entry").path())
        .filter(|path| path.extension().is_some_and(|e| e == "json"))
        .collect();
    files.sort();

    // Every file is reported, so one stale case does not hide the rest.
    let mut mismatches = Vec::new();
    for path in &files {
        let name = path.file_name().unwrap_or_default().to_string_lossy().into_owned();
        if let Err(why) = replay::verify(&replay::load(path)) {
            mismatches.push(format!("  {name}: {why}"));
        }
    }
    assert!(
        mismatches.is_empty(),
        "{} of {} saved cases no longer reproduce:\n{}",
        mismatches.len(),
        files.len(),
        mismatches.join("\n")
    );
}
