//! The corpus: every saved case in `replay/`, replayed against today's product.
//!
//! Not ignored. A finding is only kept once it runs in the ordinary `cargo test`.
use lk_agents_fuzz::{drive::run, input::Input, replay};

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

/// A case the survey saves holds the run's own ending (its release rounds, `Settle`, `Close`,
/// `AwaitExit`): replayed, it is that run, finding for finding, not the run finished twice.
#[test]
fn a_freshly_saved_case_replays_its_run() {
    let corpus = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("replay");
    // A minimised case holds no ending: running it is a fresh run, which `finish` ends.
    let ran = run(replay::load(corpus.join("no_premature_completion-10.json")).case);
    assert!(!ran.ok(), "the case no longer fails");
    assert!(ran.case.inputs.contains(&Input::Settle {}), "the run holds no ending");

    let path = std::env::temp_dir().join(format!("lk-agents-fuzz-{}.json", std::process::id()));
    replay::save(&path, &ran.case, replay::Expect::of(&ran)).expect("a writable temp dir");
    let saved = replay::load(&path);
    std::fs::remove_file(&path).ok();
    replay::verify(&saved).expect("the saved case reproduces its finding");

    let again = run(saved.case);
    assert_eq!(again.case.inputs, ran.case.inputs, "the replay appended inputs of its own");
    assert_eq!(again.failures, ran.failures);
}
