//! Long conversations: what only shows up at turn fifty. A resource the driver opens and never
//! closes, a counter that only climbs, a timer re-armed on every turn -- none of it is visible in
//! the two or three turns a survey run affords, and all of it is visible as a live-resource count
//! that keeps growing when the conversation does not.
//!
//! `VOICE_SOAK_SEEDS` and `VOICE_SOAK_BUDGET` size it; the default is eight callers of about forty
//! turns each.
//!
//! Every call must reach its close, so the residue is read off every one: a call that stopped
//! before it (a livelock, a release phase that never went quiet, a close that never finished)
//! fails the soak, by seed with why, rather than being left out of the oracle. A long call's
//! findings outside the known ones fail the soak as they fail the survey.
use std::collections::{BTreeMap, BTreeSet};

use lk_agents_fuzz::{
    check::KNOWN_FINDINGS,
    drive::{Run, Setup, simulate_with},
    record::{Kind, Phase},
    search::{Dials, explore_at, sample_setup},
};

/// Per kind, the most of it live in each quarter of the run, by stamp, and what is left live at
/// the end.
fn live_by_quarter(run: &Run) -> BTreeMap<Kind, ([u32; 4], u32)> {
    let record = &run.record;
    let mut events: Vec<(u64, Kind, i32)> = record
        .opened()
        .iter()
        .map(|o| (o.at.seq, o.kind, 1))
        .chain(record.closed().iter().map(|(at, kind, _)| (at.seq, *kind, -1)))
        .collect();
    events.sort();
    let span = events.last().map_or(1, |e| e.0.max(1));
    let mut live: BTreeMap<Kind, i32> = BTreeMap::new();
    let mut peak: BTreeMap<Kind, [u32; 4]> = BTreeMap::new();
    for (seq, kind, delta) in events {
        let n = live.entry(kind).or_default();
        *n += delta;
        let quarter = ((seq * 4) / span).min(3) as usize;
        let p = peak.entry(kind).or_default();
        p[quarter] = p[quarter].max((*n).max(0) as u32);
    }
    peak.into_iter().map(|(k, p)| (k, (p, live[&k].max(0) as u32))).collect()
}

/// What lives as long as the session does, and is rightly still there at the end: the legs and
/// the outputs the case attached, the socket. Everything else a run opens, it must have closed
/// by the time the release phase is over.
fn lives_with_the_session(kind: Kind) -> bool {
    matches!(
        kind,
        Kind::Mic
            | Kind::Vad
            | Kind::Eot
            | Kind::Interruption
            | Kind::Output
            | Kind::Video
            | Kind::Stt
            | Kind::Realtime
    )
}

#[test]
#[ignore]
fn nothing_grows_with_the_conversation() {
    let env =
        |key: &str, or: u64| std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(or);
    let seeds = env("VOICE_SOAK_SEEDS", 8);
    let budget = env("VOICE_SOAK_BUDGET", 12_000) as usize;
    let mut problems: Vec<String> = Vec::new();
    let mut closed = 0;
    for seed in 0..seeds {
        // A caller the mode can converse with: a recogniser, and no realtime socket.
        let setup = Setup {
            stt: true,
            realtime: None,
            ..sample_setup(&mut fastrand::Rng::with_seed(seed))
        };
        let outcome = explore_at(setup, seed, budget, Dials::conversation());
        for failure in &outcome.failures {
            if !KNOWN_FINDINGS.contains(&failure.contract.as_str()) {
                problems.push(format!("seed {seed}: {}: {}", failure.contract, failure.detail));
            }
        }
        let case = outcome.case;
        let run = simulate_with(case.schedule, async {
            let mut run = Run::start(&case.setup, case.schedule, case.policy).await;
            for input in &case.inputs {
                run.apply(input).await;
            }
            run
        });
        if run.record.phase() != Phase::Closed {
            let why: BTreeSet<String> = outcome.failures.into_iter().map(|f| f.contract).collect();
            problems.push(format!("seed {seed}: stopped before its close: {why:?}"));
            continue;
        }
        closed += 1;
        let turns = case.inputs.iter().filter(|i| i.name() == "SttFinal").count();
        println!("seed {seed}: {} inputs, {turns} user turns", case.inputs.len());
        for (kind, (peak, left)) in live_by_quarter(&run) {
            println!(
                "  {:<13} peak live by quarter {peak:?}, left live {left}",
                format!("{kind:?}")
            );
            // A leak is what is still live once the release phase has released everything it
            // could: the peaks are for a reader, the residue is the oracle.
            if left > 0 && !lives_with_the_session(kind) {
                problems.push(format!(
                    "seed {seed}: {left} {kind:?} still live after the release phase (peaks {peak:?})"
                ));
            }
        }
    }
    println!("{closed} of {seeds} calls reached their close and were judged");
    assert!(problems.is_empty(), "the soak failed:\n  {}", problems.join("\n  "));
}
