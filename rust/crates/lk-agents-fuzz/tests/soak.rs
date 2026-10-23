//! Long conversations: what only shows up at turn fifty. A resource the driver opens and never
//! closes, a counter that only climbs, a timer re-armed on every turn -- none of it is visible in
//! the two or three turns a survey run affords, and all of it is visible as a live-resource count
//! that keeps growing when the conversation does not.
//!
//! `VOICE_SOAK_SEEDS` and `VOICE_SOAK_BUDGET` size it; the default is eight callers of about forty
//! turns each.
use std::collections::BTreeMap;

use lk_agents_fuzz::{
    drive::{Run, Setup, simulate_with},
    record::Kind,
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
    let mut growing: Vec<String> = Vec::new();
    for seed in 0..seeds {
        // A caller the mode can converse with: a recogniser, and no realtime socket.
        let setup = Setup {
            stt: true,
            realtime: None,
            ..sample_setup(&mut fastrand::Rng::with_seed(seed))
        };
        let case = explore_at(setup, seed, budget, Dials::conversation()).case;
        let run = simulate_with(case.schedule, async {
            let mut run = Run::start(&case.setup).await;
            for input in &case.inputs {
                run.apply(input).await;
            }
            run
        });
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
                growing.push(format!(
                    "seed {seed}: {left} {kind:?} still live after the release phase (peaks {peak:?})"
                ));
            }
        }
    }
    assert!(
        growing.is_empty(),
        "resources left live after the release phase:\n  {}",
        growing.join("\n  ")
    );
}
