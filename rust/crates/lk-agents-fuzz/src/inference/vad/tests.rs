use super::*;

/// A run that keeps every rule and reached a stale result, both speech edges and a flush.
fn reached_run() -> Trace {
    (0..400)
        .map(|seed| run(seed, Sampling::BasicPos))
        .find(|o| {
            o.violations.is_empty()
                && ["VadStaleResult", "VadStartOfSpeech", "VadEndOfSpeech"]
                    .iter()
                    .all(|r| o.reached.contains(r))
        })
        .expect("a seed reaches a stale result and both edges")
        .trace
}

fn breaks(rule: fn(&Trace) -> Vec<String>, mutate: impl FnOnce(&mut Trace)) {
    let mut trace = reached_run();
    assert!(rule(&trace).is_empty(), "{:?}", rule(&trace));
    mutate(&mut trace);
    assert!(!rule(&trace).is_empty(), "the counterexample passed");
}

fn first(trace: &mut Trace, what: impl Fn(&Step) -> bool) -> &mut Step {
    trace.steps.iter_mut().find(|s| what(s)).expect("the trace has such a step")
}

#[test]
fn a_second_window_while_one_awaits_its_result_is_rejected() {
    breaks(one_in_flight, |t| {
        let i = t.steps.iter().position(|s| s.returned.predict.is_some()).unwrap();
        let next = t.steps[i + 1..].iter_mut().find(|s| matches!(s.input, Input::Frame { .. }));
        next.unwrap().returned.predict = Some(1_000_000);
    });
}

#[test]
fn a_stale_result_that_changes_anything_is_rejected() {
    breaks(stale_inert, |t| {
        let mut in_flight = None;
        let mut stale = None;
        for (i, s) in t.steps.iter().enumerate() {
            if matches!(&s.input, Input::Result(r) if in_flight != Some(r.window)) {
                stale = Some(i);
                break;
            }
            in_flight = window_after(in_flight, s);
        }
        t.steps[stale.unwrap()].returned.predict = Some(1_000_000);
    });
}

#[test]
fn a_sample_index_that_skips_or_repeats_is_rejected() {
    breaks(index_per_result, |t| {
        let s =
            first(t, |s| s.returned.events.iter().any(|e| e.kind == VadEventType::InferenceDone));
        s.returned.events[0].samples_index += VAD_WINDOW_SAMPLES as u64;
    });
    breaks(index_per_result, |t| {
        let s =
            first(t, |s| s.returned.events.iter().any(|e| e.kind == VadEventType::InferenceDone));
        let again = s.returned.events[0].clone();
        s.returned.events.push(again);
    });
}

#[test]
fn speech_edges_out_of_turn_are_rejected() {
    breaks(edges_alternate, |t| {
        let s =
            first(t, |s| s.returned.events.iter().any(|e| e.kind == VadEventType::StartOfSpeech));
        let start =
            s.returned.events.iter().find(|e| e.kind == VadEventType::StartOfSpeech).unwrap();
        let again = start.clone();
        s.returned.events.push(again);
    });
    breaks(edges_alternate, |t| {
        let s =
            first(t, |s| s.returned.events.iter().any(|e| e.kind == VadEventType::StartOfSpeech));
        for e in &mut s.returned.events {
            if e.kind == VadEventType::StartOfSpeech {
                e.kind = VadEventType::EndOfSpeech;
            }
        }
    });
}

#[test]
fn a_seed_replays_its_run() {
    let (a, b) = (run(11, Sampling::BasicPos), run(11, Sampling::BasicPos));
    assert_eq!(a.trace.steps, b.trace.steps);
}
