use lk_agents_abi::tts::segments::{SegmentAudio, SegmentDone};

use super::*;

fn chunk(text: &str) -> SpeechChunk {
    SpeechChunk { text: text.into(), said: text.into() }
}

fn audio(id: &str, byte: u8) -> SegmentOutput {
    SegmentOutput::Audio(SegmentAudio { segment_id: id.into(), data: vec![byte] })
}

fn done(id: &str) -> SegmentOutput {
    SegmentOutput::Done(SegmentDone { segment_id: id.into() })
}

/// Takes every queued input, as the socket's send loop does.
fn send_all(machine: &mut SegmentMachine, effects: &mut Vec<SegmentEffect>) -> Vec<SegmentInput> {
    std::iter::from_fn(|| machine.next(effects)).collect()
}

// Catches: a later segment's audio played into the segment ahead of it when the provider
// answers out of order, mixing two sentences' audio and losing the flush between them.
#[test]
fn a_later_segments_audio_waits_until_the_one_before_it_is_done() {
    let mut machine = SegmentMachine::default();
    let mut effects = Vec::new();
    machine.push(chunk("one"), || "a".into(), &mut effects);
    machine.flush();
    machine.push(chunk("two"), || "b".into(), &mut effects);
    machine.flush();
    send_all(&mut machine, &mut effects);
    effects.clear();

    machine.deliver(audio("b", 2), || "next".into(), &mut effects).unwrap();
    machine.deliver(done("b"), || "next".into(), &mut effects).unwrap();
    assert_eq!(effects, [SegmentEffect::Progress], "b waits for a");

    effects.clear();
    machine.deliver(audio("a", 1), || "next".into(), &mut effects).unwrap();
    machine.deliver(done("a"), || "next".into(), &mut effects).unwrap();
    assert_eq!(
        effects,
        [
            SegmentEffect::Progress,
            SegmentEffect::Push(vec![1]),
            SegmentEffect::EndSegment,
            SegmentEffect::StartSegment("b".into()),
            SegmentEffect::Push(vec![2]),
            SegmentEffect::EndSegment,
            SegmentEffect::CompleteResponse,
        ]
    );
    assert!(machine.settled());
}

// Catches: a read deadline armed while the input waits for the LLM's next sentence (a slow LLM
// failing the turn), or left armed after the segment's answer came.
#[test]
fn the_deadline_runs_from_a_segments_end_until_no_ended_segment_is_unanswered() {
    let mut machine = SegmentMachine::default();
    let mut effects = Vec::new();
    machine.push(chunk("one"), || "a".into(), &mut effects);
    send_all(&mut machine, &mut effects);
    assert!(!effects.contains(&SegmentEffect::ExpectResponse), "open input owes nothing");

    machine.flush();
    machine.push(chunk("two"), || "b".into(), &mut effects);
    effects.clear();
    send_all(&mut machine, &mut effects);
    assert_eq!(effects, [SegmentEffect::ExpectResponse]);

    effects.clear();
    machine.deliver(done("a"), || "next".into(), &mut effects).unwrap();
    assert_eq!(effects.last(), Some(&SegmentEffect::CompleteResponse), "b is still taking text");
    assert!(!machine.settled());
}

// Catches: a segment completed twice, or a provider's report for a segment it was never sent
// silently accepted.
#[test]
fn a_report_for_no_open_segment_is_refused() {
    let mut machine = SegmentMachine::default();
    let mut effects = Vec::new();
    let next = || "next".to_owned();
    assert_eq!(machine.deliver(audio("x", 0), next, &mut effects), Err(Unknown));
    machine.push(chunk("one"), || "a".into(), &mut effects);
    machine.flush();
    send_all(&mut machine, &mut effects);
    machine.deliver(done("a"), next, &mut effects).unwrap();
    assert_eq!(machine.deliver(done("a"), next, &mut effects), Err(Unknown));
}

// Catches: a segment the provider ended while the LLM was slow to send its next text (its
// context expired) failing the stream, or the rest of its text sent to the ended context.
#[test]
fn a_segment_the_provider_ended_early_carries_its_rest_to_a_new_segment() {
    let mut machine = SegmentMachine::default();
    let mut effects = Vec::new();
    machine.push(chunk("one"), || "a".into(), &mut effects);
    send_all(&mut machine, &mut effects);
    machine.push(chunk("two"), || "unused".into(), &mut effects);
    machine.flush();
    effects.clear();

    machine.deliver(done("a"), || "b".into(), &mut effects).unwrap();
    assert_eq!(
        effects,
        [
            SegmentEffect::Expired("a".into()),
            SegmentEffect::EndSegment,
            SegmentEffect::StartSegment("b".into()),
            SegmentEffect::CompleteResponse,
        ]
    );
    let start = SegmentInput::Start(SegmentStart { segment_id: "b".into() });
    let text = SegmentInput::Text(SegmentText { segment_id: "b".into(), chunk: chunk("two") });
    let end = SegmentInput::End(SegmentEnd { segment_id: "b".into() });
    assert_eq!(send_all(&mut machine, &mut effects), [start, text, end]);
    assert_eq!(machine.open(), ["b"]);

    // ended while no text was queued: the next text opens its own segment
    machine.push(chunk("three"), || "c".into(), &mut effects);
    send_all(&mut machine, &mut effects);
    machine.deliver(done("c"), || "unused".into(), &mut effects).unwrap();
    machine.push(chunk("four"), || "d".into(), &mut effects);
    assert_eq!(machine.open(), ["b", "d"]);
}

// Catches: a flush with no text opening an empty segment, so the segment count the stream
// checks no longer matches the flushes that carried text.
#[test]
fn a_flush_with_no_text_opens_no_segment() {
    let mut machine = SegmentMachine::default();
    let mut effects = Vec::new();
    machine.flush();
    assert!(send_all(&mut machine, &mut effects).is_empty());
    assert!(effects.is_empty());
    assert!(machine.settled());
}

// Catches: a provider that finishes each segment by ending its session (closing, or saying it
// finished) leaving the segment open until the read deadline fails it, or a segment whose text
// had not ended completed with it.
#[test]
fn a_finished_session_completes_the_segments_whose_text_ended() {
    let mut machine = SegmentMachine::default();
    let mut effects = Vec::new();
    machine.push(chunk("one"), || "a".into(), &mut effects);
    machine.flush();
    machine.push(chunk("two"), || "b".into(), &mut effects);
    send_all(&mut machine, &mut effects);
    machine.deliver(audio("a", 1), || "next".into(), &mut effects).unwrap();
    effects.clear();

    machine.complete_ended(&mut effects);
    assert_eq!(
        effects,
        [
            SegmentEffect::EndSegment,
            SegmentEffect::StartSegment("b".into()),
            SegmentEffect::CompleteResponse,
        ]
    );
    assert_eq!(machine.open(), ["b"], "b's text has not ended");
}

// Catches: a provider's billed characters for a later segment reported while an earlier one
// plays, so they ride on the wrong segment's final audio and metrics.
#[test]
fn a_segments_usage_is_reported_while_it_plays() {
    use lk_agents_abi::tts::segments::SegmentUsage;
    let mut machine = SegmentMachine::default();
    let mut effects = Vec::new();
    machine.push(chunk("one"), || "a".into(), &mut effects);
    machine.flush();
    machine.push(chunk("two"), || "b".into(), &mut effects);
    machine.flush();
    send_all(&mut machine, &mut effects);
    effects.clear();

    let usage = |id: &str, n| {
        SegmentOutput::Usage(SegmentUsage {
            segment_id: id.into(),
            usage: None,
            characters_count: Some(n),
        })
    };
    machine.deliver(usage("b", 3), || "next".into(), &mut effects).unwrap();
    machine.deliver(done("b"), || "next".into(), &mut effects).unwrap();
    effects.clear();
    machine.deliver(usage("a", 2), || "next".into(), &mut effects).unwrap();
    machine.deliver(done("a"), || "next".into(), &mut effects).unwrap();
    let reported: Vec<_> = effects
        .iter()
        .filter_map(|e| match e {
            SegmentEffect::Usage(_, n) => Some(*n),
            SegmentEffect::EndSegment => Some(None),
            _ => None,
        })
        .collect();
    assert_eq!(reported, [Some(2), None, Some(3), None], "each before its own segment's end");
}
