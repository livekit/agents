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

    machine.deliver(audio("b", 2), &mut effects).unwrap();
    machine.deliver(done("b"), &mut effects).unwrap();
    assert_eq!(effects, [SegmentEffect::Progress], "b waits for a");

    effects.clear();
    machine.deliver(audio("a", 1), &mut effects).unwrap();
    machine.deliver(done("a"), &mut effects).unwrap();
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
    machine.deliver(done("a"), &mut effects).unwrap();
    assert_eq!(effects.last(), Some(&SegmentEffect::CompleteResponse), "b is still taking text");
    assert!(!machine.settled());
}

// Catches: a segment completed twice, or a provider's done for a segment it was never sent
// (or before its text ended) silently accepted.
#[test]
fn a_report_for_no_open_segment_or_an_early_done_is_refused() {
    let mut machine = SegmentMachine::default();
    let mut effects = Vec::new();
    assert_eq!(machine.deliver(audio("x", 0), &mut effects), Err(Refused::Unknown));
    machine.push(chunk("one"), || "a".into(), &mut effects);
    send_all(&mut machine, &mut effects);
    assert_eq!(machine.deliver(done("a"), &mut effects), Err(Refused::Early));
    machine.flush();
    send_all(&mut machine, &mut effects);
    machine.deliver(done("a"), &mut effects).unwrap();
    assert_eq!(machine.deliver(done("a"), &mut effects), Err(Refused::Unknown));
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
