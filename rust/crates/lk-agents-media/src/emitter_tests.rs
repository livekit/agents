//! Tests for [`AudioEmitter`](super::AudioEmitter) (`tts/tts.py:835-1344`), from
//! `agents/tests/test_tts.py:799-1110` on a virtual clock.
//!
//! The fixture is the reference's: 1 kHz mono, `frame_size_ms=100`, so a 100 ms frame is 100
//! samples and the tail is 10. `AudioByteStream` is progressive: a 100-sample push comes out as
//! 20 + 40 + (40 held).

use crate::Frame;
use lk_agents_abi::media::emitter::EmitterError;
use lk_agents_abi::{SynthesizedAudio, TranscriptChunk, Ts};

use super::{AudioEmitter, Effects, HostEffect};

const SR: u32 = 1000;
const CH: u16 = 1;
const FRAME_MS: u32 = 100;
/// `_TAIL_SAMPLES` at 1 kHz.
const TAIL: u32 = 10;

fn secs(s: f64) -> Ts {
    Ts::from_nanos((s * 1e9) as u64)
}

/// Every frame sent, in order.
fn sent(eff: &Effects) -> Vec<&SynthesizedAudio> {
    eff.iter()
        .filter_map(|e| match e {
            HostEffect::Send(a) => Some(&**a),
            _ => None,
        })
        .collect()
}

/// When the drift timer is armed for, for effects recorded at `Ts::ZERO`; replays arm/cancel.
fn armed_at(eff: &Effects) -> Option<Ts> {
    let mut at = None;
    for e in eff {
        match e {
            HostEffect::Arm(delay) => at = Some(Ts::ZERO.saturating_add(*delay)),
            HostEffect::Cancel => at = None,
            _ => {}
        }
    }
    at
}

/// `n` samples of non-zero PCM, distinguishable from the silence marker.
fn pcm(samples: usize) -> Vec<u8> {
    vec![0xffu8; samples * 2]
}

struct Fx {
    em: AudioEmitter,
    eff: Effects,
    now: Ts,
}

impl Fx {
    fn new(stream: bool) -> Fx {
        let mut em = AudioEmitter::default();
        em.initialize("req", SR, CH, FRAME_MS, stream).unwrap();
        Fx { em, eff: Effects::new(), now: Ts::ZERO }
    }

    fn streaming(segment_id: &str) -> Fx {
        let mut fx = Fx::new(true);
        fx.em.start_segment(segment_id).unwrap();
        fx
    }

    fn at(&mut self, s: f64) -> &mut Self {
        self.now = secs(s);
        self
    }

    fn push(&mut self, samples: usize) -> &mut Self {
        self.em.push(self.now, &pcm(samples), &mut self.eff).unwrap();
        self
    }

    fn flush(&mut self) -> &mut Self {
        self.em.flush(self.now, &mut self.eff).unwrap();
        self
    }

    fn end_segment(&mut self) -> &mut Self {
        self.em.end_segment(self.now, &mut self.eff).unwrap();
        self
    }

    fn fire_drift(&mut self) -> &mut Self {
        self.em.on_drift(self.now, &mut self.eff).unwrap();
        self
    }

    /// Samples per delivered frame, in order.
    fn sizes(&self) -> Vec<u32> {
        sent(&self.eff).iter().map(|a| a.frame.samples_per_channel).collect()
    }

    fn finals(&self) -> Vec<bool> {
        sent(&self.eff).iter().map(|a| a.is_final).collect()
    }

    fn bytes(&self) -> Vec<u8> {
        sent(&self.eff).iter().flat_map(|a| a.frame.data.iter().copied()).collect()
    }
}

// --------------------------------------------------------------------- the drift watchdog

// Catches: the drift timer not firing, or firing without releasing the held tail (audio stalls
// for a provider slower than realtime).
#[test]
fn the_drift_flush_releases_the_held_back_tail() {
    let mut fx = Fx::streaming("s");
    fx.at(0.0).push(100).push(100).push(100);
    let armed = armed_at(&fx.eff).expect("armed");
    // sent_duration 0.23 s, no wall time elapsed => 0.23 - 0 - 0.02.
    assert_eq!(armed, secs(0.21));

    // 300 pushed, 230 delivered: 60 buffered short of a frame plus the 10-sample tail.
    assert_eq!(fx.sizes().iter().sum::<u32>(), 230);
    fx.at(0.21).fire_drift();
    assert_eq!(fx.sizes().iter().sum::<u32>(), 300, "everything held back is released");
    assert_eq!(*fx.sizes().last().unwrap(), TAIL, "the tail goes last");
    assert_eq!(fx.finals().last(), Some(&false), "a drift flush is not the end of the segment");
    assert!(armed_at(&fx.eff).is_none(), "and the watchdog stands down until new audio arrives");
}

// Catches: arming with a non-positive delay (a flush after every frame once behind).
#[test]
fn no_timer_is_armed_once_the_provider_is_already_behind() {
    let mut fx = Fx::streaming("s");
    // 0.23 s of audio produced, but 0.30 s of wall clock spent producing it.
    fx.at(0.0).push(100).at(0.30).push(100).push(100);
    assert!(armed_at(&fx.eff).is_none(), "delay <= 0 arms nothing");
}

// Catches: leaking timer handles, or cancelling after re-arming.
#[test]
fn every_frame_cancels_the_previous_drift_timer_before_arming_a_new_one() {
    let mut fx = Fx::streaming("s");
    fx.at(0.0).push(100).push(100).push(100);
    assert!(armed_at(&fx.eff).is_some());
    std::mem::take(&mut fx.eff);

    fx.push(100);
    let calls = std::mem::take(&mut fx.eff);
    let shape: Vec<&str> = calls
        .iter()
        .map(|c| match c {
            HostEffect::Send(_) => "send",
            HostEffect::Cancel => "cancel",
            HostEffect::Arm(_) => "arm",
            _ => "other",
        })
        .collect();
    assert_eq!(shape, vec!["send", "cancel", "arm"]);
}

// Catches: forgetting the `sent_start`/`sent_duration` reset in `_flush_frame`.
#[test]
fn flushing_restarts_the_drift_measurement() {
    let mut fx = Fx::streaming("s");
    fx.at(0.0).push(100).push(100).push(100);
    assert!(armed_at(&fx.eff).is_some());

    fx.flush();
    assert!(armed_at(&fx.eff).is_none(), "the flush disarms");

    std::mem::take(&mut fx.eff);
    fx.at(0.5).push(100);
    assert!(
        armed_at(&fx.eff).is_none(),
        "accounting restarted: one 100 ms frame is under the 0.15 s floor again"
    );
}

// --------------------------------------------------------------------- tail splitting

// Catches: computing the split index in samples instead of bytes.
#[test]
fn splitting_the_tail_moves_no_bytes_and_loses_none() {
    let mut fx = Fx::streaming("s");
    fx.push(100).push(100).push(100).end_segment();
    assert_eq!(fx.bytes(), pcm(300), "every pushed byte comes out, in order, exactly once");
    assert_eq!(fx.sizes().iter().sum::<u32>(), 300);
}

// Catches: emitting a zero-sample head.
#[test]
fn a_frame_no_bigger_than_the_tail_is_held_whole() {
    let mut fx = Fx::streaming("s");
    fx.push(TAIL as usize).end_segment();
    assert_eq!(fx.sizes(), vec![TAIL], "one frame, not a zero-sample head and a tail");
    assert_eq!(fx.finals(), vec![true]);
}

// Catches: "fixing" the reference: a segment that never received a byte is never closed, so the
// next `start_segment()` raises.
#[test]
fn ending_a_segment_that_never_got_a_byte_does_not_close_it() {
    let mut fx = Fx::streaming("e1");
    fx.end_segment();
    assert_eq!(
        fx.em.start_segment("e2").unwrap_err(),
        EmitterError::SegmentAlreadyOpen,
        "reference quirk, pinned deliberately"
    );
}

// --------------------------------------------------------------------- segment accounting

// Catches: `pushed_duration` losing Python's indexing, especially a default of `0` instead of `-1`
// (a late failure would replay the whole utterance).
#[test]
fn pushed_duration_indexes_the_way_python_does() {
    let mut fx = Fx::streaming("a");
    fx.push(100).end_segment();
    fx.em.start_segment("b").unwrap();
    fx.push(200).end_segment();

    assert!((fx.em.pushed_duration(0) - 0.1).abs() < 1e-9);
    assert!((fx.em.pushed_duration(1) - 0.2).abs() < 1e-9);
    assert_eq!(fx.em.pushed_duration(-1), fx.em.pushed_duration(1), "-1 is the newest segment");
    assert_eq!(fx.em.pushed_duration(-2), fx.em.pushed_duration(0));
    assert_eq!(fx.em.pushed_duration(2), 0.0, "out of range is 0.0, never a panic");
    assert_eq!(fx.em.pushed_duration(-3), 0.0);
}

// Catches: overlapping segments going unnoticed, the streaming-only guard on the segment API
// dropped, `initialize()` callable twice, a push before it ignored, or `stop()` requiring it.
#[test]
fn misuse_of_the_lifecycle_is_refused() {
    let mut fx = Fx::streaming("a");
    fx.push(100);
    assert_eq!(fx.em.start_segment("b").unwrap_err(), EmitterError::SegmentAlreadyOpen);
    assert_eq!(
        fx.em.initialize("req2", SR, CH, FRAME_MS, true).unwrap_err(),
        EmitterError::AlreadyStarted
    );

    let mut fx = Fx::new(false);
    assert_eq!(fx.em.start_segment("x").unwrap_err(), EmitterError::StartSegmentNotStreaming);
    assert_eq!(
        fx.em.end_segment(fx.now, &mut fx.eff).unwrap_err(),
        EmitterError::EndSegmentNotStreaming
    );

    let mut em = AudioEmitter::default();
    let mut eff = Effects::new();
    assert_eq!(em.push(Ts::ZERO, &pcm(10), &mut eff).unwrap_err(), EmitterError::NotStarted);
    em.stop(&mut eff);
    assert!(eff.is_empty());
}

// Catches: losing the provider id when the segment it arrived with is rejected.
#[test]
fn a_provider_id_is_recorded_even_when_its_segment_is_rejected() {
    let mut fx = Fx::streaming("p1");
    assert_eq!(fx.em.take_provider_ids_update(), Some(vec!["p1".to_string()]));

    // No byte was pushed, so "p1" is still open (see the quirk above) and this is rejected.
    assert!(fx.em.start_segment("p2").is_err());
    assert_eq!(
        fx.em.take_provider_ids_update(),
        Some(vec!["p1".to_string(), "p2".to_string()]),
        "the id was noted before the segment was refused"
    );
    assert_eq!(fx.em.take_provider_ids_update(), None, "and it is published once, not per call");
}

// --------------------------------------------------------------------- identity and ids

// Catches: `started()` and `request_id()` answering constants, and `initialize("")` not becoming
// `"unknown"`.
#[test]
fn the_request_id_is_normalised_once_and_rides_every_frame() {
    let mut fx = Fx::streaming("s");
    assert!(fx.em.started(), "initialize() must latch, not just refuse a second call");
    assert_eq!(fx.em.request_id(), "req");
    fx.push(100).end_segment();
    assert!(!sent(&fx.eff).is_empty());
    assert!(sent(&fx.eff).iter().all(|a| a.request_id == "req"), "every frame carries it");

    let mut blank = AudioEmitter::default();
    blank.initialize("", SR, CH, FRAME_MS, true).unwrap();
    assert_eq!(blank.request_id(), "unknown", "an empty id becomes `unknown`, not `\"\"`");
}

// Catches: `provider_request_ids()` answering a constant, and a dedup guard written as `&&`.
#[test]
fn provider_ids_accumulate_in_order_without_blanks_or_repeats() {
    let mut fx = Fx::streaming("p1");
    assert_eq!(fx.em.provider_request_ids(), ["p1"]);

    assert!(!fx.em.note_provider_request_id(""), "an empty context id is not an id");
    assert_eq!(fx.em.provider_request_ids(), ["p1"], "and must not be appended");

    assert!(!fx.em.note_provider_request_id("p1"), "the same id twice is not new");
    assert_eq!(fx.em.provider_request_ids(), ["p1"]);

    assert!(fx.em.note_provider_request_id("p2"));
    assert_eq!(fx.em.provider_request_ids(), ["p1", "p2"], "in the order they were learnt");
}

// --------------------------------------------------------------------- the empty-segment guards

// Catches: a silence marker for a segment that opened its byte stream (a zero-byte push) but
// produced no audio.
#[test]
fn a_segment_that_opened_a_byte_stream_but_produced_no_audio_stays_silent() {
    let mut fx = Fx::streaming("s");
    fx.em.push(fx.now, &[], &mut fx.eff).unwrap();
    fx.end_segment();

    assert!(sent(&fx.eff).is_empty(), "no 10 ms click for a segment that never spoke");
    assert_eq!(fx.em.pushed_duration(0), 0.0);
}

// Catches: `audio_duration` folded with anything but `+` in `_flush_frame`, in the one shape where
// the flushed tail is the only audio.
#[test]
fn a_segment_whose_only_audio_was_the_flushed_tail_still_closes() {
    let mut fx = Fx::streaming("s");
    fx.push(5); // 5 samples: below TAIL, so nothing is ever handed to `_do_send`
    assert!(sent(&fx.eff).is_empty(), "the premise: the frame is too small to give a head away");

    fx.flush();
    assert_eq!(fx.sizes(), vec![5], "the flush releases it, and it is not final");
    assert_eq!(fx.finals(), vec![false]);

    fx.end_segment();
    assert_eq!(fx.sizes(), vec![5, SR / 100], "the segment did speak, so it gets its marker");
    assert_eq!(fx.finals(), vec![false, true]);
}

// --------------------------------------------------------------------- torn samples

// Catches: a flush dropping a partial sample more bytes complete, or an ended segment's torn
// tail not counted; `take_dropped_bytes` answering a constant.
#[test]
fn a_flush_keeps_a_partial_sample_and_an_ended_segment_drops_it() {
    let mut fx = Fx::streaming("s");
    fx.em.push(fx.now, &[0xff, 0xff, 0xff], &mut fx.eff).unwrap(); // 1.5 samples at 16-bit mono
    fx.flush();
    assert_eq!(fx.em.take_dropped_bytes(), 0, "a flush mid-stream keeps the partial sample");
    fx.end_segment();
    assert_eq!(fx.em.take_dropped_bytes(), 1, "the segment ended inside a sample");
    assert_eq!(fx.em.take_dropped_bytes(), 0, "reading it is destructive");
}

// --------------------------------------------------------------------- the 150 ms floor

// Catches: the drift floor written as `>=`; 20 ms frames land exactly on 0.150 s.
#[test]
fn exactly_150ms_of_sent_audio_is_not_yet_past_the_floor() {
    let mut em = AudioEmitter::default();
    em.initialize("req", SR, CH, 20, true).unwrap(); // 20 ms frames: sends of 10, then 20
    let mut eff = Effects::new();
    em.start_segment("s").unwrap();

    em.push(Ts::ZERO, &pcm(160), &mut eff).unwrap();
    assert_eq!(
        sent(&eff).iter().map(|a| a.frame.samples_per_channel).sum::<u32>(),
        150,
        "the premise: sent_duration is exactly 0.150 s"
    );
    assert_eq!(armed_at(&eff), None, "0.15 is not greater than 0.15");

    em.push(Ts::ZERO, &pcm(20), &mut eff).unwrap();
    assert_eq!(armed_at(&eff), Some(secs(0.15)), "0.17 is, and delay is 0.17 - 0 - 0.02");
}

// --------------------------------------------------------------------- timed transcripts

// Catches: copying the buffered words onto every frame instead of consuming them.
#[test]
fn buffered_words_ride_exactly_one_frame() {
    let mut fx = Fx::streaming("s");
    fx.em.push_timed_transcript([TranscriptChunk::new("hello")]).unwrap();
    fx.push(100).push(100);

    let sent = sent(&fx.eff);
    let carrying: Vec<usize> = sent
        .iter()
        .enumerate()
        .filter(|(_, a)| !a.timed_transcript.is_empty())
        .map(|(i, _)| i)
        .collect();
    assert_eq!(carrying, vec![0], "the first frame out, and only that one");
    assert_eq!(sent[0].timed_transcript[0].text, "hello");
}

// --------------------------------------------------------------------- lifecycle

// Catches: raising after `end_input()` instead of returning silently.
#[test]
fn everything_is_a_silent_no_op_after_end_input() {
    let mut fx = Fx::streaming("s");
    fx.push(100);
    fx.em.end_input(fx.now, &mut fx.eff).unwrap();
    let after = std::mem::take(&mut fx.eff);
    assert!(!after.is_empty(), "end_input finishes the segment first");

    fx.em.push(fx.now, &pcm(100), &mut fx.eff).unwrap();
    fx.em.flush(fx.now, &mut fx.eff).unwrap();
    fx.em.push_timed_transcript([TranscriptChunk::new("x")]).unwrap();
    assert!(std::mem::take(&mut fx.eff).is_empty(), "nothing may reach the consumer after close");
}

// Catches: `aclose()` leaving the drift timer armed.
#[test]
fn stopping_disarms_the_drift_timer() {
    let mut fx = Fx::streaming("s");
    fx.at(0.0).push(100).push(100).push(100);
    assert!(armed_at(&fx.eff).is_some());
    fx.em.stop(&mut fx.eff);
    assert!(armed_at(&fx.eff).is_none());
}

// Catches: a framed chunk re-chunked into progressive frames, or taken outside a segment.
#[test]
fn a_pushed_frame_goes_out_as_framed_with_only_the_tail_held() {
    let mut fx = Fx::streaming("s");
    let frame = Frame { data: vec![0u8; 960 * 2], samples_per_channel: 960 };
    fx.em.push_frame(fx.now, frame, &mut fx.eff).unwrap();
    let out: u32 = sent(&fx.eff).iter().map(|a| a.frame.samples_per_channel).sum();
    assert!(out < 960 && out > 0, "one frame out, its 10 ms tail held: {out}");
    fx.end_segment();
    let total: u32 = sent(&fx.eff).iter().map(|a| a.frame.samples_per_channel).sum();
    assert_eq!(total, 960);

    let mut idle = Fx::new(true);
    let frame = Frame { data: vec![0u8; 4], samples_per_channel: 2 };
    assert_eq!(idle.em.push_frame(idle.now, frame, &mut idle.eff), Err(EmitterError::NoSegment));
}

// --------------------------------------------------------------------- segments

// Catches: one byte stream (or its progressive size) carried across segments or a flush, so every
// sentence after the first starts a whole frame late instead of with the 20 ms first frame.
#[test]
fn each_segment_and_each_flush_restarts_with_the_small_first_frame() {
    let mut fx = Fx::streaming("a");
    fx.push(100);
    let first = fx.sizes()[0];
    assert_eq!(first, 20 - TAIL, "the premise: 20 ms out first, its 10 ms tail held");
    fx.end_segment();

    fx.em.start_segment("b").unwrap();
    let before = fx.sizes().len();
    fx.push(100);
    assert_eq!(fx.sizes()[before], first, "the next segment starts small again");

    fx.flush();
    let before = fx.sizes().len();
    fx.push(100);
    assert_eq!(fx.sizes()[before], first, "and so does the burst after a flush");
}

// Catches: the segment cleared before its final frame is built (a final with an empty id), a
// second final per segment, or the silence marker attributed to the next segment.
#[test]
fn every_frame_carries_its_segment_id_and_each_segment_ends_exactly_once() {
    let mut fx = Fx::streaming("a");
    fx.push(100).end_segment();
    fx.em.start_segment("b").unwrap();
    fx.push(50).flush().end_segment(); // the tail went out on the flush: b ends on the marker

    let frames: Vec<(&str, bool)> =
        sent(&fx.eff).iter().map(|a| (a.segment_id.as_str(), a.is_final)).collect();
    let split = frames.iter().position(|(id, _)| *id == "b").expect("b spoke");
    let (a, b) = frames.split_at(split);
    assert!(a.iter().all(|(id, _)| *id == "a") && b.iter().all(|(id, _)| *id == "b"));
    assert_eq!(a.iter().filter(|(_, f)| *f).count(), 1);
    assert_eq!(b.iter().filter(|(_, f)| *f).count(), 1);
    assert!(a.last().unwrap().1 && b.last().unwrap().1, "the final is each segment's last frame");
    assert_eq!(fx.em.num_segments(), 2);
}

// Catches: a non-streaming synthesis requiring `start_segment`, or `end_input` not finalizing its
// implicit segment (the consumer would wait for an `is_final` that never comes).
#[test]
fn a_non_streaming_synthesis_is_one_implicit_segment_that_end_input_closes() {
    let mut fx = Fx::new(false);
    fx.push(100);
    fx.em.end_input(fx.now, &mut fx.eff).unwrap();

    assert_eq!(fx.sizes().iter().sum::<u32>(), 100);
    assert_eq!(fx.finals().iter().filter(|f| **f).count(), 1);
    assert_eq!(fx.finals().last(), Some(&true));
    assert!(sent(&fx.eff).iter().all(|a| a.segment_id.is_empty()));
    assert_eq!(fx.em.num_segments(), 1);
    assert!((fx.em.pushed_duration(-1) - 0.1).abs() < 1e-9);
}

// Catches: words pushed after the last push lost when the segment ends, on either ending: the
// buffered audio or the silence marker (the transcript would stop one word short).
#[test]
fn words_pushed_after_the_last_audio_go_out_with_the_segments_end() {
    for flushed in [false, true] {
        let mut fx = Fx::streaming("s");
        fx.push(100);
        if flushed {
            fx.flush(); // nothing is left but the marker to carry them
        }
        std::mem::take(&mut fx.eff);
        fx.em.push_timed_transcript([TranscriptChunk::new("bye")]).unwrap();
        fx.end_segment();
        let words: Vec<&str> = sent(&fx.eff)
            .iter()
            .flat_map(|a| a.timed_transcript.iter().map(|w| w.text.as_str()))
            .collect();
        assert_eq!(words, ["bye"], "flushed={flushed}");
        assert_eq!(fx.finals().last(), Some(&true));
    }
}

// --------------------------------------------------------------------- timing

// Catches: the 10 ms silence marker counted as pushed audio, so a retry resumes past real speech
// and the TTS metrics overstate the audio produced.
#[test]
fn pushed_duration_counts_the_providers_audio_but_not_the_silence_marker() {
    let mut fx = Fx::streaming("s");
    fx.push(50).flush().end_segment();
    assert_eq!(fx.sizes().iter().sum::<u32>(), 50 + SR / 100, "the premise: the marker went out");
    assert!((fx.em.pushed_duration(0) - 0.05).abs() < 1e-9);
}

// --------------------------------------------------------------------- channels

// Catches: the tail split or the marker sized without the channel count: a stereo frame torn
// between left and right swaps the channels for the rest of the segment.
#[test]
fn stereo_frames_hold_whole_sample_frames_and_keep_every_byte_in_order() {
    let mut em = AudioEmitter::default();
    em.initialize("req", SR, 2, FRAME_MS, true).unwrap();
    em.start_segment("s").unwrap();
    let mut eff = Effects::new();
    let input: Vec<u8> = (0..300 * 4).map(|i| (i % 251) as u8).collect();
    for chunk in input.chunks(7) {
        em.push(Ts::ZERO, chunk, &mut eff).unwrap();
    }
    em.flush(Ts::ZERO, &mut eff).unwrap();
    em.end_segment(Ts::ZERO, &mut eff).unwrap();

    let frames = sent(&eff);
    assert!(frames.iter().all(|a| a.frame.data.len() == a.frame.samples_per_channel as usize * 4));
    let (audio, marker) = frames.split_at(frames.len() - 1);
    let out: Vec<u8> = audio.iter().flat_map(|a| a.frame.data.iter().copied()).collect();
    assert_eq!(out, input, "every byte once, in order");
    assert_eq!(*marker[0].frame.data, [0u8; (SR / 100) as usize * 4], "10 ms of stereo silence");
}

// --------------------------------------------------------------------- interruption

// Catches: stopping (an interruption) releasing the held tail or sending a final marker, so the
// agent keeps talking after the user barged in.
#[test]
fn after_stop_mid_segment_nothing_more_is_sent() {
    let mut fx = Fx::streaming("s");
    fx.push(100);
    let delivered = fx.sizes().len();
    fx.em.stop(&mut fx.eff);
    fx.push(100).flush().fire_drift().end_segment();
    fx.em.end_input(fx.now, &mut fx.eff).unwrap();
    assert_eq!(fx.sizes().len(), delivered, "the held and buffered audio is dropped");
    assert!(fx.finals().iter().all(|f| !f), "and no final marker follows an interruption");
}

// Catches (review21): a streaming report made while no segment is open raising `NoSegment`, or
// lost. It rides on the next segment's final frame, added to that segment's own report.
#[test]
fn usage_reported_between_segments_rides_on_the_next_one() {
    let mut fx = Fx::new(true);
    fx.em.report_usage(None, Some(3)).unwrap();
    fx.em.start_segment("seg").unwrap();
    fx.push(4800);
    fx.em.report_usage(None, Some(4)).unwrap();
    fx.end_segment();
    let reported: Vec<_> = sent(&fx.eff).iter().map(|a| a.characters_count).collect();
    let (last, rest) = reported.split_last().unwrap();
    assert!(rest.iter().all(Option::is_none));
    assert_eq!(*last, Some(7));
}
