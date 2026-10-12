//! Tests for [`AudioEmitter`](crate::AudioEmitter) (`tts/tts.py:835-1344`), from
//! `agents/tests/test_tts.py:799-1110` on a virtual clock.
//!
//! The fixture is the reference's: 1 kHz mono, `frame_size_ms=100`, so a 100 ms frame is 100
//! samples and the tail is 10. `AudioByteStream` is progressive: a 100-sample push comes out as
//! 20 + 40 + (40 held).

use crate::TimedString;
use crate::Ts;

use super::{AudioEmitter, Effects, EmitterError, HostEffect};

const SR: u32 = 1000;
const CH: u16 = 1;
const FRAME_MS: u32 = 100;
/// `_TAIL_SAMPLES` at 1 kHz.
const TAIL: u32 = 10;

fn secs(s: f64) -> Ts {
    Ts::from_nanos((s * 1e9) as u64)
}

/// Every frame sent, in order.
fn sent(eff: &Effects) -> Vec<&crate::SynthesizedAudio> {
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
        let mut em = AudioEmitter::new("test");
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

// Catches: arming the drift flush during the progressive ramp-up (`sent_duration > 0.15`).
#[test]
fn the_drift_timer_stays_disarmed_below_150ms_of_sent_audio() {
    let mut fx = Fx::streaming("s");
    fx.push(100).push(100); // 10 + 40 + 80 samples delivered = 0.13 s
    assert_eq!(fx.sizes(), vec![10, 40, 80]);
    assert_eq!(armed_at(&fx.eff), None, "0.13 s is still inside the ramp");

    fx.push(100); // +100 samples => 0.23 s
    assert!(armed_at(&fx.eff).is_some(), "past 0.15 s the watchdog must arm");
}

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

// Catches: counting anything but `start_segment()` calls as segments.
#[test]
fn num_segments_counts_starts_including_ones_that_produced_no_audio() {
    let mut fx = Fx::new(true);
    fx.em.start_segment("a").unwrap();
    fx.push(100).end_segment();
    fx.em.start_segment("b").unwrap();
    fx.end_segment(); // no audio at all
    assert_eq!(fx.em.num_segments(), 2);
}

// Catches: overlapping segments going unnoticed, the streaming-only guard on the segment API
// dropped, `initialize()` callable twice, a push before it ignored, or `stop()` requiring it.
#[test]
fn misuse_of_the_lifecycle_is_refused() {
    let mut fx = Fx::streaming("a");
    fx.push(100);
    assert_eq!(fx.em.start_segment("b").unwrap_err(), EmitterError::SegmentAlreadyOpen);
    assert_eq!(fx.em.initialize("req2", SR, CH, FRAME_MS, true).unwrap_err(), EmitterError::AlreadyStarted);

    let mut fx = Fx::new(false);
    assert_eq!(fx.em.start_segment("x").unwrap_err(), EmitterError::StartSegmentNotStreaming);
    assert_eq!(fx.em.end_segment(fx.now, &mut fx.eff).unwrap_err(), EmitterError::EndSegmentNotStreaming);

    let mut em = AudioEmitter::new("test");
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

    let mut blank = AudioEmitter::new("test");
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

// Catches: a silence marker or a debug dump for a segment that opened its byte stream (a zero-byte
// push) but produced no audio.
#[test]
fn a_segment_that_opened_a_byte_stream_but_produced_no_audio_stays_silent() {
    let mut fx = Fx::streaming("s");
    fx.em.set_dump(true);

    fx.em.push(fx.now, &[], &mut fx.eff).unwrap();
    fx.end_segment();

    assert!(sent(&fx.eff).is_empty(), "no 10 ms click for a segment that never spoke");
    assert!(fx.em.take_dumps().is_empty(), "and no empty dump file for it either");
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

// Catches: the dropped-byte counter folded with anything but `+`, and `take_dropped_bytes`
// answering a constant.
#[test]
fn a_tail_that_is_not_a_whole_sample_is_dropped_and_counted() {
    // Through `flush`.
    let mut fx = Fx::streaming("s");
    fx.em.push(fx.now, &[0xff, 0xff, 0xff], &mut fx.eff).unwrap(); // 1.5 samples at 16-bit mono
    fx.flush();
    assert!(sent(&fx.eff).is_empty(), "a torn sample is discarded, never emitted");
    assert_eq!(fx.em.take_dropped_bytes(), 3);
    assert_eq!(fx.em.take_dropped_bytes(), 0, "reading it is destructive");

    // ...and through `end_segment`, which has its own copy of the fold.
    let mut fx = Fx::streaming("s");
    fx.em.push(fx.now, &[0xff, 0xff, 0xff, 0xff, 0xff], &mut fx.eff).unwrap();
    fx.end_segment();
    assert!(sent(&fx.eff).is_empty());
    assert_eq!(fx.em.take_dropped_bytes(), 5);
}

// --------------------------------------------------------------------- the 150 ms floor

// Catches: the drift floor written as `>=`; 20 ms frames land exactly on 0.150 s.
#[test]
fn exactly_150ms_of_sent_audio_is_not_yet_past_the_floor() {
    let mut em = AudioEmitter::new("test");
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
    fx.em.push_timed_transcript([TimedString::new("hello")]).unwrap();
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
    fx.em.push_timed_transcript([TimedString::new("x")]).unwrap();
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

// --------------------------------------------------------------------- LK_DUMP_TTS

// Catches: an `LK_DUMP_TTS` switch connected to nothing (`test_tts.py:801,989` monkeypatch it),
// or one segment's debug audio leaking into the next file.
#[test]
fn the_dump_switch_actually_gates_collection() {
    let mut off = Fx::streaming("s");
    off.push(100).end_segment();
    assert!(off.em.take_dumps().is_empty(), "off by default, nothing retained");

    let mut on = Fx::streaming("s");
    on.em.set_dump(true);
    on.push(100).end_segment();
    let dumps = on.em.take_dumps();
    assert_eq!(dumps.len(), 1);
    assert_eq!(dumps[0].segment_id, "s");
    assert_eq!(
        dumps[0].frames.iter().map(|f| f.samples_per_channel).sum::<u32>(),
        100,
        "the real audio, and not the synthetic marker"
    );
    assert!(on.em.take_dumps().is_empty(), "draining is destructive");

    on.em.start_segment("b").unwrap();
    on.push(200).end_segment();
    let dumps = on.em.take_dumps();
    assert_eq!(dumps.len(), 1);
    assert_eq!(dumps[0].frames.iter().map(|f| f.samples_per_channel).sum::<u32>(), 200);
}

// Catches: a WAV header that disagrees with its payload.
#[test]
fn the_dump_wav_header_describes_its_frames() {
    let mut fx = Fx::streaming("s");
    fx.em.set_dump(true);
    fx.push(100).end_segment();
    let d = fx.em.take_dumps().remove(0);
    let wav = d.wav_bytes();
    let pcm: Vec<u8> = d.frames.iter().flat_map(|f| f.data.iter().copied()).collect();
    assert_eq!(pcm.len(), 200, "100 samples of 16-bit mono");
    assert_eq!(wav.len(), 44 + pcm.len());
    assert_eq!(&wav[0..4], b"RIFF");
    assert_eq!(u32::from_le_bytes(wav[4..8].try_into().unwrap()), 36 + 200);
    assert_eq!(&wav[8..16], b"WAVEfmt ");
    assert_eq!(u16::from_le_bytes(wav[20..22].try_into().unwrap()), 1, "PCM");
    assert_eq!(u16::from_le_bytes(wav[22..24].try_into().unwrap()), CH);
    assert_eq!(u32::from_le_bytes(wav[24..28].try_into().unwrap()), SR);
    assert_eq!(u32::from_le_bytes(wav[28..32].try_into().unwrap()), SR * 2, "byte rate");
    assert_eq!(u16::from_le_bytes(wav[32..34].try_into().unwrap()), 2, "block align");
    assert_eq!(u16::from_le_bytes(wav[34..36].try_into().unwrap()), 16, "bits per sample");
    assert_eq!(&wav[36..40], b"data");
    assert_eq!(u32::from_le_bytes(wav[40..44].try_into().unwrap()), 200);
    assert_eq!(&wav[44..], &pcm[..]);
}
