use super::*;

#[allow(clippy::disallowed_methods, reason = "S0: runtime migration")]
// `HalvingResampler`, `Factory` and `Counting` are gone with the trait pair (PLAN.md F1).
// Most of the tests below never touched them: they run at the recording rate, where
// `Track::resample` short-circuits before the resampler exists. The two that DID are
// re-derived over real soxr below, and the properties they pin are unchanged -- the tail
// belongs to the run that ended, and a rate change starts a new resampler.

fn mono(samples: &[i16], rate: u32) -> PcmFrame {
    PcmFrame::new(samples.to_vec(), rate, 1)
}

/// The whole point of the timeline: a gap between two placements is *silence*, not the second
/// placement moved earlier. Concatenating instead is what #6921 replaced, and the symptom was
/// a recording where the agent answers before the user finishes.
#[test]
fn a_gap_between_runs_is_written_as_silence() {
    let mut t = Track::new(10, 100.0);
    t.push(100.0, &mono(&[32767, 32767], 10));
    t.push(100.5, &mono(&[-32768], 10));

    let block = t.take(0, 8);
    assert_eq!(block.len(), 8);
    assert!(block[0] > 0.9 && block[1] > 0.9);
    assert_eq!(&block[2..5], &[0.0, 0.0, 0.0], "the 0.3 s gap must be silent");
    assert_eq!(block[5], -1.0, "the second run lands at its own timestamp");
}

/// Within tolerance the run keeps counting its own samples, so per-frame jitter cannot
/// accumulate into a slide. Re-anchoring on every frame would drift the channel by whatever
/// the capture clock is doing.
#[test]
fn small_jitter_does_not_re_anchor_the_run() {
    let mut t = Track::new(10, 0.0);
    t.push(0.0, &mono(&[1000], 10));
    // 0.1 s per frame; this one claims to start 0.09 s late, inside RESYNC_TOLERANCE
    t.push(0.19, &mono(&[2000], 10));
    let block = t.take(0, 4);
    assert!(block[1] > 0.0, "the second sample stayed contiguous at index 1");
    assert_eq!(block[2], 0.0);

    let mut t = Track::new(10, 0.0);
    t.push(0.0, &mono(&[1000], 10));
    t.push(0.21, &mono(&[2000], 10)); // 0.11 s late: past tolerance, re-anchor
    let block = t.take(0, 4);
    assert_eq!(block[1], 0.0);
    assert!(block[2] > 0.0, "re-anchored to its own timestamp");

    // exactly at the tolerance still counts as continuous: the comparison is `>`, and a
    // capture clock that drifts by precisely the tolerance every frame is the case where an
    // inclusive bound would re-anchor forever and quantise the channel to frame boundaries
    let mut t = Track::new(1000, 0.0);
    t.push(0.0, &mono(&[1000; 100], 1000)); // 0.1 s, so the run expects 0.1 next
    t.push(0.2, &mono(&[2000; 100], 1000)); // 0.1 s late, exactly RESYNC_TOLERANCE
    let block = t.take(0, 300);
    assert!(block[100] > 0.0, "extended the run at sample 100");
    assert_eq!(block[200], 0.0, "and did not re-anchor to sample 200");
}

/// A resampler's tail belongs to the run that just ended, not to the one about to start.
/// Flushing it after the re-anchor would place the old audio at the new timestamp -- a
/// millisecond of the previous utterance teleported into the next one.
///
/// Real soxr now, at rates it will actually accept. The double emitted one sample per pair
/// and held the odd one, so the old assertions counted its samples; what is asserted here is
/// the *property* -- the second run starts five seconds later, so nothing the first run owed
/// may appear there.
#[test]
fn the_resampler_tail_is_flushed_into_the_run_that_ended() {
    let mut t = Track::new(16_000, 0.0);
    let loud: Vec<i16> = (0..1600).map(|i| if i % 2 == 0 { 20_000 } else { -20_000 }).collect();
    t.push(0.0, &mono(&loud, 48_000));
    // Five seconds later: a new run, and the first one's tail must already be behind it.
    t.push(5.0, &mono(&loud, 48_000));

    // 1600 samples at 48 kHz is 1/30 s, which is ~533 samples at 16 kHz.
    let first = t.take(0, 1600);
    assert!(first.iter().take(533).any(|s| *s != 0.0), "the first run is at its own start");
    assert!(
        first[600..].iter().all(|s| *s == 0.0),
        "the gap after the first run has to be silence, tail included"
    );
}

/// Audio that arrives after its place has been written is dropped and *counted*. Mixing it
/// into the current block instead would put the user's words several seconds late in the
/// recording, with nothing to say it happened.
#[test]
fn late_audio_is_dropped_and_counted() {
    let mut t = Track::new(10, 0.0);
    t.push(0.0, &mono(&[1000, 1000], 10));
    assert_eq!(t.take(5, 10), vec![0.0; 5]);
    assert_eq!(t.dropped_samples(), 2);
}

/// A placement straddling the end of a block is kept, re-anchored at the boundary, so its tail
/// appears in the next block exactly once. Dropping the remainder truncates every utterance at
/// a 2.5 s boundary; keeping it un-anchored duplicates it.
#[test]
fn a_placement_straddling_the_boundary_continues_in_the_next_block() {
    let mut t = Track::new(10, 0.0);
    // starts at sample 2, not 0, and the boundary is not a multiple of it, so an index
    // computed from the wrong end lands somewhere visible
    t.push(0.2, &mono(&[4096, 4096, 4096, 4096, 8192, 16384], 10));
    let first = t.take(0, 6);
    assert_eq!(&first[..2], &[0.0, 0.0], "silence before the run");
    assert!(first[2..].iter().all(|s| (*s - 0.125).abs() < 1e-6), "{first:?}");
    let second = t.take(6, 10);
    assert!((second[0] - 0.25).abs() < 1e-6, "the tail continues, in order: {second:?}");
    assert!((second[1] - 0.5).abs() < 1e-6, "{second:?}");
    assert_eq!(&second[2..], &[0.0, 0.0], "and does not repeat");
    assert_eq!(t.dropped_samples(), 0);
}

/// Overlapping placements are summed, not overwritten. A sink reporting the tail of one
/// segment while the next has started must not erase either.
#[test]
fn overlapping_placements_are_mixed() {
    let mut t = Track::new(10, 0.0);
    t.push(0.0, &mono(&[8192, 8192], 10));
    t.push(5.0, &mono(&[8192], 10));
    t.push(0.0, &mono(&[8192, 8192], 10));
    let block = t.take(0, 2);
    assert!((block[0] - 0.5).abs() < 1e-6, "two quarter-scale runs summed: {block:?}");
}

/// Multi-channel input is averaged down, not left-channel-only: recording only the left
/// channel of a stereo microphone loses half the room.
#[test]
fn stereo_input_is_downmixed_by_averaging() {
    let mut t = Track::new(10, 0.0);
    t.push(0.0, &PcmFrame::new(vec![16384, 0, -16384, 0], 10, 2));
    let block = t.take(0, 2);
    assert!((block[0] - 0.25).abs() < 1e-6, "{block:?}");
    assert!((block[1] + 0.25).abs() < 1e-6, "{block:?}");
}

/// The write cursor never rewinds. A `settled` estimate that goes backwards -- and it does,
/// the moment a segment starts mid-interval -- must produce nothing rather than a second copy
/// of audio already in the file.
#[test]
fn the_cursor_never_rewinds() {
    // a non-zero t0, because every sample index is relative to it
    let mut m = Mixer::new(10, 100.0);
    m.push(INPUT_CHANNEL, 100.0, &mono(&[32767; 10], 10));
    let first = m.flush_until(101.0).expect("first block");
    assert_eq!(first.len(), 20, "10 samples, stereo interleaved");
    assert!(m.flush_until(100.5).is_none());
    assert!(m.flush_until(101.0).is_none());
}

/// The user is the left channel and the agent the right. Swapping them is invisible to every
/// test that looks at one channel, and obvious to anyone who plays the file.
#[test]
fn the_user_is_left_and_the_agent_is_right() {
    let mut m = Mixer::new(10, 100.0);
    m.push(INPUT_CHANNEL, 100.0, &mono(&[16384], 10));
    m.push(OUTPUT_CHANNEL, 100.0, &mono(&[-16384], 10));
    assert_eq!(m.flush_until(100.1), Some(vec![16384, -16384]));
}

/// A stalled microphone must not hold the writer, and a segment in flight must hold it back.
/// Losing the first clamp freezes the recording; losing the second writes silence over audio
/// the sink has not yet reported.
#[test]
fn the_writer_waits_for_a_stalled_input_but_not_forever() {
    let mut r = Recorder::new(PcmRecordingEncoder::default(), 10, 100.0);
    r.on_input_frame(101.0, &mono(&[0; 4], 10)); // 0.4 s, arriving at 101.0
    assert_eq!(r.settle_until(101.5, None), 101.0, "settled follows the input");
    assert_eq!(r.settle_until(103.0, None), 102.0, "and then the stall timeout");
    assert_eq!(r.settle_until(103.0, Some(101.4)), 101.4, "a segment in flight holds it back");
    assert_eq!(r.settle_until(101.2, None), 101.0, "never past what has arrived");
}

/// A frame carries no capture timestamp, so it is taken to have *begun* a frame-duration
/// before it arrived. Anchoring it at arrival instead puts the whole user channel one frame
/// late, which is exactly the misalignment the timeline exists to avoid.
#[test]
fn an_input_frame_is_backdated_by_its_own_duration() {
    let mut r = Recorder::new(PcmRecordingEncoder::default(), 10, 100.0);
    // 0.5 s of *stereo* audio arriving at 100.7: it began at 100.2, i.e. sample 2. Stereo
    // because the duration is per channel -- reading it off the interleaved length instead
    // backdates the frame by twice as long.
    r.on_input_frame(100.7, &PcmFrame::new(vec![32767; 10], 10, 2));
    r.flush_until(100.9).expect("encoded");
    let pcm = &r.encoder().pcm;
    assert_eq!(pcm.len(), 18, "9 stereo samples");
    assert_eq!(&pcm[..4], &[0, 0, 0, 0], "silence before the frame began");
    assert!(pcm[4] > 32000, "the audio starts at sample 2: {pcm:?}");
}

/// Both channels have to reach the encoder. Each of the three hops -- input frame, output
/// report, flush -- can be dropped on its own and leave a file that plays, with one side of
/// the conversation missing.
#[test]
fn both_channels_reach_the_encoder() {
    let mut r = Recorder::new(PcmRecordingEncoder::default(), 10, 100.0);
    r.on_input_frame(100.1, &mono(&[16384], 10));
    r.on_output_placed(&Placed { started_at: 100.0, frame: PcmFrame::new(vec![-16384], 10, 1) });
    assert!(r.flush_until(100.1).expect("encoded").is_empty(), "the double buffers");
    assert_eq!(r.encoder().pcm, vec![16384, -16384]);
    assert_eq!(r.dropped_samples(), [0, 0]);
    assert_eq!(r.finish().expect("closed"), PCM_ENCODER_TRAILER, "the trailer must be written");
}

/// The dropped-sample count has to survive the two hops up to the recorder, because the
/// warning the driver logs at close is the only signal that a recording is missing audio. A
/// rollup stuck at zero turns a broken recording into a silent one.
#[test]
fn dropped_audio_is_reported_up_to_the_recorder() {
    let mut r = Recorder::new(PcmRecordingEncoder::default(), 10, 100.0);
    r.on_input_frame(101.0, &mono(&[16384; 10], 10));
    r.flush_until(102.0).expect("encoded");
    // an output report for a moment already written: three samples with nowhere to go
    r.on_output_placed(&Placed { started_at: 100.2, frame: PcmFrame::new(vec![16384; 3], 10, 1) });
    r.flush_until(103.0).expect("encoded");
    assert_eq!(r.dropped_samples(), [0, 3]);
}

/// A source that changes sample rate mid-call gets a new resampler. Keeping the old one feeds
/// it audio at a rate it was not built for, and the channel drifts for the rest of the call.
///
/// The old version counted calls to a `Counting` factory. There is no factory (PLAN.md F1),
/// so the observation is the state the count stood in for: the format the resampler was built
/// for, which is what `resample` compares to decide whether to rebuild. That is strictly closer to the thing that goes wrong -- a factory called the
/// right number of times with the wrong rates would have passed the old test.
#[test]
fn a_source_rate_change_rebuilds_the_resampler() {
    let mut t = Track::new(16_000, 0.0);
    let frame: Vec<i16> = (0..480).map(|i| (i % 100) as i16 * 100).collect();

    let built_for = |t: &Track| t.resampler.input_format().map(|f| f.sample_rate);
    t.push(0.0, &mono(&frame, 48_000));
    assert_eq!(built_for(&t), Some(48_000));
    t.push(0.01, &mono(&frame, 48_000));
    assert_eq!(built_for(&t), Some(48_000), "same rate, same resampler");

    t.push(0.02, &mono(&frame, 24_000));
    assert_eq!(
        built_for(&t),
        Some(24_000),
        "built for the NEW rate -- a rebuild with the old rates is the bug this catches"
    );
}

/// A sink that reports its own progress must not *also* get the endpoint fallback at
/// `playback_finished`, or every segment is written into the recording twice.
#[test]
fn a_reporting_sink_does_not_also_get_the_endpoint_fallback() {
    let mut p = SegmentPlacer::new();
    p.capture_frame(10.0, &mono(&[1, 2, 3, 4], 4));
    p.on_playback_started(10.0);
    let placed = p.on_playback_progressed(10.0, 0.0, 1.0).expect("reported run");
    assert_eq!(placed.frame.samples, vec![1, 2, 3, 4]);
    assert!(p.on_playback_finished(12.0, 1.0).is_none());
}

/// A sink that reports nothing -- a remote avatar worker -- has its segment placed from its
/// endpoints, anchored at `playback_started` when there was one.
#[test]
fn a_silent_sink_falls_back_to_its_segment_endpoints() {
    let mut p = SegmentPlacer::new();
    p.capture_frame(10.0, &mono(&[1, 2, 3, 4], 4));
    p.on_playback_started(9.5);
    let placed = p.on_playback_finished(12.0, 1.0).expect("endpoint fallback");
    assert_eq!(placed.started_at, 9.5);
    assert_eq!(placed.frame.samples, vec![1, 2, 3, 4]);

    // and with no `playback_started`, the finish time less the position
    let mut p = SegmentPlacer::new();
    p.capture_frame(10.0, &mono(&[1, 2], 2));
    let placed = p.on_playback_finished(12.0, 1.0).expect("endpoint fallback");
    assert_eq!(placed.started_at, 11.0);
}

/// The slice bounds are in *interleaved* samples, so they scale by the channel count. Dropping
/// the factor cuts a stereo segment to a quarter and pans what is left.
#[test]
fn the_slice_bounds_scale_by_the_channel_count() {
    let mut p = SegmentPlacer::new();
    p.capture_frame(10.0, &PcmFrame::new(vec![1, -1, 2, -2, 3, -3, 4, -4], 4, 2));
    let placed = p.on_playback_progressed(10.0, 0.5, 0.5).expect("second half");
    assert_eq!(placed.frame.samples, vec![3, -3, 4, -4]);
    assert_eq!(placed.frame.num_channels, 2);
}

/// The offset is honoured, because a jump in it is audio the sink *discarded* on a pause
/// (#6728). Assuming the reports are contiguous writes discarded audio into the file.
#[test]
fn a_jump_in_the_offset_skips_the_audio_the_sink_discarded() {
    let mut p = SegmentPlacer::new();
    p.capture_frame(10.0, &mono(&[1, 2, 3, 4, 5, 6, 7, 8], 8));
    let placed = p.on_playback_progressed(10.0, 0.5, 0.25).expect("second half");
    assert_eq!(placed.frame.samples, vec![5, 6]);
}

/// Three ways a report describes no audio, all of which must place nothing rather than an
/// empty or reversed slice: a zero duration, a report before any frame was captured (#5279),
/// and a report starting past the end of what was captured.
#[test]
fn a_report_covering_no_audio_places_nothing() {
    let mut p = SegmentPlacer::new();
    assert!(p.on_playback_progressed(10.0, 0.0, 1.0).is_none(), "nothing captured yet");
    p.capture_frame(10.0, &mono(&[1, 2], 2));
    assert!(p.on_playback_progressed(10.0, 0.0, 0.0).is_none(), "zero duration");
    assert!(p.on_playback_progressed(10.0, 5.0, 1.0).is_none(), "past the end");
    let mut stereo = SegmentPlacer::new();
    stereo.capture_frame(10.0, &PcmFrame::new(vec![1, -1, 2, -2], 2, 2));
    assert!(
        stereo.on_playback_progressed(10.0, f64::MAX, 1.0).is_none(),
        "an out-of-range stereo report cannot overflow sample arithmetic"
    );
}

/// `pending_since` is set by the first frame of a segment and cleared when it finishes -- it
/// is what holds the writer back, so a stuck `Some` freezes the recording and a premature
/// `None` writes silence over the agent.
#[test]
fn pending_since_spans_exactly_the_segment() {
    let mut p = SegmentPlacer::new();
    assert_eq!(p.pending_since(), None);
    p.capture_frame(10.0, &mono(&[1, 2], 2));
    p.capture_frame(10.5, &mono(&[3, 4], 2));
    assert_eq!(p.pending_since(), Some(10.0), "the first frame of the segment, not the last");
    p.on_playback_finished(12.0, 1.0);
    assert_eq!(p.pending_since(), None);
}

/// Catches a port that reached for `f64::round`: `round(0.5)` is `0` in Python and `1` in Rust,
/// and the recorder turns these straight into sample offsets.
#[test]
fn ties_go_to_even_like_python() {
    use super::round_ties_even_like_python as py_round;
    assert_eq!(py_round(0.5), 0.0);
    assert_eq!(py_round(1.5), 2.0);
    assert_eq!(py_round(2.5), 2.0);
    assert_eq!(py_round(-0.5), -0.0);
}

#[test]
fn playback_progress_releases_settled_segment_pcm() {
    let mut placer = SegmentPlacer::new();
    placer.capture_frame(0.0, &mono(&[1, 2, 3, 4], 4));
    assert_eq!(placer.on_playback_progressed(0.0, 0.0, 0.5).unwrap().frame.samples, [1, 2]);
    assert_eq!(placer.pcm.len(), 2, "settled PCM must not grow for a long segment");
    placer.capture_frame(1.0, &mono(&[5, 6], 4));
    assert_eq!(placer.on_playback_progressed(0.5, 0.5, 1.0).unwrap().frame.samples, [3, 4, 5, 6]);
    assert!(
        placer.on_playback_progressed(0.0, 0.0, 0.5).is_none(),
        "late repeated progress must not duplicate audio"
    );
    assert!(placer.on_playback_finished(1.5, 1.5).is_none());
}

#[test]
fn reported_progress_allows_the_recording_timeline_to_advance() {
    let mut placer = SegmentPlacer::new();
    placer.capture_frame(0.0, &mono(&[1, 2, 3, 4], 2));
    placer.on_playback_progressed(0.0, 0.0, 1.0).unwrap();
    assert_eq!(
        placer.pending_since(),
        Some(1.0),
        "played PCM must stop holding the timeline at the segment start"
    );
    placer.on_playback_progressed(1.0, 1.0, 1.0).unwrap();
    assert_eq!(placer.pending_since(), None, "no retained tail remains unplaced");
    placer.capture_frame(2.5, &mono(&[5], 2));
    assert_eq!(placer.pending_since(), Some(2.5));
}
