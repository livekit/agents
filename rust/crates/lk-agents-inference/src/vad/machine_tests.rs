use super::*;

/// The machine with every window answered at once: each scores the next scripted
/// probability, measured at `duration` seconds.
struct Scripted {
    m: VadMachine,
    probabilities: VecDeque<f32>,
    duration: f64,
}

impl Scripted {
    fn push(&mut self, frame: AudioFrame) -> Result<Vec<VadOutput>, SampleRateMismatch> {
        let mut out = Vec::new();
        let mut next = self.m.push(frame)?;
        while let Some(predict) = next {
            let probability = self.probabilities.pop_front().unwrap_or(0.0);
            let result = VadResult {
                window: predict.window,
                probability,
                inference_duration: self.duration,
            };
            let (outputs, following) = self.m.predicted(result);
            out.extend(outputs);
            next = following;
        }
        Ok(out)
    }
}

fn scripted(opts: VadOptions, probabilities: &[f32], duration: f64) -> Scripted {
    Scripted {
        m: VadMachine::new(opts),
        probabilities: probabilities.iter().copied().collect(),
        duration,
    }
}

fn timed_machine(probabilities: &[f32], duration: f64) -> Scripted {
    scripted(VadOptions::default(), probabilities, duration)
}

/// Fed at [`MODEL_SAMPLE_RATE`], so no resampler is built.
fn machine(probabilities: &[f32]) -> Scripted {
    timed_machine(probabilities, 0.001)
}

/// One window's worth of 16 kHz silence.
fn window() -> AudioFrame {
    filled_window(0)
}

/// One window of audio whose every byte is `fill`, so a buffer's *contents* are checkable.
fn filled_window(fill: u8) -> AudioFrame {
    mono(vec![fill; VAD_WINDOW_SAMPLES * 2], MODEL_SAMPLE_RATE)
}

fn kinds(out: &[VadOutput]) -> Vec<VadEventType> {
    out.iter().map(|o| o.event.kind).collect()
}

// Catches: building a resampler for audio already at the model's rate (soxr is not a no-op).
#[test]
fn no_resampler_is_built_when_the_input_is_already_at_the_model_rate() {
    let mut m = machine(&[0.0]);
    let out = m.push(filled_window(7)).unwrap();
    let done = out.iter().find(|o| o.event.kind == VadEventType::InferenceDone).unwrap();
    assert!(m.m.resampler.is_none(), "the model rate needs no conversion");
    assert!(
        done.frames.iter().all(|f| f.data.iter().all(|b| *b == 7)),
        "the frames the model scored were rewritten by a resampler that should not exist"
    );
}

// Catches: firing START_OF_SPEECH on the first loud window (`min_speech_duration` needs two).
#[test]
fn speech_starts_only_after_min_speech_duration_of_windows() {
    let mut m = machine(&[0.9, 0.9]);
    let out = m.push(window()).unwrap();
    assert_eq!(kinds(&out), [VadEventType::InferenceDone], "32ms < 50ms");
    let out = m.push(window()).unwrap();
    assert_eq!(kinds(&out), [VadEventType::InferenceDone, VadEventType::StartOfSpeech]);
}

// Catches: collapsing the two thresholds into one.
#[test]
fn hysteresis_keeps_a_turn_open_between_the_two_thresholds() {
    let mut m = machine(&[0.9, 0.9, 0.4, 0.4, 0.4]);
    m.push(window()).unwrap();
    m.push(window()).unwrap();
    for _ in 0..3 {
        let out = m.push(window()).unwrap();
        assert_eq!(kinds(&out), [VadEventType::InferenceDone], "0.4 > deactivation");
        assert!(out[0].event.speaking);
    }
}

// Catches: reporting the trailing silence as speech.
#[test]
fn end_of_speech_subtracts_the_trailing_silence_from_the_speech_duration() {
    let mut probabilities = vec![0.9; 20];
    probabilities.extend(std::iter::repeat_n(0.0, 10));
    let mut m = machine(&probabilities);
    let mut end = None;
    for _ in 0..30 {
        for out in m.push(window()).unwrap() {
            if out.event.kind == VadEventType::EndOfSpeech {
                end = Some(out);
            }
        }
    }
    let end = end.expect("speech ended");
    let window_duration = VAD_WINDOW_SAMPLES as f64 / f64::from(MODEL_SAMPLE_RATE);
    assert!(end.event.silence_duration >= 0.25);
    // `pub_speech_duration` counted 28 windows by end-of-speech; subtracting the silence
    // leaves exactly the 20 loud ones.
    let expected = 20.0 * window_duration;
    assert!(
        (end.event.speech_duration - expected).abs() < 1e-9,
        "got {} expected {expected}",
        end.event.speech_duration
    );
}

// Catches: dropping the fractional carry on the input-copy count.
#[test]
fn the_input_copy_fraction_carries_across_windows() {
    // A non-integer ratio (44100 -> 16000) through real soxr; the property is the carry.
    let mut m = timed_machine(&[0.0; 20], 0.0);
    // 100 ms of 44.1 kHz audio at a time.
    let mut consumed = Vec::new();
    for _ in 0..6 {
        let frame = mono(vec![0u8; 4410 * 2], 44_100);
        for out in m.push(frame).unwrap() {
            if out.event.kind == VadEventType::InferenceDone {
                consumed.push(out.frames[0].samples_per_channel);
            }
        }
    }
    assert!(consumed.len() >= 3, "got {consumed:?}");
    // 512 * 44100/16000 = 1411.2, so the counts must alternate.
    assert!(consumed.contains(&1412), "the carry never produced a long window: {consumed:?}");
}

// Catches: a flush leaving the counters running, or a result for a window cut before the
// flush applied to the stream after it (the model's state was reset in between).
#[test]
fn vad_eot_supplied_result_vectors_ignore_results_from_before_a_flush() {
    let mut m = VadMachine::new(VadOptions::default());
    let stale = m.push(window()).unwrap().expect("a whole window is scored");
    assert_eq!(m.reset(), None, "nothing was spoken");
    let current = m.push(window()).unwrap().expect("the flushed stream scores again");
    let late = VadResult { window: stale.window, probability: 1.0, inference_duration: 0.0 };
    assert_eq!(m.predicted(late), (Vec::new(), None), "a stale result changes nothing");
    let (out, next) = m.predicted(VadResult {
        window: current.window,
        probability: 0.0,
        inference_duration: 0.0,
    });
    assert_eq!(out[0].event.samples_index, VAD_WINDOW_SAMPLES as u64, "the counters restarted");
    assert_eq!(next, None);
}

// Catches: a gap or flush during speech resetting the detector silently, leaving the session
// speaking until the next end of speech that may never come.
#[test]
fn a_reset_during_speech_ends_it_first() {
    let mut vad = machine(&[1.0; 4]);
    let out = (0..4).flat_map(|_| vad.push(window()).unwrap()).collect::<Vec<_>>();
    assert!(kinds(&out).contains(&VadEventType::StartOfSpeech));
    let ended = vad.m.reset().expect("speech in progress ends");
    assert_eq!(ended.event.kind, VadEventType::EndOfSpeech);
    assert!(!ended.event.speaking);
    assert!(ended.event.speech_duration > 0.0);
    assert_eq!(vad.m.reset(), None, "the reset left it silent");
}

// Catches: a flush at the end of input ending speech at the last scored window, so the
// incomplete last window (the tail of the speech) falls outside the turn.
#[test]
fn a_reset_during_speech_ends_after_the_unscored_tail() {
    let mut vad = machine(&[1.0; 4]);
    for _ in 0..4 {
        vad.push(filled_window(1)).unwrap();
    }
    vad.push(mono(vec![2; 53 * 2], MODEL_SAMPLE_RATE)).unwrap();
    let ended = vad.m.reset().expect("speech in progress ends");
    assert_eq!(ended.event.samples_index, 4 * VAD_WINDOW_SAMPLES as u64 + 53);
    let end = ended.event.timestamp * f64::from(MODEL_SAMPLE_RATE);
    assert!((end - (4 * VAD_WINDOW_SAMPLES + 53) as f64).abs() < 1e-6);
    let pcm = pcm_samples(&ended.frames[0].data);
    assert_eq!(pcm.len(), 4 * VAD_WINDOW_SAMPLES + 53, "the tail is part of the speech");
    assert!(pcm[4 * VAD_WINDOW_SAMPLES..].iter().all(|s| *s == 0x0202));
}

// Catches: a second window handed out while one is in flight: a recurrent model would score
// them out of order. Each window follows the previous result.
#[test]
fn windows_are_scored_one_at_a_time_in_order() {
    let mut m = VadMachine::new(VadOptions::default());
    let first = m.push(filled_window(1)).unwrap().unwrap();
    assert_eq!(m.push(filled_window(2)).unwrap(), None, "one window in flight");
    let (_, second) =
        m.predicted(VadResult { window: first.window, probability: 0.0, inference_duration: 0.0 });
    let second = second.expect("the queued window follows the result");
    assert!(first.samples.iter().all(|s| *s == 0x0101));
    assert!(second.samples.iter().all(|s| *s == 0x0202));
}

// Catches: options that size the speech buffer from garbage or invert the hysteresis
// accepted.
#[test]
fn vad_options_are_validated() {
    assert_eq!(VadOptions::default().validate(), Ok(()));
    let bad = [
        VadOptions { min_silence_duration: -0.1, ..VadOptions::default() },
        VadOptions { max_buffered_speech: f64::NAN, ..VadOptions::default() },
    ];
    for opts in bad {
        assert_eq!(opts.validate(), Err(InvalidVadOptions::Duration));
    }
    let inverted = VadOptions {
        activation_threshold: 0.3,
        deactivation_threshold: 0.4,
        ..VadOptions::default()
    };
    assert_eq!(inverted.validate(), Err(InvalidVadOptions::Threshold));
    let above = VadOptions { activation_threshold: 1.5, ..VadOptions::default() };
    assert_eq!(above.validate(), Err(InvalidVadOptions::Threshold));
}

// Catches: the speech-buffer bookkeeping (write cursor, prefix rewind, available space);
// the other tests run on silence, where a wrong buffer looks right.
#[test]
fn start_of_speech_carries_the_prefix_padding_plus_the_speech_so_far() {
    let mut probabilities = vec![0.0f32; 20];
    probabilities.extend([0.9, 0.9]);
    let mut m = machine(&probabilities);

    for _ in 0..20 {
        m.push(filled_window(0)).unwrap();
    }
    let mut onset = None;
    for _ in 0..2 {
        for out in m.push(filled_window(7)).unwrap() {
            if out.event.kind == VadEventType::StartOfSpeech {
                onset = Some(out);
            }
        }
    }
    let onset = onset.expect("speech started");
    let frame = &onset.frames[0];

    // 0.5 s of prefix padding at 16 kHz, plus the two windows that crossed
    // `min_speech_duration`.
    let prefix = (0.5 * 16_000.0) as u32;
    assert_eq!(frame.samples_per_channel, prefix + 2 * VAD_WINDOW_SAMPLES as u32);
    assert_eq!(frame.data.len(), frame.samples_per_channel as usize * 2);
    // The tail is the speech; the head is the run-up that was rewound into place.
    let speech_bytes = VAD_WINDOW_SAMPLES * 2 * 2;
    assert!(frame.data[frame.data.len() - speech_bytes..].iter().all(|&b| b == 7));
    assert!(frame.data[..frame.data.len() - speech_bytes].iter().all(|&b| b == 0));
}

// Catches: a running machine ignoring `update_options` (the stream's options read once, at
// start): with the activation threshold raised past 0.6, 0.6 windows must not start speech.
#[test]
fn a_threshold_updated_mid_stream_applies_from_the_next_window() {
    let mut m = machine(&[0.6; 4]);
    m.push(window()).unwrap();
    m.m.update_options(VadOptions { activation_threshold: 0.9, ..VadOptions::default() });
    for _ in 0..3 {
        assert_eq!(kinds(&m.push(window()).unwrap()), [VadEventType::InferenceDone]);
    }
}

// Catches: the speech buffer keeping its first size, or a grown buffer staying "full".
#[test]
fn a_buffer_updated_mid_stream_is_resized_and_may_fill_again() {
    let none = VadOptions {
        max_buffered_speech: 0.0,
        prefix_padding_duration: 0.0,
        ..VadOptions::default()
    };
    let mut m = scripted(none, &[0.9; 4], 0.0);
    m.push(window()).unwrap();
    assert!(m.m.speech_buffer_max_reached, "a zero-length buffer is full at once");

    m.m.update_options(VadOptions { max_buffered_speech: 1.0, ..none });
    assert_eq!(m.m.speech_buffer.len(), 16_000);
    assert!(!m.m.speech_buffer_max_reached);
    m.push(window()).unwrap();
    assert_eq!(
        m.m.speech_buffer_index, VAD_WINDOW_SAMPLES,
        "the grown buffer took the next window"
    );
}

// Catches: a mismatched sample rate silently accepted or dropped.
#[test]
fn a_frame_at_a_different_sample_rate_is_rejected() {
    let mut m = machine(&[0.0]);
    m.push(window()).unwrap();
    let err = m.push(mono(vec![0u8; 64], 48_000)).expect_err("mismatched rate");
    assert_eq!(err, SampleRateMismatch { expected: 16_000, got: 48_000 });
}

// Catches: dropping the slow-inference signal.
#[test]
fn slow_inference_is_reported_with_its_accumulated_lag() {
    let mut m = timed_machine(&[0.0; 4], 0.3);
    let out = m.push(window()).unwrap();
    assert!(out[0].slow_inference, "300ms > the 200ms threshold");
    let window_duration = VAD_WINDOW_SAMPLES as f64 / f64::from(MODEL_SAMPLE_RATE);
    assert!((out[0].extra_inference_time - (0.3 - window_duration)).abs() < 1e-9);
    // ...and the lag accumulates rather than being reported per call.
    let out = m.push(window()).unwrap();
    assert!(out[0].extra_inference_time > 0.5);
}

/// However audio is partitioned and at any input rate, the model only scores whole windows.
/// https://github.com/livekit/agents/issues/402 https://github.com/livekit/agents/issues/405
#[test]
fn arbitrary_partitions_only_reach_supported_inference_windows() {
    for rate in [8_000u32, 16_000, 24_000, 44_100, 48_000] {
        let mut m = VadMachine::new(VadOptions::default());
        let mut scored = Vec::new();
        let mut pushed = 0usize;
        for (i, size) in
            [1usize, 37, 160, 480, 511, 1, 2048, 999, 13].iter().cycle().take(60).enumerate()
        {
            let frame = mono(vec![(i % 7) as u8; size * 2], rate);
            pushed += size;
            let mut next = m.push(frame).expect("same rate");
            while let Some(predict) = next {
                scored.push(predict.samples.len());
                let result =
                    VadResult { window: predict.window, probability: 0.0, inference_duration: 0.0 };
                next = m.predicted(result).1;
            }
        }
        assert!(!scored.is_empty(), "{rate} Hz: {pushed} samples produced no window");
        assert!(
            scored.iter().all(|n| *n == VAD_WINDOW_SAMPLES),
            "{rate} Hz: windows of {scored:?}"
        );
    }
}
