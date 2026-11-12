use super::*;
use crate::{Quality, Resampler};

// The written file against what was actually heard: every channel is decoded sample for sample
// and compared with the known signal placed where it played.

const RATE: u32 = 48_000;
const T0: f64 = 100.0;

/// Hands the PCM it is given to the writer, so the written file is the timeline itself.
struct RawEncoder;

impl OggOpusEncoder for RawEncoder {
    fn push(&mut self, pcm: &[i16]) -> crate::Result<Vec<u8>> {
        Ok(crate::pcm_bytes(pcm))
    }

    fn finish(self) -> crate::Result<Vec<u8>> {
        Ok(Vec::new())
    }
}

fn frame(samples: &[i16], rate: u32) -> AudioFrame {
    AudioFrame::new(crate::pcm_bytes(samples), rate, 1, samples.len() as u32)
}

/// A tone at `rate` that never repeats within a segment, so a shifted copy cannot pass.
fn tone(rate: u32, samples: usize, freq: f64) -> Vec<i16> {
    (0..samples)
        .map(|i| {
            let t = i as f64 / f64::from(rate);
            (8000.0 * (2.0 * std::f64::consts::PI * (freq + 40.0 * t) * t).sin()) as i16
        })
        .collect()
}

/// `pcm` at `rate` as the recording's rate, converted as one uninterrupted stream.
fn at_recording_rate(pcm: &[i16], rate: u32) -> Vec<i16> {
    if rate == RATE {
        return pcm.to_vec();
    }
    let mut resampler = Resampler::new(rate, RATE, 1, Quality::Medium).expect("resampler");
    let mut out = resampler.push(pcm).expect("pushed").to_vec();
    out.extend_from_slice(resampler.flush().expect("flushed"));
    out
}

fn index(at: f64) -> usize {
    ((at - T0) * f64::from(RATE)).round() as usize
}

/// How the sink tells the session what it played.
#[derive(Clone, Copy, Debug)]
enum Sink {
    /// Nothing until the end: the segment is placed from its endpoints.
    Silent,
    /// Progress every this many seconds of audio, as it plays.
    Reporting(f64),
}

/// One agent segment: its audio, when playback started, and how much of it played.
struct Segment {
    pcm: Vec<i16>,
    rate: u32,
    start: f64,
    played: usize,
}

impl Segment {
    fn played_seconds(&self) -> f64 {
        self.played as f64 / f64::from(self.rate)
    }

    fn end(&self) -> f64 {
        self.start + self.played_seconds()
    }
}

/// Drives a file the way the session does, in time order: each segment's frames are captured as
/// it starts, the sink reports, and the session flushes every 10 ms throughout.
fn record(segments: &[Segment], frame_samples: usize, sink: Sink) -> Vec<i16> {
    let mut timed: Vec<(f64, usize, RecordingCommand)> = Vec::new();
    for segment in segments {
        let mut order = timed.len();
        let mut at = |time: f64, command| {
            timed.push((time, order, command));
            order += 1;
        };
        for chunk in segment.pcm.chunks(frame_samples) {
            at(segment.start, RecordingCommand::Capture(segment.start, frame(chunk, segment.rate)));
        }
        at(segment.start, RecordingCommand::PlaybackStarted(segment.start));
        if let Sink::Reporting(every) = sink {
            let mut offset = 0.0;
            while offset < segment.played_seconds() {
                let duration = every.min(segment.played_seconds() - offset);
                let reported = RecordingCommand::Progress(segment.start + offset, offset, duration);
                // Never after the finish, whatever the float sum says.
                at((segment.start + offset + duration).min(segment.end()), reported);
                offset += duration;
            }
        }
        at(
            segment.end(),
            RecordingCommand::PlaybackFinished(segment.end(), segment.played_seconds()),
        );
    }
    let end = segments.iter().map(Segment::end).fold(T0, f64::max) + 2.0;
    let ticks = ((end - T0) / 0.01) as usize;
    for tick in 1..=ticks {
        timed.push((
            T0 + tick as f64 * 0.01,
            usize::MAX,
            RecordingCommand::Flush(T0 + tick as f64 * 0.01),
        ));
    }
    timed.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));

    let limits = RecordingLimits::default().file();
    let mut file = RecordingFile::new(RawEncoder, Vec::new(), RATE, Some(T0), limits);
    for (_, _, command) in timed {
        file.apply(command, &mut || Ok(())).expect("applied");
    }
    file.apply(RecordingCommand::FinishRecording(Some(end)), &mut || Ok(())).expect("finished");
    assert_eq!(file.summary().dropped_output_samples, 0, "no agent audio arrived late");
    let pcm = crate::pcm_samples(&file.writer);
    pcm.as_chunks::<2>().0.iter().map(|pair| pair[1]).collect()
}

/// `channel` is silence except for each `(start, pcm)`, sample for sample from `start`.
fn assert_channel(channel: &[i16], heard: &[(f64, Vec<i16>)], case: &str) {
    let mut want = vec![0_i16; channel.len()];
    for (start, pcm) in heard {
        let from = index(*start);
        assert!(from + pcm.len() <= want.len(), "{case}: the file ends before the audio does");
        for (i, sample) in pcm.iter().enumerate() {
            want[from + i] += sample;
        }
    }
    if let Some(i) = channel.iter().zip(&want).position(|(got, want)| got != want) {
        let at = |i: usize| T0 + i as f64 / f64::from(RATE);
        let until = channel[i..].iter().zip(&want[i..]).position(|(g, w)| g == w).map(|n| i + n);
        panic!(
            "{case}: the channel departs from what was heard at {:.4} s (sample {i}: {} != {}), \
             rejoining at {:?}",
            at(i),
            channel[i],
            want[i],
            until.map(at),
        );
    }
}

/// The agent channel is each segment's played audio from the moment its playback started.
fn assert_heard(agent: &[i16], segments: &[Segment], case: &str) {
    let heard: Vec<_> =
        segments.iter().map(|s| (s.start, at_recording_rate(&s.pcm[..s.played], s.rate))).collect();
    assert_channel(agent, &heard, case);
}

const RATES: [u32; 6] = [8_000, 16_000, 22_050, 24_000, 44_100, 48_000];

fn segment(rate: u32, start: f64, seconds: f64, played: f64, freq: f64) -> Segment {
    let pcm = tone(rate, (seconds * f64::from(rate)).round() as usize, freq);
    let played = ((played * f64::from(rate)).round() as usize).min(pcm.len());
    Segment { pcm, rate, start, played }
}

/// A 24 kHz segment lost ~20 ms just before its end: the writer passed the resampler's output
/// while its tail was still held, and the tail then arrived behind the written block.
#[test]
fn the_agent_channel_is_sample_continuous_at_every_tts_rate() {
    for rate in RATES {
        // 10 ms frames, and frame and report sizes that divide nothing evenly.
        for frame_ms in [10.0, 7.3, 21.1] {
            for sink in [Sink::Silent, Sink::Reporting(0.05), Sink::Reporting(0.0137)] {
                let frame_samples = (f64::from(rate) * frame_ms / 1000.0).round() as usize;
                let segments = [
                    segment(rate, 100.5, 1.0, 1.0, 300.0),
                    segment(rate, 103.25, 0.73, 0.73, 500.0),
                ];
                let agent = record(&segments, frame_samples, sink);
                assert_heard(
                    &agent,
                    &segments,
                    &format!("{rate} Hz, {frame_ms} ms frames, {sink:?}"),
                );
            }
        }
    }
}

/// An interrupted segment ends exactly where its playback stopped: nothing it did not play is in
/// the file, and the next segment starts at its own playback start.
#[test]
fn an_interrupted_segment_is_truncated_where_playback_stopped() {
    for rate in RATES {
        for sink in [Sink::Silent, Sink::Reporting(0.02)] {
            let segments =
                [segment(rate, 100.5, 2.0, 0.4137, 300.0), segment(rate, 101.4, 0.5, 0.5, 700.0)];
            let agent = record(&segments, (rate / 100) as usize, sink);
            assert_heard(&agent, &segments, &format!("{rate} Hz, {sink:?}"));
        }
    }
}

/// Back-to-back segments a few tens of milliseconds apart each start where they played; the
/// second is not pulled onto the end of the first.
#[test]
fn close_segments_each_start_at_their_own_playback_start() {
    for rate in [24_000, 48_000] {
        for sink in [Sink::Silent, Sink::Reporting(0.02)] {
            let segments =
                [segment(rate, 100.5, 0.5, 0.5, 300.0), segment(rate, 101.03, 0.5, 0.5, 700.0)];
            let agent = record(&segments, (rate / 100) as usize, sink);
            assert_heard(&agent, &segments, &format!("{rate} Hz, {sink:?}"));
        }
    }
}

/// A sink that paused and discarded part of its buffer: what played before the pause is where it
/// played, the discarded audio is nowhere, and what played after resumes where it resumed, even
/// when the jump is shorter than the input channel's jitter tolerance.
#[test]
fn audio_the_sink_discarded_is_skipped_and_the_rest_plays_where_it_resumed() {
    for rate in [24_000, 48_000] {
        let pcm = tone(rate, rate as usize, 300.0);
        let cut = |seconds: f64| (seconds * f64::from(rate)).round() as usize;
        let limits = RecordingLimits::default().file();
        let mut file = RecordingFile::new(RawEncoder, Vec::new(), RATE, Some(T0), limits);
        let mut apply = |command| file.apply(command, &mut || Ok(())).expect("applied");
        apply(RecordingCommand::Capture(100.5, frame(&pcm, rate)));
        apply(RecordingCommand::PlaybackStarted(100.5));
        apply(RecordingCommand::Progress(100.5, 0.0, 0.3));
        // Paused 40 ms later, 50 ms of the buffer discarded, the rest played out.
        apply(RecordingCommand::Progress(100.84, 0.35, 0.65));
        apply(RecordingCommand::PlaybackFinished(101.49, 0.95));
        apply(RecordingCommand::FinishRecording(Some(103.0)));
        assert_eq!(file.summary().dropped_output_samples, 0);
        let pcm_out = crate::pcm_samples(&file.writer);
        let agent: Vec<_> = pcm_out.as_chunks::<2>().0.iter().map(|pair| pair[1]).collect();
        let heard = [
            (100.5, at_recording_rate(&pcm[..cut(0.3)], rate)),
            (100.84, at_recording_rate(&pcm[cut(0.35)..], rate)),
        ];
        assert_channel(&agent, &heard, &format!("{rate} Hz"));
    }
}

/// The user channel is the microphone sample for sample from its first frame, after minutes of
/// call: counting the run's own samples never drifts from the input, and audio dropped upstream is
/// silence of its own length, with what follows still in place.
#[test]
fn the_user_channel_follows_the_microphone_without_drift() {
    for rate in [16_000, 44_100, 48_000] {
        let frame_samples = rate as usize / 100;
        let mic = tone(rate, rate as usize * 180, 200.0);
        let (first, gap) = (rate as usize * 60, 0.25);
        let limits = RecordingLimits::default().file();
        let mut file = RecordingFile::new(RawEncoder, Vec::new(), RATE, Some(T0), limits);
        let start = T0 + 0.5;
        let mut now = start;
        let mut next_flush = T0;
        for (i, chunk) in mic.chunks(frame_samples).enumerate() {
            if i * frame_samples == first {
                file.input_gap(gap);
                now += gap;
            }
            now += chunk.len() as f64 / f64::from(rate);
            file.apply(RecordingCommand::Input(now, frame(chunk, rate)), &mut || Ok(()))
                .expect("input");
            while next_flush + 0.1 <= now {
                next_flush += 0.1;
                file.apply(RecordingCommand::Flush(next_flush), &mut || Ok(())).expect("flush");
            }
        }
        file.apply(RecordingCommand::FinishRecording(Some(now + 1.0)), &mut || Ok(()))
            .expect("finished");
        assert_eq!(file.summary().dropped_input_samples, 0);
        let pcm = crate::pcm_samples(&file.writer);
        let user: Vec<_> = pcm.as_chunks::<2>().0.iter().map(|pair| pair[0]).collect();
        let first_heard = at_recording_rate(&mic[..first], rate);
        let resumed = start + first_heard.len() as f64 / f64::from(RATE) + gap;
        let heard = [(start, first_heard), (resumed, at_recording_rate(&mic[first..], rate))];
        assert_channel(&user, &heard, &format!("{rate} Hz"));
    }
}
