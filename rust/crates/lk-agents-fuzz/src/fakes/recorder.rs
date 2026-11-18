use std::sync::{Arc, Mutex};

use lk_agents_abi::recorder::{AudioRecorder, RecordingFailure};
use lk_agents_abi::{AudioFrame, Clock, Ts};
use lk_agents_media::recorder::{
    OUTPUT_CHANNEL, OggOpusEncoder, RecordingCommand, RecordingFile, RecordingLimits,
};

use super::Registry;
use crate::record::{RecorderCall, Sink};

// The recording port: the session's tap feeds the real recording timeline (the media crate's
// `RecordingFile`, sans-I/O), written as PCM in memory. At every flush the newly written agent
// channel is held against what the sink reported played. Room-only runs additionally compare
// its PCM with the source's independent speaker timeline, including silence and unplayed tails.

/// The recording's rate, as the session's.
const RATE: u32 = 48_000;
/// A played stretch's first and last samples ramp through the resampler's filter: only its inside
/// is held to be sound. The fakes' audio is a constant tag per chunk, never near zero. The
/// recording's medium-quality resampler rings past one unit for up to 2.6 ms around a full-scale
/// step, on both sides: 3 ms covers it.
const EDGE: i64 = (RATE / 1000 * 3) as i64;

/// Keeps the timeline's interleaved PCM as written: the recording.
#[derive(Clone, Default)]
struct Pcm(Arc<Mutex<Vec<i16>>>);

impl OggOpusEncoder for Pcm {
    fn push(&mut self, pcm: &[i16]) -> lk_agents_media::Result<Vec<u8>> {
        self.0.lock().unwrap().extend_from_slice(pcm);
        Ok(Vec::new())
    }

    fn finish(self) -> lk_agents_media::Result<Vec<u8>> {
        Ok(Vec::new())
    }
}

struct Recording {
    file: RecordingFile<Pcm, std::io::Sink>,
    pcm: Pcm,
    /// What the sink reported played, as `(start, end)` call time, not yet checked through.
    played: Vec<(f64, f64)>,
    /// The playing segment: its playback start, the audio captured for it, whether the sink
    /// reported its progress.
    started: Option<f64>,
    captured: f64,
    reported: bool,
    /// The agent channel is checked up to this sample.
    checked: i64,
    late: u64,
    failure: Option<RecordingFailure>,
    finished: bool,
}

pub struct FakeRecorder {
    registry: Arc<Registry>,
    recording: Mutex<Recording>,
    speaker: Option<RoomSpeaker>,
}

struct RoomSpeaker {
    room: Arc<super::room::FakeRoom>,
    anchor: (f64, Ts),
}

impl FakeRecorder {
    pub fn new(
        registry: Arc<Registry>,
        speaker: Option<(Arc<super::room::FakeRoom>, Arc<dyn Clock>)>,
    ) -> Arc<Self> {
        let speaker = speaker.map(|(room, clock)| RoomSpeaker {
            room,
            anchor: (lk_agents_abi::unix_secs(clock.wall()), clock.now()),
        });
        Arc::new(FakeRecorder { registry, recording: Mutex::default(), speaker })
    }

    fn apply(&self, call: RecorderCall, command: RecordingCommand) {
        self.registry.record_sink(Sink::Recorder(call));
        let mut recording = self.recording.lock().unwrap();
        if recording.failure.is_some() || recording.finished {
            return;
        }
        match recording.file.apply(command, &mut || Ok(())) {
            Ok(finished) => recording.finished = finished,
            Err(error) => {
                recording.failure = Some(error);
                self.registry.record_sink(Sink::Recorder(RecorderCall::Failed));
            }
        }
    }

    /// The harness owns attached recorders as a hosting job does. Finalization writes and
    /// checks the unflushed tail; a file failure is recorded and never fails the conversation.
    pub fn finish_recording(&self) {
        self.apply(RecorderCall::Finished, RecordingCommand::FinishRecording(None));
        self.check_written();
        let recording = self.recording.lock().unwrap();
        let summary = recording.file.summary();
        if let Some(at) = summary.started_at {
            self.registry.record_sink(Sink::Recorder(RecorderCall::Origin {
                unix_micros: (at * 1_000_000.0).round() as u64,
                written_through_unix_micros: summary
                    .last_flushed
                    .map(|at| (at * 1e6).round() as u64),
            }));
        }
    }

    fn check_written(&self) {
        // A custom output has its own speaker; room PCM alone cannot judge that mixture.
        if self.registry.only_room_speaker()
            && let Some(speaker) = &self.speaker
        {
            let recording = self.recording.lock().unwrap();
            if let Some(mismatch) = speaker.check(&recording) {
                self.registry.record_sink(Sink::Recorder(mismatch));
            }
            let samples = recording.pcm.0.lock().unwrap().len() / 2;
            let samples = samples.saturating_sub(recording.checked.max(0) as usize);
            self.registry.record_sink(Sink::Recorder(RecorderCall::SpeakerChecked { samples }));
        }
        let (unheard, late) = self.recording.lock().unwrap().check();
        if let Some(at_ms) = unheard {
            self.registry.record_sink(Sink::Recorder(RecorderCall::Unheard { at_ms }));
        }
        if late > 0 {
            self.registry.record_sink(Sink::Recorder(RecorderCall::Late { samples: late }));
        }
    }
}

impl Default for Recording {
    fn default() -> Self {
        let pcm = Pcm::default();
        let file = RecordingFile::new(
            pcm.clone(),
            std::io::sink(),
            RATE,
            None,
            RecordingLimits::default().file(),
        );
        Self {
            file,
            pcm,
            played: Vec::new(),
            started: None,
            captured: 0.0,
            reported: false,
            checked: 0,
            late: 0,
            failure: None,
            finished: false,
        }
    }
}

impl RoomSpeaker {
    /// Compare identity, placement and silence against the source's own PCM, rather than the
    /// callbacks the recorder received. Ignore only `EDGE` at frame boundaries for resampling.
    fn check(&self, recording: &Recording) -> Option<RecorderCall> {
        let summary = recording.file.summary();
        let t0 = summary.started_at?;
        let pcm = recording.pcm.0.lock().unwrap();
        let end = pcm.len() / 2;
        let start = recording.checked.max(0) as usize;
        let mut expected = vec![0i16; end.saturating_sub(start)];
        let mut edges = vec![false; expected.len()];
        let at = |ts: Ts| {
            let (wall, origin) = self.anchor;
            let seconds = if ts >= origin {
                ts.saturating_duration_since(origin).as_secs_f64()
            } else {
                -origin.saturating_duration_since(ts).as_secs_f64()
            };
            ((wall + seconds - t0) * f64::from(RATE)).round() as i64
        };
        // Audio before the recorder attached is outside its promise.
        let start = start.max((at(self.anchor.1) + EDGE).max(EDGE) as usize);
        let checked = recording.checked.max(0) as usize;
        for source in self.room.local_participant().sources() {
            for (since, frame) in source.heard_audio() {
                let lo = at(since);
                let hi = lo + (frame.duration() * f64::from(RATE)).round() as i64;
                let samples = lk_agents_media::pcm_samples(&frame.data);
                let mono = lk_agents_media::downmix_mono(&samples, frame.num_channels);
                for edge in [lo, hi] {
                    for i in (edge - EDGE).max(checked as i64)..(edge + EDGE).min(end as i64) {
                        edges[i as usize - checked] = true;
                    }
                }
                for i in lo.max(checked as i64)..hi.min(end as i64) {
                    let sample =
                        ((i - lo) as u64 * u64::from(frame.sample_rate) / u64::from(RATE)) as usize;
                    expected[i as usize - checked] = mono[sample.min(mono.len() - 1)];
                }
            }
        }
        (start..end).find_map(|i| {
            let expected = expected[i - checked];
            let actual = pcm[i * 2 + OUTPUT_CHANNEL];
            (!edges[i - checked] && (i32::from(expected) - i32::from(actual)).abs() > 1).then_some(
                RecorderCall::Mismatch {
                    at_ms: i as u64 * 1000 / u64::from(RATE),
                    expected,
                    actual,
                },
            )
        })
    }
}

impl Recording {
    /// The newly written agent channel against the played stretches: the first silence inside
    /// one, and the played audio dropped as late since the last check.
    fn check(&mut self) -> (Option<u64>, u64) {
        let summary = self.file.summary();
        let (Some(t0), Some(until)) = (summary.started_at, summary.last_flushed) else {
            return (None, 0);
        };
        let late = summary.dropped_output_samples - self.late;
        self.late = summary.dropped_output_samples;
        let index = |at: f64| ((at - t0) * f64::from(RATE)).round() as i64;
        let written = index(until);
        let pcm = self.pcm.0.lock().unwrap();
        let agent = |i: i64| pcm.get(i as usize * 2 + OUTPUT_CHANNEL).copied().unwrap_or(0);
        let mut unheard = None;
        for (start, end) in &self.played {
            let (lo, hi) = (index(*start) + EDGE, index(*end) - EDGE);
            let silent = (lo.max(self.checked)..hi.min(written)).find(|i| agent(*i) == 0);
            if let Some(i) = silent {
                unheard = Some(unheard.unwrap_or(u64::MAX).min(i as u64 * 1000 / u64::from(RATE)));
            }
        }
        drop(pcm);
        self.played.retain(|(_, end)| index(*end) - EDGE > written);
        self.checked = self.checked.max(written);
        (unheard, late)
    }

    /// The segment ended: a sink that reported no progress had it placed from its endpoints.
    fn finished(&mut self, now: f64, position: f64) {
        if !self.reported && self.captured > 0.0 {
            let played = position.min(self.captured);
            let start = self.started.unwrap_or(now - position);
            self.played.push((start, start + played));
        }
        (self.started, self.captured, self.reported) = (None, 0.0, false);
    }
}

impl AudioRecorder for FakeRecorder {
    fn on_input_frame(&self, now: f64, frame: &AudioFrame) {
        self.apply(RecorderCall::InputFrame, RecordingCommand::Input(now, frame.clone()));
    }
    fn capture_frame(&self, now: f64, frame: &AudioFrame) {
        self.recording.lock().unwrap().captured += frame.duration();
        self.apply(RecorderCall::CaptureFrame, RecordingCommand::Capture(now, frame.clone()));
    }
    fn on_playback_started(&self, created_at: f64) {
        self.recording.lock().unwrap().started.get_or_insert(created_at);
        self.apply(RecorderCall::PlaybackStarted, RecordingCommand::PlaybackStarted(created_at));
    }
    fn on_playback_progressed(&self, started_at: f64, offset: f64, duration: f64) {
        {
            let mut recording = self.recording.lock().unwrap();
            recording.reported = true;
            recording.played.push((started_at, started_at + duration));
        }
        let command = RecordingCommand::Progress(started_at, offset, duration);
        self.apply(RecorderCall::PlaybackProgressed, command);
    }
    fn on_playback_finished(&self, now: f64, playback_position: f64) {
        self.recording.lock().unwrap().finished(now, playback_position);
        let command = RecordingCommand::PlaybackFinished(now, playback_position);
        let position_ms = (playback_position * 1000.0).round() as u64;
        self.apply(RecorderCall::PlaybackFinished { position_ms }, command);
    }
    fn flush(&self, now: f64) -> Result<(), RecordingFailure> {
        self.apply(RecorderCall::Flush, RecordingCommand::Flush(now));
        self.check_written();
        self.recording.lock().unwrap().failure.clone().map_or(Ok(()), Err)
    }
}

#[cfg(test)]
#[path = "recorder_tests.rs"]
mod tests;
