use std::sync::{Arc, Mutex};

use lk_agents_abi::AudioFrame;
use lk_agents_abi::recorder::{AudioRecorder, RecordingFailure};
use lk_agents_media::recorder::{
    OUTPUT_CHANNEL, OggOpusEncoder, RecordingCommand, RecordingFile, RecordingLimits,
};

use super::Registry;
use crate::record::{RecorderCall, Sink};

// The recording port: the session's tap feeds the real recording timeline (the media crate's
// `RecordingFile`, sans-I/O), written as PCM in memory. At every flush the newly written agent
// channel is held against what the sink reported played: a played stretch is never silence, and
// no played audio arrives after its place was written.

/// The recording's rate, as the session's.
const RATE: u32 = 48_000;
/// A played stretch's first and last samples ramp through the resampler's filter: only its inside
/// is held to be sound. The fakes' audio is a constant tag per chunk, never near zero.
const EDGE: i64 = (RATE / 500) as i64;

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
}

pub struct FakeRecorder {
    registry: Arc<Registry>,
    recording: Mutex<Recording>,
}

impl FakeRecorder {
    pub fn new(registry: Arc<Registry>) -> Arc<Self> {
        let pcm = Pcm::default();
        let limits = RecordingLimits::default().file();
        let file = RecordingFile::new(pcm.clone(), std::io::sink(), RATE, None, limits);
        let recording = Recording {
            file,
            pcm,
            played: Vec::new(),
            started: None,
            captured: 0.0,
            reported: false,
            checked: 0,
            late: 0,
        };
        Arc::new(FakeRecorder { registry, recording: Mutex::new(recording) })
    }

    fn apply(&self, call: RecorderCall, command: RecordingCommand) {
        self.registry.record_sink(Sink::Recorder(call));
        let mut recording = self.recording.lock().unwrap();
        // A refused command fails the recording; what was written so far is still checked.
        let _ = recording.file.apply(command, &mut || Ok(()));
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
        let (unheard, late) = self.recording.lock().unwrap().check();
        if let Some(at_ms) = unheard {
            self.registry.record_sink(Sink::Recorder(RecorderCall::Unheard { at_ms }));
        }
        if late > 0 {
            self.registry.record_sink(Sink::Recorder(RecorderCall::Late { samples: late }));
        }
        Ok(())
    }
}
