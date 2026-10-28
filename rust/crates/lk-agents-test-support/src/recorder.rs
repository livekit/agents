//! Recorder observations for native session conformance tests.
use std::sync::Mutex;

use lk_agents_abi::AudioFrame;
use lk_agents_abi::recorder::{AudioRecorder, RecordingFailure};

/// Counts actual recorder callbacks without synthesizing any playback observations.
#[derive(Default)]
pub struct RecorderProbe(Mutex<[usize; 3]>);

impl RecorderProbe {
    pub fn playback_counts(&self) -> [usize; 3] {
        *self.0.lock().unwrap()
    }
}

impl AudioRecorder for RecorderProbe {
    fn on_input_frame(&self, _: f64, _: &AudioFrame) {}
    fn capture_frame(&self, _: f64, _: &AudioFrame) {}
    fn on_playback_started(&self, _: f64) {
        self.0.lock().unwrap()[0] += 1;
    }
    fn on_playback_progressed(&self, _: f64, _: f64, _: f64) {
        self.0.lock().unwrap()[1] += 1;
    }
    fn on_playback_finished(&self, _: f64, _: f64) {
        self.0.lock().unwrap()[2] += 1;
    }
    fn flush(&self, _: f64) -> Result<(), RecordingFailure> {
        Ok(())
    }
}
