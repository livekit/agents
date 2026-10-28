//! Playback boundaries and observation storage for keyed output conformance tests.
use std::sync::Arc;

#[derive(Default)]
pub struct MissingPlaybackReports(
    pub Arc<lk_agents_abi::voice::io::PlaybackTracker>,
    pub std::sync::OnceLock<lk_agents_abi::voice::io::PlaybackSegment>,
);

impl lk_agents_abi::voice::io::AudioOutput for MissingPlaybackReports {
    fn label(&self) -> &str {
        "missing playback reports"
    }
    fn tracker(&self) -> &Arc<lk_agents_abi::voice::io::PlaybackTracker> {
        &self.0
    }
    fn capabilities(&self) -> lk_agents_abi::voice::io::AudioOutputCapabilities {
        Default::default()
    }
    fn capture<'a>(
        &'a self,
        _: &'a lk_agents_abi::AudioFrame,
    ) -> lk_agents_abi::BoxFuture<'a, lk_agents_abi::voice::io::Result<()>> {
        Box::pin(async {
            self.1.set(self.0.current_segment().expect("admitted capture")).unwrap();
            Ok(())
        })
    }
    fn flush_segment(&self) {}
    fn clear_buffer(&self) {}
}

/// Observation storage shared with ABI unit tests, whose local event type is distinct from the
/// dependency build of ABI used by this fixture crate.
pub struct PlaybackReports<T>(std::sync::Mutex<Vec<T>>);

impl<T> Default for PlaybackReports<T> {
    fn default() -> Self {
        Self(std::sync::Mutex::new(Vec::new()))
    }
}

impl<T> PlaybackReports<T> {
    pub fn record(&self, event: T) {
        self.0.lock().unwrap().push(event);
    }

    pub fn event_count(&self) -> usize {
        self.0.lock().unwrap().len()
    }
}

/// A physical output that latches the operation key before accepting each frame. The test
/// reports real progress and completion through its tracker using that captured key.
pub struct KeyedPlayback {
    tracker: Arc<lk_agents_abi::voice::io::PlaybackTracker>,
    captured: std::sync::Mutex<Option<lk_agents_abi::voice::io::PlaybackSegment>>,
}

impl Default for KeyedPlayback {
    fn default() -> Self {
        Self { tracker: Arc::default(), captured: std::sync::Mutex::new(None) }
    }
}

impl KeyedPlayback {
    pub fn segment(&self) -> lk_agents_abi::voice::io::PlaybackSegment {
        self.captured.lock().unwrap().expect("the physical output accepted a frame")
    }
}

impl lk_agents_abi::voice::io::AudioOutput for KeyedPlayback {
    fn label(&self) -> &str {
        "keyed playback"
    }
    fn tracker(&self) -> &Arc<lk_agents_abi::voice::io::PlaybackTracker> {
        &self.tracker
    }
    fn capabilities(&self) -> lk_agents_abi::voice::io::AudioOutputCapabilities {
        lk_agents_abi::voice::io::AudioOutputCapabilities { pause: true }
    }
    fn capture<'a>(
        &'a self,
        _: &'a lk_agents_abi::AudioFrame,
    ) -> lk_agents_abi::BoxFuture<'a, lk_agents_abi::voice::io::Result<()>> {
        Box::pin(async move {
            *self.captured.lock().unwrap() = self.tracker.current_segment();
            Ok(())
        })
    }
    fn flush_segment(&self) {}
    fn clear_buffer(&self) {}
}
