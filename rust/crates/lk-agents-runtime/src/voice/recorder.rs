//! Installs the recorder's keyed observers on the executor's admitted outputs.
use crate::voice::session::SessionShared;
use lk_agents_abi::resources::ResourceId;
use lk_agents_abi::voice::io::PlaybackObserver;
use lk_agents_voice::recorder_io::RecorderPlaybackObserver;
use std::collections::BTreeMap;
use std::sync::Arc;

#[derive(Default)]
pub(super) struct RecorderOutputs(BTreeMap<ResourceId, Arc<dyn PlaybackObserver>>);

impl RecorderOutputs {
    pub(super) fn detach(&mut self, shared: &SessionShared, output: Option<ResourceId>) {
        let removed: Vec<_> = match output {
            Some(id) => self.0.remove(&id).map(|observer| vec![(id, observer)]).unwrap_or_default(),
            None => std::mem::take(&mut self.0).into_iter().collect(),
        };
        for (id, observer) in removed {
            if let Some(output) = shared.resources.audio_output(id) {
                output.tracker().unobserve(&observer);
            }
        }
    }

    pub(super) fn attach(&mut self, shared: &SessionShared, id: ResourceId) {
        let Some(tap) = shared.recorder_tap() else { return };
        let Some(output) = shared.resources.audio_output(id) else { return };
        let observer: Arc<dyn PlaybackObserver> = Arc::new(RecorderPlaybackObserver::new(tap, id));
        if let Some(old) = self.0.insert(id, observer.clone()) {
            output.tracker().unobserve(&old);
        }
        output.tracker().observe(observer);
    }
}
