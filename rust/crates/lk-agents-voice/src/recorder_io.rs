//! Recorder attachment facts; VoiceDriver coordinates flush deadlines and execution.
use lk_agents_abi::resources::ResourceId;

pub(crate) const RECORDER_FLUSH: std::time::Duration = std::time::Duration::from_millis(2500);

pub(crate) struct RecorderIo {
    initial: Option<ResourceId>,
    attached: Option<ResourceId>,
}

impl RecorderIo {
    pub fn new(initial: Option<ResourceId>) -> Self {
        Self { initial, attached: None }
    }
    pub fn take_initial(&mut self) -> Option<ResourceId> {
        self.initial.take()
    }
    pub fn resource(&self) -> Option<ResourceId> {
        self.attached.or(self.initial)
    }
    pub fn attach(&mut self, recorder: ResourceId) {
        self.attached = Some(recorder);
    }
    #[must_use = "a detached recorder must be released by the runtime"]
    pub fn detach(&mut self) -> bool {
        self.attached.take().is_some()
    }
    pub fn is_attached(&self) -> bool {
        self.attached.is_some()
    }
}
