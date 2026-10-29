//! `_core.HostRealtimeModel`: a plugin's realtime model as the core's own, its session's tasks on
//! the caller's executor and its events stamped on the host clock.

use std::sync::Arc;

use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::llm::realtime::{HostRealtime, RealtimeCapabilities};
use lk_agents_abi::python::Ambient;
use lk_agents_abi::{Realtime, RealtimeSession, Spawner, WallAnchor};
use lk_agents_runtime::inference::realtime::host;

pub struct HostRealtimeModel(host::HostRealtimeModel);

impl lk_agents_abi::llm::realtime::HostRealtimeModel for HostRealtimeModel {
    fn new(model: Arc<dyn HostRealtime>) -> Self {
        let anchor = WallAnchor::of(&*lk_agents_runtime::process::host_clock());
        HostRealtimeModel(host::HostRealtimeModel::new(
            model,
            anchor,
            <Arc<dyn Spawner>>::ambient(),
        ))
    }
}

impl Realtime for HostRealtimeModel {
    fn label(&self) -> String {
        self.0.label()
    }

    fn capabilities(&self) -> RealtimeCapabilities {
        self.0.capabilities()
    }

    fn session(
        &self,
        turn_detection_disabled: bool,
    ) -> lk_agents_abi::BoxFuture<'_, Result<Arc<dyn RealtimeSession>, ProviderError>> {
        self.0.session(turn_detection_disabled)
    }
}
