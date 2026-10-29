//! Gives the real runtime queue owner to the deterministic harness.
//! The harness chooses ready observations and task turns; Runtime executes every effect over
//! the same injected provider/SDK ports and session Clock/Spawner as production.

use lk_agents_abi::resources::Resources;
use lk_agents_abi::{BoxFuture, Spawner, Stt, Vad};
use lk_agents_runtime::voice::{Executor, Runtime};
use lk_agents_voice::agent_session::{
    DriverError, Now, OpeningLegs, QueueOwner, SessionCores, SessionExecutor, Timebase,
};
use std::sync::Arc;
use tokio::sync::oneshot;

pub(crate) struct Handover {
    factory: Executor,
    run: Option<oneshot::Sender<Runtime>>,
}

impl Handover {
    pub(crate) fn new(
        run: oneshot::Sender<Runtime>,
        time: Timebase,
        spawner: Arc<dyn Spawner>,
    ) -> Self {
        Self { factory: Executor::new(time, spawner), run: Some(run) }
    }
}

impl SessionExecutor for Handover {
    fn stt_adapter(&self) -> lk_agents_voice::agent_session::SttAdapter {
        self.factory.stt_adapter()
    }
    fn tts_adapter(&self) -> lk_agents_voice::agent_session::TtsAdapter {
        self.factory.tts_adapter()
    }
    fn open_legs<'a>(
        &'a mut self,
        resources: &'a Resources,
        stt: Option<Arc<dyn Stt>>,
        vad: Option<Arc<dyn Vad>>,
    ) -> BoxFuture<'a, Result<OpeningLegs, DriverError>> {
        self.factory.open_legs(resources, stt, vad)
    }
    fn own(self: Box<Self>, cores: SessionCores) -> Box<dyn QueueOwner> {
        let Self { factory, run } = *self;
        Box::new(Running { runtime: Some(factory.into_runtime(cores)), run })
    }
}

struct Running {
    runtime: Option<Runtime>,
    run: Option<oneshot::Sender<Runtime>>,
}

impl QueueOwner for Running {
    fn open(&mut self, now: Now) {
        if let Some(runtime) = &mut self.runtime {
            runtime.open(now);
        }
    }
    fn run(mut self: Box<Self>) -> Option<BoxFuture<'static, ()>> {
        if let (Some(runtime), Some(run)) = (self.runtime.take(), self.run.take()) {
            let _ = run.send(runtime);
        }
        None
    }
}
