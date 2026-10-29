//! `_core.DuplexRealtimeModel`, `_core.FixedGate` and `_core.AdaptiveNoiseGate`: the engine over the
//! adapter's declared host views. The plugin session's updates apply one after the other, its
//! output audio and events are pulled here.

use std::marker::PhantomData;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use lk_agents_abi::errors::{HostError, ProviderError};
use lk_agents_abi::llm::duplex as abi;
use lk_agents_abi::llm::inference::InvalidPayload;
use lk_agents_abi::llm::realtime::{
    Event, GenerateReplyFailed, HostRealtimeError, RealtimeCapabilities, RealtimeError,
    RealtimeModelError,
};
use lk_agents_abi::{AbortHandle, AudioFrame, Realtime, Spawner, SpawnerExt};
use lk_agents_net::env::Env;
use tokio::sync::mpsc;

use super::session::{DuplexHost, DuplexHostSession, DuplexModel};
use super::{
    AdaptiveNoiseGate, AudioGate, DuplexCapabilities, DuplexCommand, DuplexEvent, FixedGate,
    Fragment,
};
use crate::env::{Services, Standalone};

// ------------------------------------------------------------------------------------ gates

/// A gate the host holds and the session drives: one state, whoever calls.
pub struct GateClass<G>(Arc<Mutex<G>>);

impl<G: AudioGate> GateClass<G> {
    fn new(gate: G) -> Self {
        GateClass(Arc::new(Mutex::new(gate)))
    }

    fn update(&self, frame: &AudioFrame) -> Result<bool, String> {
        self.0.lock().unwrap_or_else(|p| p.into_inner()).update(frame)
    }

    fn deactivate(&self) {
        self.0.lock().unwrap_or_else(|p| p.into_inner()).deactivate();
    }
}

impl<G: AudioGate + 'static> abi::AudioGate for GateClass<G> {
    fn update(&self, frame: AudioFrame) -> Result<bool, HostError> {
        GateClass::update(self, &frame).map_err(HostError::new)
    }

    fn deactivate(&self) -> Result<(), HostError> {
        GateClass::deactivate(self);
        Ok(())
    }
}

impl abi::FixedGate for GateClass<FixedGate> {
    fn new(
        silence: f64,
        activation_ratio: f64,
        deactivation_ratio: f64,
        min_silence_duration: f64,
    ) -> Self {
        GateClass::new(FixedGate::new(
            silence,
            activation_ratio,
            deactivation_ratio,
            min_silence_duration,
        ))
    }

    fn update(&self, frame: AudioFrame) -> Result<bool, InvalidPayload> {
        GateClass::update(self, &frame).map_err(InvalidPayload)
    }

    fn deactivate(&self) {
        GateClass::deactivate(self);
    }
}

impl abi::AdaptiveNoiseGate for GateClass<AdaptiveNoiseGate> {
    fn new(
        activation_ratio: f64,
        deactivation_ratio: f64,
        min_silence_duration: f64,
        window: f64,
    ) -> Self {
        GateClass::new(AdaptiveNoiseGate::new(
            activation_ratio,
            deactivation_ratio,
            min_silence_duration,
            window,
        ))
    }

    fn update(&self, frame: AudioFrame) -> Result<bool, InvalidPayload> {
        GateClass::update(self, &frame).map_err(InvalidPayload)
    }

    fn deactivate(&self) {
        GateClass::deactivate(self);
    }
}

/// The declared gate: one of the classes above as itself, any other host object behind its proxy.
struct DeclaredGate(Arc<dyn abi::AudioGate>);

impl AudioGate for DeclaredGate {
    fn update(&mut self, frame: &AudioFrame) -> Result<bool, String> {
        self.0.update(frame.clone()).map_err(|e| describe(&e))
    }

    fn deactivate(&mut self) {
        if let Err(error) = self.0.deactivate() {
            log::error!(target: "livekit.agents", "duplex gate deactivate failed: {error}");
        }
    }
}

/// `{type(e).__name__}: {e}`, as the output stream's failure is reported.
fn describe(e: &HostError) -> String {
    format!("{}: {}", e.kind, e.message)
}

fn provider_error(e: HostError) -> ProviderError {
    ProviderError::Local(lk_agents_abi::errors::ApiError::new(e.message))
}

// ------------------------------------------------------------------------------------ the host

/// The adapter's model.
struct HostModel {
    host: Arc<dyn abi::DuplexHost>,
    label: String,
    caps: DuplexCapabilities,
    spawner: Arc<dyn Spawner>,
}

#[async_trait::async_trait]
impl DuplexHost for HostModel {
    fn label(&self) -> String {
        self.label.clone()
    }

    fn capabilities(&self) -> DuplexCapabilities {
        self.caps
    }

    async fn open(
        &self,
    ) -> Result<(Arc<dyn DuplexHostSession>, Option<Box<dyn AudioGate>>), ProviderError> {
        let gate = self.host.gate().map_err(provider_error)?;
        let session = self.host.session().await.map_err(provider_error)?;
        let gate = gate.map(|gate| Box::new(DeclaredGate(gate)) as Box<dyn AudioGate>);
        Ok((HostSession::open(session, self.label.clone(), &self.spawner), gate))
    }
}

/// A plugin's session.
struct HostSession {
    view: Arc<dyn abi::DuplexSession>,
    lane: mpsc::UnboundedSender<DuplexCommand>,
    /// Every event, the audio and the refused replies among them; `None` once the session closed.
    events: tokio::sync::Mutex<mpsc::UnboundedReceiver<Option<DuplexEvent>>>,
    tx: mpsc::UnboundedSender<Option<DuplexEvent>>,
    spawner: Arc<dyn Spawner>,
    tasks: Mutex<Vec<AbortHandle>>,
}

impl HostSession {
    fn open(
        view: Arc<dyn abi::DuplexSession>,
        label: String,
        spawner: &Arc<dyn Spawner>,
    ) -> Arc<Self> {
        let (lane, lane_rx) = mpsc::unbounded_channel();
        let (tx, events) = mpsc::unbounded_channel();
        let session = Arc::new(HostSession {
            view: Arc::clone(&view),
            lane,
            events: tokio::sync::Mutex::new(events),
            tx: tx.clone(),
            spawner: Arc::clone(spawner),
            tasks: Mutex::new(Vec::new()),
        });
        session.spawn(Self::forward(Arc::clone(&view), tx.clone()));
        session.spawn(Self::pump_audio(Arc::clone(&view), tx.clone()));
        session.spawn(Self::run_lane(view, lane_rx, label, tx));
        session
    }

    fn spawn(&self, f: impl std::future::Future<Output = ()> + Send + 'static) {
        let mut tasks = self.tasks.lock().unwrap_or_else(|p| p.into_inner());
        tasks.retain(|t| !t.is_finished());
        tasks.push(self.spawner.spawn_unboxed(f));
    }

    /// The session callbacks' events, until the view reports the end.
    async fn forward(
        view: Arc<dyn abi::DuplexSession>,
        tx: mpsc::UnboundedSender<Option<DuplexEvent>>,
    ) {
        loop {
            let event = match view.next_event().await {
                Ok(None) => {
                    let _ = tx.send(None);
                    return;
                }
                Ok(Some(abi::DuplexEvent::Transcript(t))) => DuplexEvent::Transcript(Fragment {
                    text: t.text,
                    start_ms: t.start_ms,
                    end_ms: t.end_ms,
                }),
                Ok(Some(abi::DuplexEvent::Session(event))) => DuplexEvent::Session(event),
                // a malformed event is the view's bug; the session goes on
                Err(e) => {
                    log::warn!(target: "livekit.agents", error = e.to_string().as_str(); "duplex event refused");
                    continue;
                }
            };
            if tx.send(Some(event)).is_err() {
                return;
            }
        }
    }

    /// The model's output audio, then how it ended.
    async fn pump_audio(
        view: Arc<dyn abi::DuplexSession>,
        tx: mpsc::UnboundedSender<Option<DuplexEvent>>,
    ) {
        let error = loop {
            match view.next_audio().await {
                Ok(Some(audio)) => {
                    let event = DuplexEvent::Audio { frame: audio.frame, start_ms: audio.start_ms };
                    let _ = tx.send(Some(event));
                }
                Ok(None) => break None,
                Err(e) => break Some(describe(&e)),
            }
        };
        let _ = tx.send(Some(DuplexEvent::OutputEnded { error }));
    }

    /// The configuration and its updates, each applied once the previous one has. A session that
    /// cannot carry its startup configuration is unusable: it reports the error and closes.
    async fn run_lane(
        view: Arc<dyn abi::DuplexSession>,
        mut lane: mpsc::UnboundedReceiver<DuplexCommand>,
        label: String,
        tx: mpsc::UnboundedSender<Option<DuplexEvent>>,
    ) {
        while let Some(cmd) = lane.recv().await {
            let configure = matches!(cmd, DuplexCommand::Configure { .. });
            let applied = match cmd {
                DuplexCommand::Configure { instructions, items, tools } => {
                    view.configure(instructions, items, tools).await
                }
                DuplexCommand::UpdateInstructions(instructions) => {
                    view.update_instructions(instructions).await
                }
                DuplexCommand::AppendItems(items) => view.append_items(items).await,
                DuplexCommand::UpdateTools(names) => view.update_tools(names).await,
                DuplexCommand::PushAudio(_) | DuplexCommand::GenerateReply { .. } => continue,
            };
            let Err(e) = applied else { continue };
            if !configure {
                log::error!(target: "livekit.agents", error = e.message.as_str(); "failed to update the duplex session");
                continue;
            }
            let error = RealtimeModelError {
                label: label.clone(),
                message: e.message,
                recoverable: false,
                ..Default::default()
            };
            let _ = tx.send(Some(DuplexEvent::Session(Event::HostError(HostRealtimeError {
                error,
                timestamp: lk_agents_abi::unix_now(),
            }))));
            view.close_plugin().await;
        }
    }

    /// A reply the plugin refused is reported, so the turn waiting for it fails.
    async fn generate(
        view: Arc<dyn abi::DuplexSession>,
        req_id: String,
        instructions: Option<String>,
        tool_choice: Option<lk_agents_abi::ToolChoice>,
        tx: mpsc::UnboundedSender<Option<DuplexEvent>>,
    ) {
        if let Err(e) = view.generate(instructions, tool_choice).await {
            let error = RealtimeError { message: e.message, code: None };
            let _ = tx.send(Some(DuplexEvent::Session(Event::GenerateReplyFailed(
                GenerateReplyFailed { req_id, error },
            ))));
        }
    }
}

#[async_trait::async_trait]
impl DuplexHostSession for HostSession {
    fn send(&self, cmd: DuplexCommand) {
        match cmd {
            DuplexCommand::PushAudio(frame) => self.view.push_audio(frame),
            DuplexCommand::GenerateReply { req_id, instructions, tool_choice } => {
                let view = Arc::clone(&self.view);
                self.spawn(Self::generate(
                    view,
                    req_id,
                    instructions,
                    tool_choice,
                    self.tx.clone(),
                ));
            }
            DuplexCommand::UpdateInstructions(_)
            | DuplexCommand::AppendItems(_)
            | DuplexCommand::UpdateTools(_)
            | DuplexCommand::Configure { .. } => {
                let _ = self.lane.send(cmd);
            }
        }
    }

    async fn next_event(&self) -> Option<DuplexEvent> {
        self.events.lock().await.recv().await.flatten()
    }

    async fn aclose(&self) {
        for task in std::mem::take(&mut *self.tasks.lock().unwrap_or_else(|p| p.into_inner())) {
            task.abort();
        }
        self.view.aclose().await;
    }
}

// ------------------------------------------------------------------------------------ the model

/// `DuplexRealtimeAdapter`'s engine; `E` is the hosting process whose runtime and clock its
/// sessions run on.
pub struct NativeDuplexModel<E: Env<Services = Services> = Standalone> {
    model: Arc<DuplexModel>,
    env: PhantomData<fn() -> E>,
}

impl<E: Env<Services = Services>> abi::DuplexRealtimeModel for NativeDuplexModel<E> {
    fn new(host: Arc<dyn abi::DuplexHost>, audio_timeout: f64) -> Result<Self, HostError> {
        let label = host.label()?;
        let caps = host.capabilities()?;
        let spawner = E::spawner().map_err(HostError::new)?;
        let host = Arc::new(HostModel { host, label, caps, spawner });
        let timeout = Duration::from_secs_f64(audio_timeout.max(0.0));
        let model = DuplexModel::new(host, timeout, E::clock());
        Ok(NativeDuplexModel { model: Arc::new(model), env: PhantomData })
    }

    fn capabilities(&self) -> RealtimeCapabilities {
        Realtime::capabilities(&*self.model)
    }
}

impl<E: Env<Services = Services>> Realtime for NativeDuplexModel<E> {
    fn label(&self) -> String {
        self.model.label()
    }

    fn capabilities(&self) -> RealtimeCapabilities {
        Realtime::capabilities(&*self.model)
    }

    fn session(
        &self,
        turn_detection_disabled: bool,
    ) -> lk_agents_abi::BoxFuture<'_, Result<Arc<dyn lk_agents_abi::RealtimeSession>, ProviderError>>
    {
        self.model.session(turn_detection_disabled)
    }
}
