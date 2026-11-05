//! `_core.DuplexRealtimeModel`, `_core.FixedGate` and `_core.AdaptiveNoiseGate`: the engine over the
//! adapter's declared host views. The caller's audio goes into the plugin's input, its updates
//! apply one after the other, its output audio and events are pulled here.

use crate::telemetry;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use lk_agents_abi::errors::{HostError, ProviderError};
use lk_agents_abi::llm::duplex as abi;
use lk_agents_abi::llm::inference::InvalidPayload;
use lk_agents_abi::llm::realtime::{
    Event, GenerateReplyFailed, HostRealtimeError, RealtimeCapabilities, RealtimeError,
};
use lk_agents_abi::{AudioFrame, OwnedTask, Realtime, Spawner, SpawnerExt, WallAnchor};
use parking_lot::Mutex;
use tokio::sync::mpsc;

use super::session::DuplexAdapterModel;
use super::{AdaptiveNoiseGate, AudioGate, DuplexCommand, DuplexEvent, FixedGate, Fragment};
use lk_agents_net::env::ProcessEnv;

// ------------------------------------------------------------------------------------ gates

/// A gate the host holds and the session drives: one state, whoever calls.
pub struct GateClass<G>(Arc<Mutex<G>>);

impl<G: AudioGate> GateClass<G> {
    fn gate(&self) -> parking_lot::MutexGuard<'_, G> {
        self.0.lock()
    }
}

impl<G: AudioGate + 'static> abi::AudioGate for GateClass<G> {
    fn update(&self, frame: AudioFrame) -> Result<bool, HostError> {
        self.gate().update(&frame).map_err(HostError::new)
    }

    fn deactivate(&self) -> Result<(), HostError> {
        self.gate().deactivate();
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
        let gate =
            FixedGate::new(silence, activation_ratio, deactivation_ratio, min_silence_duration);
        GateClass(Arc::new(Mutex::new(gate)))
    }

    fn update(&self, frame: AudioFrame) -> Result<bool, InvalidPayload> {
        self.gate().update(&frame).map_err(InvalidPayload)
    }

    fn deactivate(&self) {
        self.gate().deactivate();
    }
}

impl abi::AdaptiveNoiseGate for GateClass<AdaptiveNoiseGate> {
    fn new(
        activation_ratio: f64,
        deactivation_ratio: f64,
        min_silence_duration: f64,
        window: f64,
    ) -> Self {
        let gate = AdaptiveNoiseGate::new(
            activation_ratio,
            deactivation_ratio,
            min_silence_duration,
            window,
        );
        GateClass(Arc::new(Mutex::new(gate)))
    }

    fn update(&self, frame: AudioFrame) -> Result<bool, InvalidPayload> {
        self.gate().update(&frame).map_err(InvalidPayload)
    }

    fn deactivate(&self) {
        self.gate().deactivate();
    }
}

/// The session's gate: the host's declared one (one of the classes above as itself, any other host
/// object behind its proxy), else the default adaptive gate. Called with no session lock held.
pub(super) struct SessionGate(Arc<dyn abi::AudioGate>);

impl SessionGate {
    pub(super) fn new(declared: Option<Arc<dyn abi::AudioGate>>) -> Self {
        Self(declared.unwrap_or_else(|| {
            Arc::new(GateClass(Arc::new(Mutex::new(AdaptiveNoiseGate::default()))))
        }))
    }

    pub(super) fn update(&self, frame: &AudioFrame) -> Result<bool, String> {
        self.0.update(frame.clone()).map_err(|e| describe(&e))
    }

    pub(super) fn deactivate(&self) {
        if let Err(error) = self.0.deactivate() {
            log::error!(target: "livekit.agents", "duplex gate deactivate failed: {error}");
        }
    }
}

/// `{type(e).__name__}: {e}`, as the output stream's failure is reported.
fn describe(e: &HostError) -> String {
    format!("{}: {}", e.kind, e.message)
}

pub(super) fn provider_error(e: HostError) -> ProviderError {
    ProviderError::Local(lk_agents_abi::errors::ApiError::new(e.message))
}

// ------------------------------------------------------------------------------------ the host

/// A plugin's session.
pub(super) struct HostSession {
    view: Arc<dyn abi::DuplexSession>,
    input: Arc<dyn abi::DuplexInputQueue>,
    lane: mpsc::UnboundedSender<DuplexCommand>,
    /// Every event, the audio and the refused replies among them; `None` once the session closed.
    events: tokio::sync::Mutex<mpsc::UnboundedReceiver<Option<DuplexEvent>>>,
    tx: mpsc::UnboundedSender<Option<DuplexEvent>>,
    /// Microseconds of output audio queued ahead of the session: its output lag.
    output_queued: Arc<AtomicU64>,
    lag: telemetry::DuplexOutputLag,
    cx: Mutex<Option<telemetry::Cx>>,
    spawner: Arc<dyn Spawner>,
    tasks: Mutex<Vec<OwnedTask>>,
}

impl HostSession {
    pub(super) fn open(
        view: Arc<dyn abi::DuplexSession>,
        label: String,
        spawner: &Arc<dyn Spawner>,
    ) -> Self {
        let (lane, lane_rx) = mpsc::unbounded_channel();
        let (tx, events) = mpsc::unbounded_channel();
        let output_queued = Arc::default();
        let id = lk_agents_abi::ProviderId::of_label(label.clone());
        let session = HostSession {
            view: Arc::clone(&view),
            input: Arc::clone(view.input()),
            lane,
            events: tokio::sync::Mutex::new(events),
            tx: tx.clone(),
            output_queued: Arc::clone(&output_queued),
            lag: telemetry::DuplexOutputLag { model: telemetry::model_labels(&id) },
            cx: Mutex::new(None),
            spawner: Arc::clone(spawner),
            tasks: Mutex::new(Vec::new()),
        };
        session.spawn(Self::forward(Arc::clone(&view), tx.clone()));
        session.spawn(Self::pump_audio(Arc::clone(&view), tx.clone(), output_queued));
        session.spawn(Self::run_lane(view, lane_rx, label, tx));
        session
    }

    fn spawn(&self, f: impl std::future::Future<Output = ()> + Send + 'static) {
        let mut tasks = self.tasks.lock();
        tasks.retain(|t| !t.is_finished());
        tasks.push(self.spawner.spawn(f));
    }

    /// The session callbacks' events, until the view reports the end.
    async fn forward(
        view: Arc<dyn abi::DuplexSession>,
        tx: mpsc::UnboundedSender<Option<DuplexEvent>>,
    ) {
        loop {
            let event = match view.next_event(&telemetry::Cx::NONE).await {
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
        output_queued: Arc<AtomicU64>,
    ) {
        let error = loop {
            match view.next_audio(&telemetry::Cx::NONE).await {
                Ok(Some(audio)) => {
                    output_queued.fetch_add(micros(&audio.frame), Ordering::Relaxed);
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
                    view.configure(&telemetry::Cx::NONE, instructions, items, tools).await
                }
                DuplexCommand::UpdateInstructions(instructions) => {
                    view.update_instructions(&telemetry::Cx::NONE, instructions).await
                }
                DuplexCommand::AppendItems(items) => {
                    view.append_items(&telemetry::Cx::NONE, items).await
                }
                DuplexCommand::UpdateTools(names) => {
                    view.update_tools(&telemetry::Cx::NONE, names).await
                }
                DuplexCommand::PushAudio(_) | DuplexCommand::GenerateReply { .. } => continue,
            };
            let Err(e) = applied else { continue };
            if !configure {
                log::error!(target: "livekit.agents", error = e.message.as_str(); "failed to update the duplex session");
                continue;
            }
            let _ = tx.send(Some(DuplexEvent::Session(Event::HostError(HostRealtimeError {
                label: label.clone(),
                message: e.message,
                recoverable: false,
                timestamp: lk_agents_net::clock::unix_now(),
                req_id: None,
                code: None,
            }))));
            view.close_plugin(&telemetry::Cx::NONE).await;
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
        if let Err(e) = view.generate(&telemetry::Cx::NONE, instructions, tool_choice).await {
            let error = RealtimeError { message: e.message, code: None };
            let _ = tx.send(Some(DuplexEvent::Session(Event::GenerateReplyFailed(
                GenerateReplyFailed { req_id, error },
            ))));
        }
    }
    /// Queued on the host, in order; the caller's audio goes into the plugin's input.
    pub(super) fn send(&self, cmd: DuplexCommand) {
        match cmd {
            DuplexCommand::PushAudio(frame) => self.input.push(frame),
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

    /// Pre-connect audio, kept in full by the plugin's input.
    pub(super) fn head(&self, frame: AudioFrame) {
        self.input.head(frame);
    }

    /// Measures the input's drops and the output lag under `cx`'s job.
    pub(super) fn observe(&self, cx: telemetry::Cx) {
        self.input.observe(&cx);
        *self.cx.lock() = Some(cx);
    }

    /// `None` once the session is closed.
    pub(super) async fn next_event(&self) -> Option<DuplexEvent> {
        let event = self.events.lock().await.recv().await.flatten();
        if let Some(DuplexEvent::Audio { frame, .. }) = &event {
            let lag = self.output_queued.fetch_sub(micros(frame), Ordering::Relaxed);
            if let Some(cx) = self.cx.lock().as_ref() {
                telemetry::measure(cx, &self.lag, lag as f64 / 1e6);
            }
        }
        event
    }

    pub(super) async fn aclose(&self) {
        // Cancelled, not awaited.
        drop(std::mem::take(&mut *self.tasks.lock()));
        self.view.aclose(&telemetry::Cx::NONE).await;
    }
}

fn micros(frame: &AudioFrame) -> u64 {
    (frame.duration() * 1e6) as u64
}

// ------------------------------------------------------------------------------------ the model

impl abi::DuplexRealtimeModel for DuplexAdapterModel {
    type Env = ProcessEnv;

    fn new(
        env: &ProcessEnv,
        host: Arc<dyn abi::DuplexHost>,
        audio_timeout: f64,
    ) -> Result<Self, HostError> {
        Ok(DuplexAdapterModel {
            label: host.label()?,
            caps: super::realtime_capabilities(host.capabilities()?),
            host,
            audio_timeout: Duration::from_secs_f64(audio_timeout.max(0.0)),
            clock: env.clock.clone(),
            anchor: WallAnchor::of(&*env.clock),
            spawner: (env.spawner)().map_err(HostError::new)?,
        })
    }

    fn capabilities(&self) -> RealtimeCapabilities {
        Realtime::capabilities(self)
    }
}
