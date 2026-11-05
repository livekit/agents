//! A plugin's realtime model as the core's own: its session's updates apply one after the other,
//! a reply is requested once the updates sent before it applied, and each generation is drained in
//! its own task.

use std::sync::Arc;

use lk_agents_abi::errors::HostError;
use lk_agents_abi::llm::realtime::{
    CancelReply, Event, GenerateReplyFailed, HostRealtime, HostRealtimeSession,
    RealtimeCapabilities,
};
use lk_agents_abi::{Realtime, RealtimeCommand, RealtimeSession, WallAnchor};

use lk_agents_net::env::ProcessEnv;
use lk_agents_net::telemetry;

/// The plugin session opens only when the driver calls `session()`; asked again, it is the same.
pub struct HostRealtimeModel {
    host: Arc<dyn HostRealtime>,
    label: String,
    caps: RealtimeCapabilities,
    anchor: lk_agents_abi::WallAnchor,
    spawner: Arc<dyn lk_agents_abi::Spawner>,
    session: tokio::sync::Mutex<Option<Arc<HostRealtimeLeg>>>,
}

impl lk_agents_abi::llm::realtime::HostRealtimeModel for HostRealtimeModel {
    type Env = ProcessEnv;

    fn new(env: &ProcessEnv, model: Arc<dyn HostRealtime>) -> Result<Self, HostError> {
        let spawner = (env.spawner)().map_err(HostError::new)?;
        Ok(Self::new(model, WallAnchor::of(&*env.clock), spawner))
    }
}

impl HostRealtimeModel {
    pub fn new(
        host: Arc<dyn HostRealtime>,
        anchor: lk_agents_abi::WallAnchor,
        spawner: Arc<dyn lk_agents_abi::Spawner>,
    ) -> Self {
        HostRealtimeModel {
            label: host.label(),
            caps: host.capabilities(),
            host,
            anchor,
            spawner,
            session: Default::default(),
        }
    }
}

impl Realtime for HostRealtimeModel {
    fn instance_id(&self) -> Option<usize> {
        self.host.instance_id()
    }

    fn label(&self) -> String {
        self.label.clone()
    }

    fn id(&self) -> lk_agents_abi::ProviderId {
        self.host.id()
    }

    fn capabilities(&self) -> RealtimeCapabilities {
        self.caps
    }

    fn session(
        &self,
        cx: telemetry::Cx,
        turn_detection_disabled: bool,
    ) -> lk_agents_abi::BoxFuture<
        '_,
        Result<Arc<dyn RealtimeSession>, lk_agents_abi::errors::ProviderError>,
    > {
        Box::pin(async move {
            let mut slot = self.session.lock().await;
            if let Some(leg) = slot.as_ref() {
                return Ok(Arc::clone(leg) as Arc<dyn RealtimeSession>);
            }
            // The plugin's `session()` raised: the driver's start fails with its message, as the
            // reference's activity start does.
            let host = self.host.session(&cx, turn_detection_disabled).await.map_err(|e| {
                lk_agents_abi::errors::ProviderError::Local(lk_agents_abi::errors::ApiError::new(
                    format!("realtime session could not be opened: {}", e.message()),
                ))
            })?;
            let leg = HostRealtimeLeg::open(host, cx, self.caps, self.anchor, &self.spawner);
            *slot = Some(Arc::clone(&leg));
            Ok(leg as Arc<dyn RealtimeSession>)
        })
    }
}

/// Requests between being asked for and their generation: a cancelled one is dropped before it is
/// sent, or its pending future cancelled after.
#[derive(Default)]
struct Requests {
    cancelled: std::collections::HashSet<String>,
    pending: std::collections::HashMap<String, lk_agents_abi::HostRef>,
}

/// A plugin session as the driver's [`RealtimeSession`]. Updates apply one after the other, and a
/// reply is requested once the updates sent before it applied (a tool's output reaches the server
/// before the reply that reads it); each generation is drained in its own task.
pub struct HostRealtimeLeg {
    host: Arc<dyn HostRealtimeSession>,
    /// What the host calls run under: the context the session was opened in.
    cx: telemetry::Cx,
    caps: RealtimeCapabilities,
    lane: tokio::sync::mpsc::UnboundedSender<RealtimeCommand>,
    requests: Arc<parking_lot::Mutex<Requests>>,
    events: tokio::sync::Mutex<tokio::sync::mpsc::UnboundedReceiver<Event>>,
    spawner: Arc<dyn lk_agents_abi::Spawner>,
    tasks: parking_lot::Mutex<Vec<lk_agents_abi::OwnedTask>>,
    /// The host's event pump, apart from `tasks`: it runs until the host reports the end, so a
    /// close loses no event.
    forward: std::sync::OnceLock<lk_agents_abi::OwnedTask>,
}

impl HostRealtimeLeg {
    fn open(
        host: Arc<dyn HostRealtimeSession>,
        cx: telemetry::Cx,
        caps: RealtimeCapabilities,
        anchor: lk_agents_abi::WallAnchor,
        spawner: &Arc<dyn lk_agents_abi::Spawner>,
    ) -> Arc<Self> {
        let (lane, lane_rx) = tokio::sync::mpsc::unbounded_channel();
        let (events_tx, events) = tokio::sync::mpsc::unbounded_channel();
        let leg = Arc::new(HostRealtimeLeg {
            host: Arc::clone(&host),
            cx: cx.clone(),
            caps,
            lane,
            requests: Default::default(),
            events: tokio::sync::Mutex::new(events),
            spawner: Arc::clone(spawner),
            tasks: Default::default(),
            forward: Default::default(),
        });
        let weak = Arc::downgrade(&leg);
        let forward = lk_agents_abi::SpawnerExt::spawn(
            &**spawner,
            Self::forward(weak.clone(), host, cx, anchor, events_tx.clone()),
        );
        let _ = leg.forward.set(forward);
        leg.spawn(Self::run_lane(weak, lane_rx, events_tx));
        leg
    }

    fn spawn(&self, f: impl std::future::Future<Output = ()> + Send + 'static) {
        use lk_agents_abi::SpawnerExt;
        let mut tasks = self.tasks.lock();
        tasks.retain(|t| !t.is_finished());
        tasks.push(self.spawner.spawn(f));
    }

    /// The host's events, in order; a server-started generation is drained instead.
    async fn forward(
        leg: std::sync::Weak<Self>,
        host: Arc<dyn HostRealtimeSession>,
        cx: telemetry::Cx,
        anchor: lk_agents_abi::WallAnchor,
        events: tokio::sync::mpsc::UnboundedSender<Event>,
    ) {
        loop {
            match host.next_event(&cx).await {
                Ok(Some(Event::ServerGeneration(g))) => {
                    let Some(leg) = leg.upgrade() else { return };
                    let (host, cx) = (Arc::clone(&host), cx.clone());
                    leg.spawn(async move { host.drain(&cx, g.generation, None).await });
                }
                Ok(Some(ev)) => {
                    if events.send(ev.anchored(Some(anchor))).is_err() {
                        return;
                    }
                }
                Ok(None) => return,
                // A malformed event is the plugin's bug; the session goes on.
                Err(e) => {
                    log::warn!(target: "livekit.agents", error = e.message.as_str(); "realtime event refused")
                }
            }
        }
    }

    /// Updates and reply requests, in the order the driver sent them.
    async fn run_lane(
        leg: std::sync::Weak<Self>,
        mut lane: tokio::sync::mpsc::UnboundedReceiver<RealtimeCommand>,
        events: tokio::sync::mpsc::UnboundedSender<Event>,
    ) {
        while let Some(cmd) = lane.recv().await {
            let Some(leg) = leg.upgrade() else { return };
            let req_id = match &cmd {
                RealtimeCommand::GenerateReply(c) => c.req_id.clone(),
                RealtimeCommand::Say(c) => c.req_id.clone(),
                _ => {
                    leg.update(cmd).await;
                    continue;
                }
            };
            leg.request_reply(req_id, cmd, &events).await;
        }
    }

    /// Sends a reply request unless it was cancelled, and drains its generation in its own task.
    async fn request_reply(
        &self,
        req_id: String,
        cmd: RealtimeCommand,
        events: &tokio::sync::mpsc::UnboundedSender<Event>,
    ) {
        if self.requests.lock().cancelled.remove(&req_id) {
            return;
        }
        let request = match self.host.request(&self.cx, cmd).await {
            Ok(request) => request,
            Err(error) => {
                let _ =
                    events.send(Event::GenerateReplyFailed(GenerateReplyFailed { req_id, error }));
                return;
            }
        };
        // Cancelled while it was being sent: the future is cancelled instead of awaited.
        if self.requests.lock().cancelled.remove(&req_id) {
            self.host.cancel(&self.cx, request).await;
            return;
        }
        self.requests.lock().pending.insert(req_id.clone(), request.clone());
        self.spawn(Self::deliver_reply(
            Arc::clone(&self.host),
            self.cx.clone(),
            Arc::clone(&self.requests),
            events.clone(),
            req_id,
            request,
        ));
    }

    /// Awaits the request's generation and drains it; a failure is reported unless cancelled.
    async fn deliver_reply(
        host: Arc<dyn HostRealtimeSession>,
        cx: telemetry::Cx,
        requests: Arc<parking_lot::Mutex<Requests>>,
        events: tokio::sync::mpsc::UnboundedSender<Event>,
        req_id: String,
        request: lk_agents_abi::HostRef,
    ) {
        let generation = host.generation(&cx, request).await;
        // Gone from `pending`: cancelled, and a cancelled request reports nothing.
        let asked = requests.lock().pending.remove(&req_id).is_some();
        match generation {
            Ok(generation) => host.drain(&cx, generation, Some(req_id)).await,
            Err(error) if asked => {
                let _ =
                    events.send(Event::GenerateReplyFailed(GenerateReplyFailed { req_id, error }));
            }
            Err(_) => {}
        }
    }
}

impl HostRealtimeLeg {
    /// A session update, awaited before the next command; a failure is logged.
    async fn update(&self, cmd: RealtimeCommand) {
        let (host, cx) = (&self.host, &self.cx);
        let updated = match cmd {
            RealtimeCommand::UpdateInstructions(c) => {
                host.update_instructions(cx, c.instructions).await
            }
            RealtimeCommand::UpdateChatCtx(c) => host.update_chat_ctx(cx, c.chat_ctx.items).await,
            RealtimeCommand::PushItem(c) => host.push_item(cx, *c.item).await,
            RealtimeCommand::UpdateTools(c) => {
                host.update_tools(cx, c.tools.into_iter().map(|t| t.name).collect()).await
            }
            // `send` routes the rest here only as updates.
            _ => Ok(()),
        };
        if let Err(e) = updated {
            log::error!(target: "livekit.agents", error = e.message.as_str(); "failed to update the realtime session");
        }
    }
}

impl RealtimeSession for HostRealtimeLeg {
    fn capabilities(&self) -> RealtimeCapabilities {
        self.caps
    }

    fn send(&self, cmd: RealtimeCommand) {
        match cmd {
            RealtimeCommand::CancelReply(CancelReply { req_id }) => {
                let pending = {
                    let mut requests = self.requests.lock();
                    let pending = requests.pending.remove(&req_id);
                    if pending.is_none() {
                        requests.cancelled.insert(req_id);
                    }
                    pending
                };
                if let Some(request) = pending {
                    let (host, cx) = (Arc::clone(&self.host), self.cx.clone());
                    self.spawn(async move { host.cancel(&cx, request).await });
                }
            }
            RealtimeCommand::UpdateInstructions(_)
            | RealtimeCommand::UpdateChatCtx(_)
            | RealtimeCommand::UpdateTools(_)
            | RealtimeCommand::PushItem(_)
            | RealtimeCommand::GenerateReply(_)
            | RealtimeCommand::Say(_) => {
                let _ = self.lane.send(cmd);
            }
            RealtimeCommand::PushAudio(_)
            | RealtimeCommand::CommitAudio(_)
            | RealtimeCommand::ClearAudio(_)
            | RealtimeCommand::Interrupt(_)
            | RealtimeCommand::Truncate(_) => self.host.send(cmd),
        }
    }

    /// The host session keeps no input policy: pre-connect audio is ordinary audio to it.
    fn head(&self, frame: lk_agents_abi::AudioFrame) {
        self.host.send(RealtimeCommand::PushAudio(lk_agents_abi::PushAudio { frame }));
    }

    fn next_event(&self) -> lk_agents_abi::BoxFuture<'_, Option<Event>> {
        Box::pin(async move { self.events.lock().await.recv().await })
    }

    fn chat_ctx(&self) -> lk_agents_abi::BoxFuture<'_, lk_agents_abi::ChatContext> {
        Box::pin(async move {
            // Only read when no agent context is bound, which never happens for a host session.
            lk_agents_abi::ChatContext::default()
        })
    }

    fn aclose(&self) -> lk_agents_abi::BoxFuture<'_, ()> {
        Box::pin(async move {
            // As the reference's `_core_aclose`: the drains and updates are cancelled, then the
            // session closes.
            drop(std::mem::take(&mut *self.tasks.lock()));
            self.host.aclose(&self.cx).await;
        })
    }
}
