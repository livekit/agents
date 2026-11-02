//! The warm transfer (`beta.workflows.WarmTransfer`): while the caller waits on hold, a second
//! session in a room of its own dials a human agent, briefs them on the call, and either moves
//! them into the caller's room or reports why not. The core runs the sequence and decides; the
//! host (`TransferHost`) connects rooms, starts the second session, plays the hold music and
//! makes the LiveKit and Twilio requests.

use lk_agents_abi::Maybe;
use lk_agents_abi::errors::HostError;
use lk_agents_abi::voice::rtc::{DisconnectReason, RoomEvent, RtcRoom, TRACK_KIND_AUDIO};
use lk_agents_abi::voice::workflows::InstructionsArg;
pub use lk_agents_abi::voice::workflows::warm_transfer::{
    HUMAN_AGENT_IDENTITY, TWILIO_RINGING_TIMEOUT, TransferFailed, WarmTransferRefused,
};
use lk_agents_abi::voice::workflows::{SipParticipantRequest, TwilioDial};
use lk_agents_abi::{Clock, HostRef, Spawner, SpawnerExt};
use lk_agents_net::task::{TaskId, TaskSet};
use lk_agents_voice::workflows::effects::{TransferEffects, TransferToolEffect};
use serde_json::Value;

use std::collections::BTreeMap;

use parking_lot::Mutex;
use std::sync::Arc;
use std::time::Duration;

const LOG: &str = "livekit.agents";

use lk_agents_voice::workflows::twilio::{self, TwilioError};
use lk_agents_voice::workflows::warm_transfer::{
    Origination, Transfer, TransferFailure, TransferOutcome, TransferPlan, TransferTool,
    ends_merged_call, human_agent_room_name, transfer_tools,
};

use lk_agents_net::env::ProcessEnv;
pub use lk_agents_voice::workflows::warm_transfer::TransferConfig;

pub use lk_agents_abi::voice::workflows::warm_transfer::TransferHost;

/// The run's shared state: who settles it, and what the tools reach.
#[derive(Default)]
struct Running {
    host: Option<Arc<dyn TransferHost>>,
    caller: Option<Arc<dyn RtcRoom>>,
    core: Transfer,
    hold: Option<HostRef>,
    watcher: Option<TaskId>,
}

/// `_core.WarmTransfer`: the transfer's plan (its briefing instructions and tools, how it dials)
/// and its run.
pub struct TransferRun {
    plan: TransferPlan,
    clock: Arc<dyn Clock>,
    spawner: Arc<dyn Spawner>,

    running: Mutex<Running>,
    done: tokio::sync::Notify,
    /// The human-room watcher and the caller-room follower.
    tasks: Arc<TaskSet>,
}

impl TransferRun {
    pub fn new(plan: TransferPlan, clock: Arc<dyn Clock>, spawner: Arc<dyn Spawner>) -> Self {
        TransferRun {
            plan,
            clock,
            spawner,
            running: Mutex::new(Running::default()),
            tasks: Arc::default(),
            done: tokio::sync::Notify::new(),
        }
    }

    /// The briefing agent's static instructions.
    pub fn instructions(&self) -> String {
        self.plan.instructions.clone()
    }

    /// The briefing agent's three tools, as raw function schemas, each answered by
    /// [`call_tool`](Self::call_tool). All three are hidden while the briefing is entered:
    /// connecting then would cut the summary off.
    pub fn tools(&self) -> Vec<BTreeMap<String, Value>> {
        transfer_tools()
    }

    /// The transfer: the caller on hold with their session's io off, the human agent dialled and
    /// briefed, until one of the tools or the human agent's room settles it; the human agent's
    /// identity once they joined the caller. Whatever ends the
    /// run, the human agent's session is closed, the hold stops and the io is restored.
    pub async fn run(
        self: &Arc<Self>,
        host: Arc<dyn TransferHost>,
        caller: Arc<dyn RtcRoom>,
    ) -> Result<String, TransferFailed> {
        {
            let mut running = self.lock();
            if running.host.is_some() {
                return Err(HostError::new("this warm transfer is already running").into());
            }
            *running = Running {
                host: Some(host.clone()),
                caller: Some(caller.clone()),
                ..Running::default()
            };
        }
        let _restore = Restore(self.clone());
        let hold = host.start_hold().await?;
        self.lock().hold = hold;
        let io = host.io_state();
        let effects = self.lock().core.begin(io);
        self.perform(effects);
        let settled = self.done.notified();
        let mut settled = std::pin::pin!(settled);
        settled.as_mut().enable();
        tokio::select! {
            biased;
            dialled = self.dial(&host, &caller) => {
                if let Err(err) = dialled {
                    log::error!(target: LOG, "could not dial human agent: {err}");
                    self.abort_dial(&host).await;
                    self.settle(Err(TransferFailure::Dial));
                }
            }
            _ = &mut settled => {}
        }
        loop {
            let settled = self.done.notified();
            let mut settled = std::pin::pin!(settled);
            settled.as_mut().enable();
            if let Some(outcome) = self.lock().core.take_outcome() {
                return outcome.map_err(TransferFailed::from);
            }
            settled.await;
        }
    }

    /// One of [`tools`](Self::tools), called by the briefing agent: `connect_to_caller` moves
    /// the human agent into the caller's room and settles the transfer, `decline_transfer` and
    /// `voicemail_detected` settle it as failed.
    pub async fn call_tool(
        self: &Arc<Self>,
        name: String,
        arguments: Value,
    ) -> Result<(), TransferFailed> {
        let tool =
            TransferTool::parse(&name, &arguments).ok_or(TransferFailed::UnknownTool(name))?;
        let effect = self.lock().core.tool(tool).ok_or(TransferFailed::NotRunning)?;
        match effect {
            TransferToolEffect::MoveParticipant => self.connect().await,
            TransferToolEffect::Settled(effects) => {
                self.perform(effects);
                self.done.notify_waiters();
                Ok(())
            }
        }
    }
}

/// The run's `finally`: the human agent's session closed without drain if still open, the hold
/// stopped, the io restored.
struct Restore(Arc<TransferRun>);

impl Drop for Restore {
    fn drop(&mut self) {
        let effects = {
            let mut running = self.0.lock();
            if let Some(watcher) = running.watcher.take() {
                self.0.tasks.cancel(watcher);
            }
            running.caller = None;
            running.core.cancel()
        };
        self.0.perform(effects);
        self.0.lock().host = None;
    }
}

impl TransferRun {
    fn lock(&self) -> parking_lot::MutexGuard<'_, Running> {
        self.running.lock()
    }

    fn perform(&self, effects: TransferEffects) {
        let (host, hold) = {
            let mut running = self.lock();
            (running.host.clone(), effects.stop_hold.then(|| running.hold.take()).flatten())
        };
        if let Some(host) = host {
            if let Some(drain) = effects.close_human {
                host.close_human(drain);
            }
            if let Some(hold) = hold {
                host.stop_hold(hold);
            }
            if let Some(io) = effects.io {
                host.set_io(io);
            }
        }
    }

    fn settle(&self, outcome: TransferOutcome) {
        let effects = self.lock().core.settle(outcome);
        if let Some(effects) = effects {
            self.perform(effects);
            self.done.notify_waiters();
        }
    }

    /// The human agent's room connected and watched, their session started, and the call
    /// placed; returns once they answered.
    async fn dial(
        self: &Arc<Self>,
        host: &Arc<dyn TransferHost>,
        caller: &Arc<dyn RtcRoom>,
    ) -> Result<(), DialError> {
        let room_name = human_agent_room_name(&caller.name()?);
        let room = host
            .connect_human(room_name.clone(), caller.local().identity()?.to_string())
            .await?
            .rtc()
            .ok_or_else(|| HostError::new("warm transfer requires a connected human RTC room"))?;
        let events = room.events();
        let this = self.clone();
        {
            let mut running = self.lock();
            running.core.human_connected();
            running.watcher = Some(self.tasks.spawn(
                self.spawner.as_ref(),
                "warm transfer human watcher",
                async move { this.watch_human(events).await },
            ));
        }
        host.start_human().await?;
        match &self.plan.origination {
            Origination::Sip { sip_call_to, sip_trunk_id, sip_number, sip_headers, dtmf } => {
                let request = SipParticipantRequest {
                    sip_trunk_id: sip_trunk_id.clone(),
                    sip_call_to: sip_call_to.clone(),
                    room_name,
                    participant_identity: HUMAN_AGENT_IDENTITY.to_owned(),
                    wait_until_answered: true,
                    sip_number: Some(sip_number.clone()).filter(|s| !s.is_empty()),
                    headers: sip_headers.clone(),
                    dtmf: dtmf.clone().unwrap_or_default(),
                    ringing_timeout: self.plan.ringing_timeout.map(|d| d.as_secs_f64()),
                };
                host.create_sip_participant(request).await?;
            }
            Origination::Twilio(dial) => {
                twilio_originate(
                    host,
                    dial,
                    &room_name,
                    &room,
                    self.plan.ringing_timeout,
                    &*self.clock,
                    self.spawner.clone(),
                )
                .await?;
            }
        }
        Ok(())
    }

    /// A failed dial: the human agent's room no longer watched, their session closed without
    /// drain and the room left.
    async fn abort_dial(&self, host: &Arc<dyn TransferHost>) {
        let effects = {
            let mut running = self.lock();
            if let Some(watcher) = running.watcher.take() {
                self.tasks.cancel(watcher);
            }
            running.core.abort_dial()
        };
        let close = effects.close_human.is_some();
        self.perform(effects);
        if close && let Err(err) = host.disconnect_human().await {
            log::debug!(target: LOG, "leaving the human agent's room failed: {err}");
        }
    }

    /// The human agent's room closing settles the transfer, unless they are being moved.
    async fn watch_human(&self, mut events: lk_agents_abi::voice::rtc::Receiver<RoomEvent>) {
        loop {
            let event = match events.next().await {
                Ok(Some(event)) => event,
                Ok(None) => return,
                Err(error) => {
                    self.settle(Err(TransferFailure::Host(error.to_string())));
                    return;
                }
            };
            if let RoomEvent::Close { reason } = event {
                log::debug!(target: LOG, "human agent's room closed: {reason:?}");
                let effects = self.lock().core.room_closed(reason_name(reason));
                if let Some(effects) = effects {
                    self.perform(effects);
                    self.done.notify_waiters();
                }
                return;
            }
        }
    }

    /// `connect_to_caller`: the human agent moved into the caller's room, the transfer settled,
    /// and the caller's room deleted once the caller or the human agent leaves it.
    async fn connect(self: &Arc<Self>) -> Result<(), TransferFailed> {
        let (host, caller) = {
            let running = self.lock();
            let (Some(host), Some(caller)) = (running.host.clone(), running.caller.clone()) else {
                return Err(TransferFailed::NotRunning);
            };
            (host, caller)
        };
        let caller_room = caller
            .name()
            .map_err(|error| TransferFailed::Host(HostError::new(error.to_string())))?;
        log::debug!(target: LOG, "moving {HUMAN_AGENT_IDENTITY} to caller room {caller_room}");
        let moved = host
            .move_participant(
                human_agent_room_name(&caller_room),
                HUMAN_AGENT_IDENTITY.to_owned(),
                caller_room,
            )
            .await;
        if let Err(error) = moved {
            self.lock().core.merge_failed();
            return Err(error.into());
        }
        self.settle(Ok(HUMAN_AGENT_IDENTITY.to_owned()));
        let mut events = caller.events();
        self.tasks.spawn(self.spawner.as_ref(), "warm transfer caller follower", async move {
            loop {
                let event = match events.next().await {
                    Ok(Some(event)) => event,
                    Ok(None) => return,
                    Err(error) => {
                        log::error!(target: LOG, "caller room event stream failed after transfer: {error}");
                        return;
                    }
                };
                let RoomEvent::ParticipantDisconnected { participant, .. } = event else {
                    continue;
                };
                if ends_merged_call(participant.kind()) {
                    log::info!(
                        target: LOG,
                        "participant disconnected from caller room: {}, closing",
                        participant.identity()
                    );
                    if let Err(err) = host.delete_caller_room().await {
                        log::warn!(target: LOG, "failed to delete the caller's room: {err}");
                    }
                    return;
                }
            }
        });
        Ok(())
    }
}

/// `rtc.DisconnectReason.Name(reason)`.
fn reason_name(reason: DisconnectReason) -> String {
    reason.as_str().to_uppercase()
}

/// Why a dial failed: logged, while the transfer reports [`TransferFailed::Dial`].
#[derive(Debug, thiserror::Error)]
enum DialError {
    #[error(transparent)]
    Host(#[from] HostError),
    #[error(transparent)]
    Twilio(#[from] TwilioError),
    #[error("supervisor did not answer")]
    NoAnswer,
    #[error(transparent)]
    Rtc(#[from] lk_agents_abi::voice::rtc::RtcError),
}

/// One Twilio request through the host; the reply's JSON object.
async fn twilio_send(
    host: &Arc<dyn TransferHost>,
    request: twilio::Request,
) -> Result<serde_json::Map<String, Value>, DialError> {
    let (status, body) = host.http_post(request.url, request.headers, request.form).await?;
    Ok(twilio::reply(status, &body)?)
}

/// Create the outbound call, streaming into the connector: its SID.
async fn twilio_create_call(
    host: Arc<dyn TransferHost>,
    dial: TwilioDial,
    from: String,
    twiml: String,
    call_token: Option<String>,
) -> Result<String, DialError> {
    let call = twilio_send(&host, twilio::create_call(&dial, from, &twiml, call_token)).await?;
    Ok(twilio::call_sid(&call)?)
}

/// A call the transfer gave up on, cancelled, or completed when it was answered meanwhile.
async fn twilio_cancel(host: Arc<dyn TransferHost>, dial: TwilioDial, sid: String) {
    let cancelled = match twilio_send(&host, twilio::set_status(&dial, &sid, "canceled")).await {
        Err(DialError::Twilio(error)) if twilio::complete_instead(&error) => {
            twilio_send(&host, twilio::set_status(&dial, &sid, "completed")).await.map(drop)
        }
        other => other.map(drop),
    };
    if cancelled.is_err() {
        log::warn!(target: LOG, "Failed to cancel Twilio call");
    }
}

/// The outbound call while the dial waits on it: still being created, or created.
enum Call {
    Creating(tokio::sync::oneshot::Receiver<Result<String, DialError>>),
    Created(String),
}

/// Cancels the call when the dial does not get past it, however it ends (a failure or the run
/// dropped): Twilio can create the call after the transfer gave up.
struct CancelOnDrop {
    spawner: Arc<dyn Spawner>,
    host: Arc<dyn TransferHost>,
    dial: TwilioDial,
    call: Option<Call>,
}

impl CancelOnDrop {
    /// The creation's outcome; the guard then holds the call it made, if any. Awaited in place,
    /// so a dial dropped meanwhile still cancels what the creation goes on to make.
    async fn created(&mut self) -> Result<String, DialError> {
        let Some(Call::Creating(task)) = &mut self.call else {
            unreachable!("a creation is pending")
        };
        let result =
            task.await.unwrap_or_else(|err| Err(DialError::Host(HostError::new(err.to_string()))));
        self.call = result.as_ref().ok().map(|sid| Call::Created(sid.clone()));
        result
    }
}

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        let (host, dial) = (self.host.clone(), self.dial.clone());
        match self.call.take() {
            // Detached: the dial's owner is being dropped, and the call must still be cancelled.
            Some(Call::Creating(task)) => {
                self.spawner.spawn_detached(async move {
                    if let Ok(Ok(sid)) = task.await {
                        twilio_cancel(host, dial, sid).await;
                    }
                });
            }
            Some(Call::Created(sid)) => {
                self.spawner.spawn_detached(twilio_cancel(host, dial, sid));
            }
            None => {}
        }
    }
}

/// Dial through Twilio: the connector's stream URL, the call created with
/// [`twilio::first_caller`] (again with [`twilio::retry_caller`] when Twilio rejects it), then
/// the human agent's audio track awaited in their room, within `ringing_timeout`.
async fn twilio_originate(
    host: &Arc<dyn TransferHost>,
    dial: &TwilioDial,
    room_name: &str,
    room: &Arc<dyn RtcRoom>,
    ringing_timeout: Option<Duration>,
    clock: &dyn Clock,
    spawner: Arc<dyn Spawner>,
) -> Result<(), DialError> {
    let connect_url =
        host.connect_twilio_call(room_name.to_owned(), HUMAN_AGENT_IDENTITY.to_owned()).await?;
    let twiml = twilio::twiml(&connect_url);
    let (from, token) = twilio::first_caller(dial);
    let create = |from: String, token: Option<String>| {
        let (reply, created) = tokio::sync::oneshot::channel();
        let (host, dial, twiml) = (host.clone(), dial.clone(), twiml.clone());
        // Detached: a creation the caller stops waiting for is cancelled by `CancelOnDrop`.
        spawner.spawn_detached(async move {
            let _ = reply.send(twilio_create_call(host, dial, from, twiml, token).await);
        });
        created
    };
    let mut guard = CancelOnDrop {
        spawner: spawner.clone(),
        host: host.clone(),
        dial: dial.clone(),
        call: Some(Call::Creating(create(from, token.clone()))),
    };
    let mut created = guard.created().await;
    if let Err(DialError::Twilio(error)) = &created
        && let Some(from) = twilio::retry_caller(dial, token.as_deref(), error)
    {
        log::warn!(target: LOG, "Twilio rejected preserved caller ID; retrying with business caller ID");
        guard.call = Some(Call::Creating(create(from, None)));
        created = guard.created().await;
    }
    created?;
    let answered = crate::rtc::wait::track_publication(
        room.as_ref(),
        Some(HUMAN_AGENT_IDENTITY.to_owned()),
        Some(vec![TRACK_KIND_AUDIO]),
        false,
        false,
    );
    match ringing_timeout {
        Some(limit) => lk_agents_abi::timeout(clock, limit, answered)
            .await
            .map_err(|_| DialError::NoAnswer)??,
        None => answered.await?,
    };
    guard.call = None;
    Ok(())
}

pub struct WarmTransfer(Arc<TransferRun>);

impl lk_agents_abi::voice::workflows::warm_transfer::WarmTransfer for WarmTransfer {
    type Env = ProcessEnv;

    #[allow(clippy::too_many_arguments, reason = "Preserve the public host operation signature")]
    fn new(
        env: &ProcessEnv,
        sip_call_to: Option<String>,
        sip_trunk_id: Maybe<Option<String>>,
        has_sip_connection: bool,
        sip_number: Maybe<String>,
        sip_headers: Option<BTreeMap<String, String>>,
        dtmf: Option<String>,
        ringing_timeout: Option<f64>,
        instructions: Maybe<InstructionsArg>,
        conversation: Option<Vec<(String, String)>>,
        env_sip_trunk: Option<String>,
        env_sip_number: String,
        twilio: Option<TwilioDial>,
    ) -> Result<Self, WarmTransferRefused> {
        let plan = TransferPlan::resolve(TransferConfig {
            sip_call_to,
            sip_trunk_id,
            has_sip_connection,
            sip_number,
            sip_headers,
            dtmf,
            ringing_timeout,
            instructions,
            conversation,
            env_sip_trunk,
            env_sip_number,
            twilio,
        })?;
        let spawner = (env.spawner)().map_err(HostError::new)?;
        Ok(Self(Arc::new(TransferRun::new(plan, env.clock.clone(), spawner))))
    }

    fn instructions(&self) -> String {
        self.0.instructions()
    }
    fn tools(&self) -> Vec<BTreeMap<String, serde_json::Value>> {
        self.0.tools()
    }

    async fn run(
        &self,
        host: Arc<dyn lk_agents_abi::voice::workflows::warm_transfer::TransferHost>,
        caller: Arc<dyn lk_agents_abi::voice::room_owner::SessionRoom>,
    ) -> Result<String, TransferFailed> {
        self.0
            .run(
                host,
                caller.rtc().ok_or_else(|| {
                    HostError::new("warm transfer requires a connected caller RTC room")
                })?,
            )
            .await
    }

    async fn call_tool(
        &self,
        name: String,
        arguments: serde_json::Value,
    ) -> Result<(), TransferFailed> {
        self.0.call_tool(name, arguments).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lk_agents_abi::voice::rtc::{
        ByteStreamIngress, ConnectionState, LocalParticipant, ParticipantIdentity,
        RemoteParticipant, RoomEventSink, RtcError, TextStreamIngress,
    };
    use lk_agents_abi::voice::workflows::IoEnableState;
    use lk_agents_voice::workflows::warm_transfer as p;

    /// A caller room these tests never reach: each stops before the transfer touches it.
    struct Untouched;
    impl RtcRoom for Untouched {
        fn rpc_router(&self) -> &Arc<dyn lk_agents_abi::voice::rtc::ports::RpcRoutes> {
            panic!("room must remain untouched")
        }

        fn name(&self) -> Result<String, RtcError> {
            unreachable!()
        }
        fn connection_state(&self) -> Result<ConnectionState, RtcError> {
            unreachable!()
        }
        fn local(&self) -> &Arc<dyn LocalParticipant> {
            unreachable!()
        }
        fn participant(
            &self,
            _: &ParticipantIdentity,
        ) -> Result<Option<Arc<dyn RemoteParticipant>>, RtcError> {
            unreachable!()
        }
        fn remote_participants(&self) -> Result<Vec<Arc<dyn RemoteParticipant>>, RtcError> {
            unreachable!()
        }
        fn event_bus(&self) -> &Arc<dyn RoomEventSink> {
            unreachable!()
        }
        fn register_byte_stream_handler(
            &self,
            _: &str,
            _: Arc<dyn ByteStreamIngress>,
        ) -> Result<(), RtcError> {
            unreachable!()
        }
        fn unregister_byte_stream_handler(&self, _: &str) -> Result<(), RtcError> {
            unreachable!()
        }
        fn register_text_stream_handler(
            &self,
            _: &str,
            _: Arc<dyn TextStreamIngress>,
        ) -> Result<(), RtcError> {
            unreachable!()
        }
        fn unregister_text_stream_handler(&self, _: &str) -> Result<(), RtcError> {
            unreachable!()
        }
    }

    fn transfer() -> Arc<TransferRun> {
        let plan = TransferPlan::resolve(TransferConfig {
            sip_call_to: Some("+15550100".to_owned()),
            sip_trunk_id: Maybe::NotGiven,
            has_sip_connection: true,
            sip_number: Maybe::NotGiven,
            sip_headers: None,
            dtmf: None,
            ringing_timeout: None,
            instructions: Maybe::NotGiven,
            conversation: None,
            env_sip_trunk: None,
            env_sip_number: String::new(),
            twilio: None,
        })
        .unwrap();
        Arc::new(TransferRun::new(
            plan,
            Arc::new(lk_agents_abi::VirtualClock::new()),
            Arc::new(lk_agents_net::task::TokioSpawner::current()),
        ))
    }

    #[derive(Default)]
    struct Holding {
        hold_starts: std::sync::atomic::AtomicUsize,
        create_release: Mutex<Option<tokio::sync::oneshot::Receiver<()>>>,
        call_statuses: Mutex<Vec<String>>,
    }

    impl TransferHost for Holding {
        fn start_hold(&self) -> lk_agents_abi::BoxFuture<'_, Result<Option<HostRef>, HostError>> {
            Box::pin(async move {
                self.hold_starts.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                std::future::pending().await
            })
        }
        fn stop_hold(&self, _: HostRef) {
            unreachable!()
        }
        fn io_state(&self) -> IoEnableState {
            unreachable!()
        }
        fn set_io(&self, _: IoEnableState) {
            unreachable!()
        }
        fn connect_human(
            &self,
            _: String,
            _: String,
        ) -> lk_agents_abi::BoxFuture<
            '_,
            Result<Arc<dyn lk_agents_abi::voice::room_owner::SessionRoom>, HostError>,
        > {
            Box::pin(async move { unreachable!() })
        }
        fn start_human(&self) -> lk_agents_abi::BoxFuture<'_, Result<(), HostError>> {
            Box::pin(async move { unreachable!() })
        }
        fn close_human(&self, _: bool) {
            unreachable!()
        }
        fn disconnect_human(&self) -> lk_agents_abi::BoxFuture<'_, Result<(), HostError>> {
            Box::pin(async move { unreachable!() })
        }
        fn create_sip_participant(
            &self,
            _: SipParticipantRequest,
        ) -> lk_agents_abi::BoxFuture<'_, Result<(), HostError>> {
            Box::pin(async move { unreachable!() })
        }
        fn connect_twilio_call(
            &self,
            _: String,
            _: String,
        ) -> lk_agents_abi::BoxFuture<'_, Result<String, HostError>> {
            Box::pin(async { Ok("wss://connector.test".to_owned()) })
        }
        fn http_post(
            &self,
            url: String,
            _: BTreeMap<String, String>,
            form: Vec<(String, String)>,
        ) -> lk_agents_abi::BoxFuture<'_, Result<(u16, String), HostError>> {
            Box::pin(async move {
                if url.ends_with("/Calls.json") {
                    let release = self.create_release.lock().take();
                    if let Some(release) = release {
                        let _ = release.await;
                    }
                    Ok((200, r#"{"sid":"late-call"}"#.to_owned()))
                } else {
                    let status = form.iter().find(|(key, _)| key == "Status").unwrap().1.clone();
                    self.call_statuses.lock().push(status);
                    Ok((200, "{}".to_owned()))
                }
            })
        }
        fn move_participant(
            &self,
            _: String,
            _: String,
            _: String,
        ) -> lk_agents_abi::BoxFuture<'_, Result<(), HostError>> {
            Box::pin(async move { unreachable!() })
        }
        fn delete_caller_room(&self) -> lk_agents_abi::BoxFuture<'_, Result<(), HostError>> {
            Box::pin(async move { unreachable!() })
        }
    }

    #[tokio::test]
    async fn a_second_run_cannot_replace_the_first_runs_resources() {
        use std::{future::Future, task::Poll};
        let transfer = transfer();
        let host = Arc::new(Holding::default());
        let caller: Arc<dyn RtcRoom> = Arc::new(Untouched);
        let first = transfer.run(host.clone(), caller.clone());
        let mut first = std::pin::pin!(first);
        let polled = std::future::poll_fn(|cx| Poll::Ready(first.as_mut().poll(cx))).await;
        assert!(polled.is_pending());
        let second = transfer.run(host.clone(), caller);
        let mut second = std::pin::pin!(second);
        let polled = std::future::poll_fn(|cx| Poll::Ready(second.as_mut().poll(cx))).await;
        let Poll::Ready(Err(error)) = polled else { panic!("a second run must be refused") };
        assert_eq!(error.to_string(), "this warm transfer is already running");
        assert_eq!(host.hold_starts.load(std::sync::atomic::Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn tools_are_refused_until_the_human_session_is_ready() {
        let transfer = transfer();
        for tool in [p::CONNECT_TOOL, p::DECLINE_TOOL, p::VOICEMAIL_TOOL] {
            assert!(matches!(
                transfer.call_tool(tool.to_owned(), Value::Null).await,
                Err(TransferFailed::NotRunning)
            ));
        }
    }
    struct ObservedSpawner {
        inner: lk_agents_net::task::TokioSpawner,
        started: std::sync::atomic::AtomicUsize,
    }
    impl Spawner for ObservedSpawner {
        fn launch(
            &self,
            task: lk_agents_abi::Launch,
        ) -> lk_agents_abi::OwnedTask<Box<dyn std::any::Any + Send>> {
            self.started.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            self.inner.launch(task)
        }
    }

    #[tokio::test]
    async fn cancelling_a_pending_creation_retires_the_late_call_on_the_supplied_spawner() {
        use std::future::Future as _;
        let spawner = Arc::new(ObservedSpawner {
            inner: lk_agents_net::task::TokioSpawner::current(),
            started: 0.into(),
        });
        let (release, pending) = tokio::sync::oneshot::channel();
        let host =
            Arc::new(Holding { create_release: Mutex::new(Some(pending)), ..Default::default() });
        let port: Arc<dyn TransferHost> = host.clone();
        let dial = TwilioDial {
            phone_number: "+15550100".to_owned(),
            from_number: "+15550200".to_owned(),
            account_sid: "account".to_owned(),
            auth_token: "secret".to_owned(),
            call_token: None,
            original_caller_number: None,
        };
        let clock = lk_agents_abi::VirtualClock::new();
        let room: Arc<dyn RtcRoom> = Arc::new(Untouched);
        let mut dialing = Box::pin(twilio_originate(
            &port,
            &dial,
            "late-create",
            &room,
            Some(Duration::from_secs(1)),
            &clock,
            spawner.clone(),
        ));
        let first =
            std::future::poll_fn(|cx| std::task::Poll::Ready(dialing.as_mut().poll(cx))).await;
        assert!(first.is_pending(), "the host creation must remain pending");
        assert_eq!(spawner.started.load(std::sync::atomic::Ordering::SeqCst), 1);
        drop(dialing);
        assert_eq!(spawner.started.load(std::sync::atomic::Ordering::SeqCst), 2);
        release.send(()).unwrap();
        for _ in 0..64 {
            if !host.call_statuses.lock().is_empty() {
                break;
            }
            tokio::task::yield_now().await;
        }
        assert_eq!(*host.call_statuses.lock(), ["canceled"]);
    }
}
