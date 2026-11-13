use crate::telemetry;
use lk_agents_abi::Maybe;
use lk_agents_abi::errors::{ApiError, HostError};
use lk_agents_abi::voice::rtc::{DisconnectReason, RoomEvent, RtcRoom, TRACK_KIND_AUDIO};
use lk_agents_abi::voice::workflows::InstructionsArg;
use lk_agents_abi::voice::workflows::warm_transfer::HoldMusic;
pub use lk_agents_abi::voice::workflows::warm_transfer::{
    HUMAN_AGENT_IDENTITY, TWILIO_RINGING_TIMEOUT, TransferFailed, WarmTransferRefused,
};
use lk_agents_abi::voice::workflows::{TwilioCallEnd, TwilioDial};
use lk_agents_abi::{Clock, Spawner};
use lk_agents_net::owned::Background;
use lk_agents_net::room_service::{RoomService, SipCall, Telephony, api_error};
use lk_agents_net::task::{ScopedTask, TaskSet};
use lk_agents_voice::workflows::effects::{TransferEffects, TransferToolEffect};
use serde_json::Value;

use std::collections::BTreeMap;

use parking_lot::Mutex;
use std::sync::Arc;
use std::time::Duration;

use super::twilio as twilio_http;
use lk_agents_voice::workflows::machine::warm_transfer::{
    MergedCall, Origination, Transfer, TransferOutcome, TransferPlan, TransferTool,
    follow_merged_call, human_agent_room_name, transfer_tools,
};
use lk_agents_voice::workflows::twilio::{self, Abandon, Creation, TwilioError};

use lk_agents_net::env::ProcessEnv;
pub use lk_agents_voice::workflows::machine::warm_transfer::TransferConfig;

pub use lk_agents_abi::voice::workflows::warm_transfer::TransferHost;

/// The LiveKit and Twilio requests a transfer makes: [`ProjectApi`] over the server's project,
/// a fake in tests.
pub trait TransferApi: Send + Sync {
    /// Dial `call`, returning once it is answered when it waits for an answer.
    fn create_sip_participant(&self, call: SipCall) -> BoxFuture<'_, Result<(), ApiError>>;
    /// The Twilio media stream URL that joins `room` as `identity`.
    fn connect_twilio_call(
        &self,
        room: String,
        identity: String,
    ) -> BoxFuture<'_, Result<String, ApiError>>;
    /// A Twilio REST request: the first response's status and body, redirects not followed.
    fn post_form(
        &self,
        request: twilio_http::Request,
    ) -> BoxFuture<'_, Result<(u16, String), ApiError>>;
    fn move_participant(
        &self,
        room: String,
        identity: String,
        destination: String,
    ) -> BoxFuture<'_, Result<(), ApiError>>;
    /// Deletes `room`; a room already gone is not an error.
    fn delete_room(&self, room: String) -> BoxFuture<'_, Result<(), ApiError>>;
}

type BoxFuture<'a, T> = lk_agents_abi::BoxFuture<'a, T>;

/// [`TransferApi`] over a project's services, with the server's URL and key pair.
pub struct ProjectApi {
    rooms: RoomService,
    telephony: Telephony,
}

impl ProjectApi {
    /// The project of the server's `LIVEKIT_URL`, `LIVEKIT_API_KEY` and `LIVEKIT_API_SECRET`.
    pub fn of_server(env: &BTreeMap<String, String>) -> Self {
        let get = |name: &str| env.get(name).map_or("", String::as_str);
        let (url, key, secret) =
            (get("LIVEKIT_URL"), get("LIVEKIT_API_KEY"), get("LIVEKIT_API_SECRET"));
        ProjectApi {
            rooms: RoomService::new(url, key, secret),
            telephony: Telephony::new(url, key, secret),
        }
    }
}

impl TransferApi for ProjectApi {
    fn create_sip_participant(&self, call: SipCall) -> BoxFuture<'_, Result<(), ApiError>> {
        Box::pin(
            async move { self.telephony.create_sip_participant(call).await.map_err(api_error) },
        )
    }
    fn connect_twilio_call(
        &self,
        room: String,
        identity: String,
    ) -> BoxFuture<'_, Result<String, ApiError>> {
        Box::pin(async move {
            self.telephony.connect_twilio_call(&room, &identity).await.map_err(api_error)
        })
    }
    fn post_form(
        &self,
        request: twilio_http::Request,
    ) -> BoxFuture<'_, Result<(u16, String), ApiError>> {
        Box::pin(async move {
            lk_agents_net::room_service::post_form(&request.url, &request.headers, &request.form)
                .await
        })
    }
    fn move_participant(
        &self,
        room: String,
        identity: String,
        destination: String,
    ) -> BoxFuture<'_, Result<(), ApiError>> {
        Box::pin(async move {
            self.rooms.move_participant(&room, &identity, &destination).await.map_err(api_error)
        })
    }
    fn delete_room(&self, room: String) -> BoxFuture<'_, Result<(), ApiError>> {
        Box::pin(async move { self.rooms.delete_room(&room).await.map_err(api_error) })
    }
}

/// The run's shared state: who settles it, and what the tools reach.
#[derive(Default)]
struct Running {
    phase: RunPhase,
    core: Transfer,
    hold: Option<Arc<dyn HoldMusic>>,
    watcher: Option<ScopedTask>,
}

#[derive(Default)]
enum RunPhase {
    #[default]
    Idle,
    Active {
        host: Arc<dyn TransferHost>,
        api: Arc<dyn TransferApi>,
        caller: Arc<dyn RtcRoom>,
    },
    Restoring {
        host: Arc<dyn TransferHost>,
    },
}

/// A warm transfer's plan (briefing instructions, tools, how it dials) and its run.
pub struct TransferRun {
    plan: TransferPlan,
    /// The trunk to dial through when the plan names no trunk id.
    sip_connection: Option<livekit_protocol::SipOutboundConfig>,
    clock: Arc<dyn Clock>,
    spawner: Arc<dyn Spawner>,

    running: Mutex<Running>,
    /// Written as the run settles; the run waits on it while it borrows `self`, so the outcome
    /// (or the run's own `Restore`) is its terminal state.
    done: lk_agents_net::signal::Producer<()>,
    /// The human-room watcher and the caller-room follower.
    tasks: Arc<TaskSet>,
}

impl TransferRun {
    pub fn new(
        plan: TransferPlan,
        sip_connection: Option<livekit_protocol::SipOutboundConfig>,
        clock: Arc<dyn Clock>,
        spawner: Arc<dyn Spawner>,
    ) -> Self {
        TransferRun {
            plan,
            sip_connection,
            clock,
            spawner,
            running: Mutex::new(Running::default()),
            tasks: Arc::default(),
            done: lk_agents_net::signal::signal(()).0,
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
        api: Arc<dyn TransferApi>,
        caller: Arc<dyn RtcRoom>,
    ) -> Result<String, TransferFailed> {
        {
            let mut running = self.lock();
            if !matches!(running.phase, RunPhase::Idle) {
                return Err(HostError::new("this warm transfer is already running").into());
            }
            *running = Running {
                phase: RunPhase::Active {
                    host: host.clone(),
                    api: api.clone(),
                    caller: caller.clone(),
                },
                ..Running::default()
            };
        }
        let _restore = Restore(self.clone());
        let hold = host.start_hold(&telemetry::Cx::NONE).await?;
        self.lock().hold = hold;
        let io = host.io_state()?;
        let effects = self.lock().core.begin(io);
        self.perform(effects);
        let mut settled = self.done.cell();
        tokio::select! {
            biased;
            dialled = self.dial(&host, &api, &caller) => {
                if let Err(err) = dialled {
                    log::error!(target: lk_agents_abi::LOG_TARGET, "could not dial human agent: {err}");
                    self.abort_dial(&host).await;
                    self.settle(Err(TransferFailed::Dial));
                }
            }
            _ = settled.changed() => {}
        }
        loop {
            let mut settled = self.done.cell();
            if let Some(outcome) = self.lock().core.take_outcome() {
                return outcome;
            }
            if settled.changed().await.is_err() {
                return Err(TransferFailed::NotRunning);
            }
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
                self.done.modify(|()| ());
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
        let (effects, previous) = {
            let mut running = self.0.lock();
            running.watcher = None;
            let phase = match &running.phase {
                RunPhase::Active { host, .. } => RunPhase::Restoring { host: host.clone() },
                _ => return,
            };
            let previous = std::mem::replace(&mut running.phase, phase);
            (running.core.cancel(), previous)
        };
        drop(previous);
        self.0.perform(effects);
        let previous = std::mem::take(&mut self.0.lock().phase);
        drop(previous);
    }
}

impl TransferRun {
    fn lock(&self) -> parking_lot::MutexGuard<'_, Running> {
        self.running.lock()
    }

    fn perform(&self, effects: TransferEffects) {
        let (host, hold) = {
            let mut running = self.lock();
            let host = match &running.phase {
                RunPhase::Active { host, .. } | RunPhase::Restoring { host } => Some(host.clone()),
                RunPhase::Idle => None,
            };
            (host, effects.stop_hold.then(|| running.hold.take()).flatten())
        };
        if let Some(host) = host {
            if let Some(drain) = effects.close_human {
                host.close_human(drain);
            }
            if let Some(hold) = hold {
                hold.stop();
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
            self.done.modify(|()| ());
        }
    }

    /// The human agent's room connected and watched, their session started, and the call
    /// placed; returns once they answered.
    async fn dial(
        self: &Arc<Self>,
        host: &Arc<dyn TransferHost>,
        api: &Arc<dyn TransferApi>,
        caller: &Arc<dyn RtcRoom>,
    ) -> Result<(), DialError> {
        let room_name = human_agent_room_name(&caller.name()?);
        let room = host
            .connect_human(
                &telemetry::Cx::NONE,
                room_name.clone(),
                caller.local().identity()?.to_string(),
            )
            .await?
            .rtc()
            .ok_or_else(|| HostError::new("warm transfer requires a connected human RTC room"))?;
        let events = room.events();
        let this = self.clone();
        {
            let mut running = self.lock();
            running.core.human_connected();
            running.watcher = Some(self.tasks.scoped(
                self.spawner.as_ref(),
                "warm transfer human watcher",
                async move { this.watch_human(events).await },
            ));
        }
        host.start_human(&telemetry::Cx::NONE).await?;
        match &self.plan.origination {
            Origination::Sip { sip_call_to, sip_trunk_id, sip_number, sip_headers, dtmf } => {
                let call = SipCall {
                    sip_trunk_id: sip_trunk_id.clone(),
                    sip_call_to: sip_call_to.clone(),
                    room_name,
                    participant_identity: HUMAN_AGENT_IDENTITY.to_owned(),
                    wait_until_answered: true,
                    sip_number: Some(sip_number.clone()).filter(|s| !s.is_empty()),
                    headers: sip_headers.clone(),
                    dtmf: dtmf.clone().unwrap_or_default(),
                    ringing_timeout: self.plan.ringing_timeout,
                    trunk: self.sip_connection.clone(),
                };
                api.create_sip_participant(call).await.map_err(DialError::Api)?;
            }
            Origination::Twilio(dial) => {
                twilio_originate(
                    api,
                    dial,
                    &room_name,
                    &room,
                    self.plan.ringing_timeout,
                    &*self.clock,
                    Background::new(self.tasks.clone(), self.spawner.clone(), self.clock.clone()),
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
            running.watcher = None;
            running.core.abort_dial()
        };
        let close = effects.close_human.is_some();
        self.perform(effects);
        if close && let Err(err) = host.disconnect_human(&telemetry::Cx::NONE).await {
            log::debug!(target: lk_agents_abi::LOG_TARGET, "leaving the human agent's room failed: {err}");
        }
    }

    /// The human agent's room closing settles the transfer, unless they are being moved.
    async fn watch_human(&self, mut events: lk_agents_abi::voice::rtc::Receiver<RoomEvent>) {
        loop {
            let event = match events.next().await {
                Ok(Some(event)) => event,
                // The room's events ended without its close: nothing will settle it otherwise.
                Ok(None) => {
                    let ended = HostError::new("the human agent's room events ended");
                    self.settle(Err(TransferFailed::Host(ended)));
                    return;
                }
                Err(error) => {
                    self.settle(Err(TransferFailed::Host(HostError::new(error.to_string()))));
                    return;
                }
            };
            if let RoomEvent::Close { reason } = event {
                log::debug!(target: lk_agents_abi::LOG_TARGET, "human agent's room closed: {reason:?}");
                let effects = self.lock().core.room_closed(reason_name(reason));
                if let Some(effects) = effects {
                    self.perform(effects);
                    self.done.modify(|()| ());
                }
                return;
            }
        }
    }

    /// `connect_to_caller`: the human agent moved into the caller's room, the transfer settled,
    /// and the caller's room deleted once the caller or the human agent leaves it.
    async fn connect(self: &Arc<Self>) -> Result<(), TransferFailed> {
        let (api, caller) = {
            let running = self.lock();
            let RunPhase::Active { api, caller, .. } = &running.phase else {
                return Err(TransferFailed::NotRunning);
            };
            (api.clone(), caller.clone())
        };
        let caller_room = caller
            .name()
            .map_err(|error| TransferFailed::Host(HostError::new(error.to_string())))?;
        log::debug!(target: lk_agents_abi::LOG_TARGET, "moving {HUMAN_AGENT_IDENTITY} to caller room {caller_room}");
        let moved = api
            .move_participant(
                human_agent_room_name(&caller_room),
                HUMAN_AGENT_IDENTITY.to_owned(),
                caller_room.clone(),
            )
            .await;
        if let Err(error) = moved {
            self.lock().core.merge_failed();
            return Err(HostError::new(error.to_string()).into());
        }
        self.settle(Ok(HUMAN_AGENT_IDENTITY.to_owned()));
        let mut events = caller.events();
        self.tasks.spawn(self.spawner.as_ref(), "warm transfer caller follower", async move {
            loop {
                let next = events.next().await;
                match follow_merged_call(next.as_ref().map(Option::as_ref)) {
                    MergedCall::Continues => {}
                    MergedCall::Unfollowed => return,
                    MergedCall::Ends { identity } => {
                        log::info!(target: lk_agents_abi::LOG_TARGET, "participant disconnected from caller room: {identity}, closing");
                        if let Err(err) = api.delete_room(caller_room).await {
                            log::warn!(target: lk_agents_abi::LOG_TARGET, "failed to delete the caller's room: {err}");
                        }
                        return;
                    }
                }
            }
        });
        Ok(())
    }
}

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
    #[error(transparent)]
    Api(ApiError),
}

/// One Twilio request; the reply's JSON object.
async fn twilio_send(
    api: &Arc<dyn TransferApi>,
    request: twilio_http::Request,
) -> Result<serde_json::Map<String, Value>, DialError> {
    let (status, body) = api.post_form(request).await.map_err(DialError::Api)?;
    Ok(twilio_http::reply(status, &body)?)
}

/// Create the outbound call, streaming into the connector: its SID.
async fn twilio_create_call(
    api: Arc<dyn TransferApi>,
    dial: TwilioDial,
    from: String,
    twiml: String,
    call_token: Option<String>,
) -> Result<String, DialError> {
    let call = twilio_send(&api, twilio_http::create_call(&dial, from, &twiml, call_token)).await?;
    Ok(twilio_http::call_sid(&call)?)
}

/// A call the transfer gave up on, cancelled, or completed when it was answered meanwhile.
async fn twilio_cancel(api: Arc<dyn TransferApi>, dial: TwilioDial, sid: String) {
    let cancel = twilio_http::end_call(&dial, &sid, TwilioCallEnd::Canceled);
    let cancelled = match twilio_send(&api, cancel).await {
        Err(DialError::Twilio(error)) if twilio::complete_instead(&error) => {
            let complete = twilio_http::end_call(&dial, &sid, TwilioCallEnd::Completed);
            twilio_send(&api, complete).await.map(drop)
        }
        other => other.map(drop),
    };
    if cancelled.is_err() {
        log::warn!(target: lk_agents_abi::LOG_TARGET, "Failed to cancel Twilio call");
    }
}

/// The outbound call while the dial waits on it: still being created, or how its creation ended.
enum Call {
    Creating(tokio::sync::oneshot::Receiver<Result<String, DialError>>),
    Settled(Creation),
}

/// What the dial heard of a creation: a transport failure or a creation task lost may follow a
/// committed call.
fn creation(result: &Result<String, DialError>) -> Creation {
    match result {
        Ok(sid) => Creation::Created(sid.clone()),
        Err(DialError::Twilio(error)) => Creation::Failed(error.clone()),
        Err(_) => Creation::Lost,
    }
}

/// Retires the call when the dial does not get past it, however it ends (a failure or the run
/// dropped), as [`twilio::abandon`] decides: Twilio can create the call after the transfer gave
/// up, or create it and lose the answer.
struct CancelOnDrop {
    background: Background,
    api: Arc<dyn TransferApi>,
    dial: TwilioDial,
    /// The transfer's own human-agent room, the only room its call streams into.
    room: String,
    call: Option<Call>,
}

impl CancelOnDrop {
    /// The creation's outcome, which the guard then holds. Awaited in place, so a dial dropped
    /// meanwhile still retires what the creation goes on to make.
    async fn created(&mut self) -> Result<String, DialError> {
        let Some(Call::Creating(task)) = &mut self.call else {
            unreachable!("a creation is pending")
        };
        let result =
            task.await.unwrap_or_else(|err| Err(DialError::Host(HostError::new(err.to_string()))));
        self.call = Some(Call::Settled(creation(&result)));
        result
    }
}

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        let Some(call) = self.call.take() else { return };
        let (api, dial, room, clock) =
            (self.api.clone(), self.dial.clone(), self.room.clone(), self.background.clock.clone());
        // The dial's owner is being dropped, and the call must still be retired.
        self.background.cleanup("twilio call cancel", async move {
            let creation = match call {
                Call::Settled(creation) => creation,
                Call::Creating(task) => match task.await {
                    Ok(result) => creation(&result),
                    Err(_) => Creation::Lost,
                },
            };
            let retire = async move {
                match twilio::abandon(creation) {
                    Abandon::Nothing => {}
                    Abandon::Cancel(sid) => twilio_cancel(api, dial, sid).await,
                    Abandon::DeleteRoom => {
                        if let Err(error) = api.delete_room(room).await {
                            log::warn!(target: lk_agents_abi::LOG_TARGET, "could not delete the room of a Twilio call whose creation was lost: {error}");
                        }
                    }
                }
            };
            if lk_agents_abi::timeout(&*clock, lk_agents_abi::HOST_TEARDOWN_TIMEOUT, retire)
                .await
                .is_err()
            {
                log::warn!(target: lk_agents_abi::LOG_TARGET, "Twilio call cancel timed out");
            }
        });
    }
}

/// Dial through Twilio: the connector's stream URL, the call created with
/// [`twilio::first_caller`] (again with [`twilio::retry_caller`] when Twilio rejects it), then
/// the human agent's audio track awaited in their room, within `ringing_timeout`.
async fn twilio_originate(
    api: &Arc<dyn TransferApi>,
    dial: &TwilioDial,
    room_name: &str,
    room: &Arc<dyn RtcRoom>,
    ringing_timeout: Option<Duration>,
    clock: &dyn Clock,
    background: Background,
) -> Result<(), DialError> {
    let connect_url = api
        .connect_twilio_call(room_name.to_owned(), HUMAN_AGENT_IDENTITY.to_owned())
        .await
        .map_err(DialError::Api)?;
    let twiml = twilio::twiml(&connect_url);
    let (from, token) = twilio::first_caller(dial);
    let create = |from: String, token: Option<String>| {
        let (reply, created) = tokio::sync::oneshot::channel();
        let (api, dial, twiml) = (api.clone(), dial.clone(), twiml.clone());
        // Outlives the caller: a creation it stops waiting for is cancelled by `CancelOnDrop`.
        background.spawn("twilio call creation", async move {
            let _ = reply.send(twilio_create_call(api, dial, from, twiml, token).await);
        });
        created
    };
    let mut guard = CancelOnDrop {
        background: background.clone(),
        api: api.clone(),
        dial: dial.clone(),
        room: room_name.to_owned(),
        call: Some(Call::Creating(create(from, token.clone()))),
    };
    let mut created = guard.created().await;
    if let Err(DialError::Twilio(error)) = &created
        && let Some(from) = twilio::retry_caller(dial, token.as_deref(), error)
    {
        log::warn!(target: lk_agents_abi::LOG_TARGET, "Twilio rejected preserved caller ID; retrying with business caller ID");
        guard.call = Some(Call::Creating(create(from, None)));
        created = guard.created().await;
    }
    created?;
    let answered = crate::rtc::wait::track_publication(
        room.as_ref(),
        Some(HUMAN_AGENT_IDENTITY.to_owned()),
        Some(vec![TRACK_KIND_AUDIO]),
        crate::rtc::wait::TrackOptions::default(),
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
        sip_connection: Option<Vec<u8>>,
        sip_number: Maybe<String>,
        sip_headers: Option<BTreeMap<String, String>>,
        dtmf: Option<String>,
        ringing_timeout: Option<f64>,
        instructions: Maybe<InstructionsArg>,
        conversation: Option<Vec<lk_agents_abi::ChatItem>>,
        env_sip_trunk: Option<String>,
        env_sip_number: String,
        twilio: Option<TwilioDial>,
    ) -> Result<Self, WarmTransferRefused> {
        let plan = TransferPlan::resolve(TransferConfig {
            sip_call_to,
            sip_trunk_id,
            has_sip_connection: sip_connection.is_some(),
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
        let sip_connection = sip_connection
            .map(|bytes| {
                <livekit_protocol::SipOutboundConfig as prost::Message>::decode(bytes.as_slice())
            })
            .transpose()
            .map_err(|error| WarmTransferRefused::InvalidSipConnection(error.to_string()))?;
        let spawner = (env.spawner)().map_err(|e| HostError::new(e.to_string()))?;
        let run = TransferRun::new(plan, sip_connection, env.clock.clone(), spawner);
        Ok(Self(Arc::new(run)))
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
        let api = Arc::new(ProjectApi::of_server(&host.server_env()?));
        let caller = caller
            .rtc()
            .ok_or_else(|| HostError::new("warm transfer requires a connected caller RTC room"))?;
        self.0.run(host, api, caller).await
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
    use lk_agents_voice::workflows::machine::warm_transfer as p;

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
            None,
            Arc::new(lk_agents_abi::VirtualClock::new()),
            Arc::new(lk_agents_net::task::TokioSpawner::current()),
        ))
    }

    #[derive(Default)]
    struct Holding {
        hold_starts: std::sync::atomic::AtomicUsize,
    }

    impl TransferHost for Holding {
        fn start_hold(
            &self,
            _cx: &telemetry::Cx,
        ) -> lk_agents_abi::BoxFuture<'_, Result<Option<Arc<dyn HoldMusic>>, HostError>> {
            Box::pin(async move {
                self.hold_starts.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                std::future::pending().await
            })
        }
        fn io_state(&self) -> Result<IoEnableState, HostError> {
            unreachable!()
        }
        fn set_io(&self, _: IoEnableState) {
            unreachable!()
        }
        fn connect_human(
            &self,
            _cx: &telemetry::Cx,
            _: String,
            _: String,
        ) -> lk_agents_abi::BoxFuture<
            '_,
            Result<Arc<dyn lk_agents_abi::voice::room_owner::SessionRoom>, HostError>,
        > {
            Box::pin(async move { unreachable!() })
        }
        fn start_human(
            &self,
            _cx: &telemetry::Cx,
        ) -> lk_agents_abi::BoxFuture<'_, Result<(), HostError>> {
            Box::pin(async move { unreachable!() })
        }
        fn close_human(&self, _: bool) {
            unreachable!()
        }
        fn disconnect_human(
            &self,
            _cx: &telemetry::Cx,
        ) -> lk_agents_abi::BoxFuture<'_, Result<(), HostError>> {
            Box::pin(async move { unreachable!() })
        }
        fn server_env(&self) -> Result<BTreeMap<String, String>, HostError> {
            unreachable!()
        }
    }

    /// The project's services over one Twilio account that several transfers share. A creation
    /// commits its call, streaming into the room its TwiML names, before it answers;
    /// `lose_answer` then loses that answer. A call ends when cancelled by its SID, or once its
    /// room is deleted (answered, it has nowhere to stream).
    #[derive(Default)]
    struct FakeApi {
        create_release: Mutex<Option<tokio::sync::oneshot::Receiver<()>>>,
        call_statuses: Mutex<Vec<String>>,
        create_reply: Mutex<Option<(u16, String)>>,
        lose_answer: std::sync::atomic::AtomicBool,
        /// The calls ringing or running: each SID with the room it streams into.
        live: Mutex<Vec<(String, String)>>,
        deleted_rooms: Mutex<Vec<String>>,
        created: std::sync::atomic::AtomicUsize,
    }

    impl FakeApi {
        fn live_rooms(&self) -> Vec<String> {
            self.live.lock().iter().map(|(_, room)| room.clone()).collect()
        }
    }

    impl TransferApi for FakeApi {
        fn create_sip_participant(&self, _: SipCall) -> BoxFuture<'_, Result<(), ApiError>> {
            Box::pin(async move { unreachable!() })
        }
        fn connect_twilio_call(
            &self,
            room: String,
            _: String,
        ) -> BoxFuture<'_, Result<String, ApiError>> {
            Box::pin(async move { Ok(format!("wss://connector.test/{room}")) })
        }
        fn post_form(
            &self,
            request: twilio_http::Request,
        ) -> BoxFuture<'_, Result<(u16, String), ApiError>> {
            Box::pin(async move {
                let field = |name: &str| {
                    let found = request.form.iter().find(|(key, _)| key == name);
                    found.map(|(_, value)| value.clone()).unwrap()
                };
                if !request.url.ends_with("/Calls.json") {
                    let sid = request.url.rsplit('/').next().unwrap().trim_end_matches(".json");
                    self.live.lock().retain(|(live, _)| live != sid);
                    self.call_statuses.lock().push(field("Status"));
                    return Ok((200, "{}".to_owned()));
                }
                let release = self.create_release.lock().take();
                if let Some(release) = release {
                    release.await.expect("the test releases the creation");
                }
                let n = self.created.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                let sid = format!("CA{n}");
                let reply = self.create_reply.lock().take();
                let reply = reply.unwrap_or((200, serde_json::json!({ "sid": sid }).to_string()));
                if (200..300).contains(&reply.0) {
                    let twiml = field("Twiml");
                    let room = twiml.split("connector.test/").nth(1).unwrap();
                    let room = room.split('"').next().unwrap().to_owned();
                    self.live.lock().push((sid, room));
                }
                if self.lose_answer.load(std::sync::atomic::Ordering::SeqCst) {
                    return Err(ApiError::connection(None));
                }
                Ok(reply)
            })
        }
        fn move_participant(
            &self,
            _: String,
            _: String,
            _: String,
        ) -> BoxFuture<'_, Result<(), ApiError>> {
            Box::pin(async move { unreachable!() })
        }
        fn delete_room(&self, room: String) -> BoxFuture<'_, Result<(), ApiError>> {
            Box::pin(async move {
                self.live.lock().retain(|(_, live)| *live != room);
                self.deleted_rooms.lock().push(room);
                Ok(())
            })
        }
    }

    fn dial() -> TwilioDial {
        TwilioDial {
            phone_number: "+15550100".to_owned(),
            from_number: "+15550200".to_owned(),
            account_sid: "account".to_owned(),
            auth_token: "secret".to_owned(),
            call_token: None,
            original_caller_number: None,
        }
    }

    /// Until `done` holds of the fake, as the dial's background cleanup runs.
    async fn until(api: &FakeApi, done: impl Fn(&FakeApi) -> bool) {
        for _ in 0..64 {
            if done(api) {
                return;
            }
            tokio::task::yield_now().await;
        }
        panic!("the dial's cleanup did not run");
    }

    #[tokio::test]
    async fn a_second_run_cannot_replace_the_first_runs_resources() {
        use std::{future::Future, task::Poll};
        let transfer = transfer();
        let host = Arc::new(Holding::default());
        let api: Arc<dyn TransferApi> = Arc::new(FakeApi::default());
        let caller: Arc<dyn RtcRoom> = Arc::new(Untouched);
        let first = transfer.run(host.clone(), api.clone(), caller.clone());
        let mut first = std::pin::pin!(first);
        let polled = std::future::poll_fn(|cx| Poll::Ready(first.as_mut().poll(cx))).await;
        assert!(polled.is_pending());
        let second = transfer.run(host.clone(), api, caller);
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
        let api =
            Arc::new(FakeApi { create_release: Mutex::new(Some(pending)), ..Default::default() });
        let port: Arc<dyn TransferApi> = api.clone();
        let dial = dial();
        let clock = lk_agents_abi::VirtualClock::new();
        let room: Arc<dyn RtcRoom> = Arc::new(Untouched);
        let mut dialing = Box::pin(twilio_originate(
            &port,
            &dial,
            "late-create",
            &room,
            Some(Duration::from_secs(1)),
            &clock,
            Background::new(
                Arc::default(),
                spawner.clone(),
                Arc::new(lk_agents_abi::VirtualClock::new()),
            ),
        ));
        let first =
            std::future::poll_fn(|cx| std::task::Poll::Ready(dialing.as_mut().poll(cx))).await;
        assert!(first.is_pending(), "the host creation must remain pending");
        assert_eq!(spawner.started.load(std::sync::atomic::Ordering::SeqCst), 1);
        drop(dialing);
        assert_eq!(spawner.started.load(std::sync::atomic::Ordering::SeqCst), 2);
        release.send(()).unwrap();
        until(&api, |api| !api.call_statuses.lock().is_empty()).await;
        assert_eq!(*api.call_statuses.lock(), ["canceled"]);
        assert!(api.live.lock().is_empty(), "the known SID is cancelled directly");
        assert!(api.deleted_rooms.lock().is_empty(), "no room is deleted for a known call");
    }

    // Catches: a creation Twilio committed whose answer was lost leaking a ringing leg, or its
    // cleanup reaching another transfer's call from the same caller id to the same number: the
    // dial fails, its own human-agent room is deleted, and the other call rings on.
    #[tokio::test]
    async fn a_lost_creation_deletes_its_own_room_and_leaves_a_concurrent_call() {
        let api = Arc::new(FakeApi::default());
        let port: Arc<dyn TransferApi> = api.clone();
        let clock = lk_agents_abi::VirtualClock::new();
        let room: Arc<dyn RtcRoom> = Arc::new(Untouched);
        let spawner: Arc<dyn Spawner> = Arc::new(lk_agents_net::task::TokioSpawner::current());
        let background =
            Background::new(Arc::default(), spawner, Arc::new(lk_agents_abi::VirtualClock::new()));
        // Another transfer's call from the same caller id to the same number, still ringing.
        api.live.lock().push(("CA-other".to_owned(), "other".to_owned()));

        api.lose_answer.store(true, std::sync::atomic::Ordering::SeqCst);
        let dial = dial();
        let lost = twilio_originate(&port, &dial, "lost", &room, None, &clock, background);
        assert!(matches!(lost.await, Err(DialError::Api(_))), "the dial fails on the lost answer");
        until(&api, |api| !api.deleted_rooms.lock().is_empty()).await;
        assert_eq!(*api.deleted_rooms.lock(), ["lost"]);
        assert_eq!(api.live_rooms(), ["other"], "only the lost creation's call ended");
        assert!(api.call_statuses.lock().is_empty(), "no call was cancelled by a guessed SID");
    }

    // Catches: a Twilio redirect or error reply taken as a created call, or its error text lost:
    // the first response is Twilio's answer, and a non-2xx one names its status and code.
    #[tokio::test]
    async fn a_twilio_reply_outside_2xx_fails_the_request_with_its_status_and_code() {
        let api = FakeApi::default();
        *api.create_reply.lock() = Some((302, r#"{"code":20003}"#.to_owned()));
        let api: Arc<dyn TransferApi> = Arc::new(api);
        let error =
            twilio_create_call(api, dial(), "+15550200".to_owned(), "<Response/>".into(), None)
                .await
                .unwrap_err();
        assert_eq!(error.to_string(), "Twilio request failed (HTTP 302, code 20003)");
    }
}
