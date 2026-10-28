#![cfg_attr(
    feature = "python",
    allow(
        clippy::disallowed_types,
        reason = "S0: generated host siblings need module scope; exact type sites are ratcheted."
    )
)]

//! The warm transfer (`beta.workflows.WarmTransfer`): while the caller waits on hold, a second
//! session in a room of its own dials a human agent, briefs them on the call, and either moves
//! them into the caller's room or reports why not. The core runs the sequence and decides; the
//! host (`TransferHost`) connects rooms, starts the second session, plays the hold music and
//! makes the LiveKit and Twilio requests.

use crate::workflows::instructions::render;
use crate::workflows::prompts::warm_transfer as p;

use lk_agents_abi::errors::HostError;
use lk_agents_abi::voice::workflows::{
    InstructionsArg, IoEnableState, SipParticipantRequest, TwilioDial,
};
use lk_agents_abi::{ChatContext, HostRef, Maybe};
use lk_agents_rtc::{DisconnectReason, ParticipantKind, RoomEvent, RtcRoom, TRACK_KIND_AUDIO};
use serde_json::{Value, json};
#[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
use std::collections::HashMap;
#[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
use std::sync::{Arc, Mutex};
use std::time::Duration;

const LOG: &str = "livekit.agents";

/// The SIP participant's identity, the second session's `participant_identity`, and what
/// `move_participant` names.
#[lk_agents_abi::host(export(name = "WARM_TRANSFER_IDENTITY"))]
pub const HUMAN_AGENT_IDENTITY: &str = "human-agent-sip";

/// Twilio reports no-answer and failure only through status webhooks, which nothing here reads,
/// so an unanswered call is given up after this long unless `ringing_timeout` says otherwise.
#[lk_agents_abi::host(export(name = "TWILIO_RINGING_TIMEOUT"))]
pub const TWILIO_RINGING_TIMEOUT: f64 = 30.0;

const TWILIO_API_BASE: &str = "https://api.twilio.com/2010-04-01";

/// The human agent's room: the caller's, suffixed.
pub fn human_agent_room_name(caller_room: &str) -> String {
    format!("{caller_room}-human-agent")
}

/// The caller conversation as the briefing reads it: the caller's and the assistant's turns that
/// have text, one per line.
pub fn format_conversation_history(chat_ctx: &ChatContext) -> String {
    let mut out = String::new();
    for msg in chat_ctx.messages() {
        let role = match msg.role {
            lk_agents_abi::Role::User => "Caller",
            lk_agents_abi::Role::Assistant => "Assistant",
            lk_agents_abi::Role::System | lk_agents_abi::Role::Developer => continue,
        };
        let Some(text) = msg.raw_text_content() else { continue };
        if text.is_empty() {
            continue;
        }
        out.push_str(role);
        out.push_str(": ");
        out.push_str(&text);
        out.push('\n');
    }
    out
}

/// The caller session's io while the transfer runs: off, then back to what it was. The originals
/// are read once, so a leg off before the transfer stays off after it.
#[derive(Debug, Default)]
pub struct IoGate {
    original: Option<IoEnableState>,
}

impl IoGate {
    /// What the setters should be given. `current` is read only the first time.
    pub fn desired(&mut self, enabled: bool, current: IoEnableState) -> IoEnableState {
        let original = *self.original.get_or_insert(current);
        IoEnableState {
            audio_input: enabled && original.audio_input,
            audio_output: enabled && original.audio_output,
            transcription_output: enabled && original.transcription_output,
        }
    }
}

/// What dials the human agent.
#[derive(Debug, Clone, PartialEq)]
pub enum Origination {
    /// `CreateSIPParticipant` on a trunk or a custom SIP connection, waiting until answered.
    Sip {
        sip_call_to: String,
        sip_trunk_id: Option<String>,
        sip_number: String,
        #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
        sip_headers: HashMap<String, String>,
        dtmf: Option<String>,
    },
    /// The LiveKit Twilio connector's stream, called through Twilio's Calls API.
    Twilio(TwilioDial),
}

/// A transfer the reference refuses to build (`ValueError`).
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[lk_agents_abi::host(error(kind = "ValueError"))]
pub enum WarmTransferRefused {
    #[error("`sip_call_to` must be set")]
    NoSipCallTo,
    #[error(
        "`LIVEKIT_SIP_OUTBOUND_TRUNK` environment variable, `sip_trunk_id`, or `sip_connection` \
         must be set"
    )]
    NoTrunk,
    #[error(
        "Twilio credentials are required: pass `twilio_account_sid` and `twilio_auth_token` or set \
         the TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN environment variables"
    )]
    NoTwilioCredentials,
    #[error("twilio_call_token requires original_caller_number")]
    CallTokenWithoutCaller,
}

/// Why a transfer did not connect: the person's agent hears it as the tool's error.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[lk_agents_abi::host(error(kind = "livekit.agents.llm.tool_context.ToolError"))]
pub enum TransferFailed {
    #[error("could not dial human agent")]
    Dial,
    #[error("human agent declined to connect: {0}")]
    Declined(String),
    #[error("voicemail detected")]
    Voicemail,
    #[error("room closed: {0}")]
    RoomClosed(String),
    #[error("the transfer is not running")]
    #[host(kind = "RuntimeError")]
    NotRunning,
    #[error("unknown transfer tool {0:?}")]
    #[host(kind = "ValueError")]
    UnknownTool(String),
    #[error(transparent)]
    #[host(delegate)]
    Host(#[from] HostError),
}

#[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
/// What the host does for a transfer. Each method is one SDK call or two.
#[lk_agents_abi::host(proxy(live, cancel_on_drop))]
#[async_trait::async_trait]
pub trait TransferHost: Send + Sync {
    /// Start the hold music on the caller's room, looping: what stops it, or nothing when there is
    /// no hold music.
    async fn start_hold(&self) -> Result<Option<HostRef>, HostError>;
    /// Stop the hold music `start_hold` started.
    fn stop_hold(&self, hold: HostRef);
    /// The caller session's io flags, now.
    fn io_state(&self) -> IoEnableState;
    /// Set the caller session's io flags.
    fn set_io(&self, state: IoEnableState);
    /// Connect to the human agent's room as `identity` (the caller-side agent's).
    async fn connect_human(
        &self,
        room_name: String,
        identity: String,
    ) -> Result<Arc<dyn RtcRoom>, HostError>;
    /// Start the human agent's session (the briefing agent) on the connected room.
    async fn start_human(&self) -> Result<(), HostError>;
    /// Shut the human agent's session down, after its speech when `drain`.
    fn close_human(&self, drain: bool);
    /// Disconnect from the human agent's room.
    async fn disconnect_human(&self) -> Result<(), HostError>;
    /// `api.sip.create_sip_participant(request)`.
    async fn create_sip_participant(&self, request: SipParticipantRequest)
    -> Result<(), HostError>;
    /// `api.connector.connect_twilio_call(...)`, outbound into `room_name` as `identity`: the
    /// stream URL Twilio connects to.
    async fn connect_twilio_call(
        &self,
        room_name: String,
        identity: String,
    ) -> Result<String, HostError>;
    /// A form `POST`, no redirect followed: the status and the body.
    async fn http_post(
        &self,
        url: String,
        headers: HashMap<String, String>,
        form: Vec<(String, String)>,
    ) -> Result<(u16, String), HostError>;
    /// `api.room.move_participant(...)`.
    async fn move_participant(
        &self,
        room: String,
        identity: String,
        destination: String,
    ) -> Result<(), HostError>;
    /// Delete the caller's room: the call is over.
    async fn delete_caller_room(&self) -> Result<(), HostError>;
}

/// The run's shared state: who settles it, and what the tools reach.
#[derive(Default)]
struct Running {
    host: Option<Arc<dyn TransferHost>>,
    caller: Option<Arc<dyn RtcRoom>>,
    settled: bool,
    /// The human agent's identity once they joined the caller, or why not.
    outcome: Option<Result<String, TransferFailed>>,
    /// The human agent's session is up and not yet closed.
    human_open: bool,
    hold: Option<HostRef>,
    /// The human agent is being moved: their room closing is no longer a failure.
    merging: bool,
    watcher: Option<tokio::task::JoinHandle<()>>,
    io: IoGate,
}

/// `_core.WarmTransfer`: the transfer's plan (its briefing instructions and tools, how it dials)
/// and its run.
pub struct HostWarmTransfer {
    instructions: String,
    origination: Origination,
    ringing_timeout: Option<Duration>,
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    running: Mutex<Running>,
    done: tokio::sync::Notify,
}

/// `[(role, text)]` as a context with just what [`format_conversation_history`] reads. Anything
/// that is not a role literal reads as `developer`, which the briefing skips.
fn conversation_of(items: Vec<(String, String)>) -> ChatContext {
    use lk_agents_abi::{ChatContent, ChatItem, ChatMessage, Role};
    ChatContext {
        items: items
            .into_iter()
            .enumerate()
            .map(|(i, (role, text))| {
                let role = Role::from_literal(&role).unwrap_or(Role::Developer);
                ChatItem::Message(ChatMessage::new(
                    format!("history-{i}"),
                    role,
                    vec![ChatContent::Text(text)],
                    std::time::UNIX_EPOCH,
                ))
            })
            .collect(),
    }
}

/// `connect_to_caller`, `decline_transfer` and `voicemail_detected`, as raw function schemas.
fn transfer_tools() -> Vec<Value> {
    let tool = |name: &str, description: &str, reason: Option<&str>| {
        let (properties, required) = match reason {
            Some(d) => {
                (json!({ "reason": { "type": "string", "description": d } }), json!(["reason"]))
            }
            None => (json!({}), json!([])),
        };
        json!({
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": false,
            },
        })
    };
    vec![
        tool(p::CONNECT_TOOL, p::CONNECT_DESC, None),
        tool(p::DECLINE_TOOL, p::DECLINE_DESC, Some(p::DECLINE_ARG_REASON)),
        tool(p::VOICEMAIL_TOOL, p::VOICEMAIL_DESC, None),
    ]
}

/// The briefing instructions: the template over the persona, the extra section and the caller
/// conversation, unless fixed instructions were given.
fn briefing(
    instructions: Option<InstructionsArg>,
    conversation: Option<Vec<(String, String)>>,
) -> String {
    let (persona, extra) = match instructions {
        Some(InstructionsArg::Fixed(text)) => return text,
        Some(InstructionsArg::Workflow(sections)) => (sections.persona.given(), sections.extra),
        None => (None, String::new()),
    };
    let history = conversation.map(conversation_of).as_ref().map(format_conversation_history);
    render(
        p::INSTRUCTIONS_TEMPLATE,
        &[
            ("persona", persona.as_deref().unwrap_or(p::PERSONA)),
            ("extra", &extra),
            ("_conversation_history", &history.unwrap_or_default()),
        ],
    )
}

#[lk_agents_abi::host(object(name = "WarmTransfer", cell = "arc"))]
impl HostWarmTransfer {
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    /// An explicit `sip_trunk_id=None` means no trunk whatever the environment says; an absent
    /// one is the environment's trunk, unless a SIP connection is given. `twilio` dials through
    /// the Twilio connector instead of SIP. `instructions` is fixed instructions, or a
    /// `WorkflowInstructions` whose persona and extra fill the built-in briefing; `conversation`
    /// is the caller conversation as `[(role, text)]`, which the briefing summarises.
    #[host(new(kw_only, defaults(has_sip_connection = false, env_sip_number = String::new())))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        sip_call_to: Option<String>,
        sip_trunk_id: Maybe<Option<String>>,
        has_sip_connection: bool,
        sip_number: Maybe<String>,
        sip_headers: Option<HashMap<String, String>>,
        dtmf: Option<String>,
        ringing_timeout: Option<f64>,
        instructions: Maybe<InstructionsArg>,
        conversation: Option<Vec<(String, String)>>,
        env_sip_trunk: Option<String>,
        env_sip_number: String,
        twilio: Option<TwilioDial>,
    ) -> Result<Self, WarmTransferRefused> {
        let instructions = briefing(instructions.given(), conversation);
        let origination = match twilio {
            Some(t) => {
                if t.account_sid.is_empty() || t.auth_token.is_empty() {
                    return Err(WarmTransferRefused::NoTwilioCredentials);
                }
                if t.call_token.as_deref().is_some_and(|c| !c.is_empty())
                    && t.original_caller_number.as_deref().is_none_or(str::is_empty)
                {
                    return Err(WarmTransferRefused::CallTokenWithoutCaller);
                }
                Origination::Twilio(t)
            }
            None => {
                let sip_call_to = sip_call_to
                    .filter(|s| !s.is_empty())
                    .ok_or(WarmTransferRefused::NoSipCallTo)?;
                let sip_trunk_id = match sip_trunk_id {
                    Maybe::Given(id) => id,
                    Maybe::NotGiven if has_sip_connection => None,
                    Maybe::NotGiven => env_sip_trunk,
                };
                if sip_trunk_id.is_none() && !has_sip_connection {
                    return Err(WarmTransferRefused::NoTrunk);
                }
                Origination::Sip {
                    sip_call_to,
                    sip_trunk_id,
                    sip_number: sip_number.given().unwrap_or(env_sip_number),
                    sip_headers: sip_headers.unwrap_or_default(),
                    dtmf,
                }
            }
        };
        Ok(HostWarmTransfer {
            instructions,
            origination,
            ringing_timeout: ringing_timeout.map(|s| Duration::from_secs_f64(s.max(0.0))),
            running: Mutex::new(Running::default()),
            done: tokio::sync::Notify::new(),
        })
    }

    /// The briefing agent's static instructions.
    #[host(getter)]
    pub fn instructions(&self) -> String {
        self.instructions.clone()
    }

    /// The briefing agent's three tools, as raw function schemas, each answered by
    /// [`call_tool`](Self::call_tool). All three are hidden while the briefing is entered:
    /// connecting then would cut the summary off.
    #[host(method)]
    pub fn tools(&self) -> Vec<Value> {
        transfer_tools()
    }

    #[allow(clippy::disallowed_macros, reason = "S0: migrate live execution/state to runtime")]
    /// The transfer: the caller on hold with their session's io off, the human agent dialled and
    /// briefed, until one of the tools or the human agent's room settles it; the human agent's
    /// identity once they joined the caller. Whatever ends the
    /// run, the human agent's session is closed, the hold stops and the io is restored.
    #[host(method)]
    pub async fn run(
        self: &Arc<Self>,
        host: Arc<dyn TransferHost>,
        caller: Arc<dyn RtcRoom>,
    ) -> Result<String, TransferFailed> {
        {
            let mut running = self.lock();
            *running = Running {
                host: Some(host.clone()),
                caller: Some(caller.clone()),
                ..Running::default()
            };
        }
        let _restore = Restore(self.clone());
        let hold = host.start_hold().await?;
        self.lock().hold = hold;
        self.set_io(&host, false);
        let settled = self.done.notified();
        tokio::pin!(settled);
        settled.as_mut().enable();
        tokio::select! {
            dialled = self.dial(&host, &caller) => {
                if let Err(err) = dialled {
                    log::error!(target: LOG, "could not dial human agent: {err}");
                    self.abort_dial(&host).await;
                    self.settle(Err(TransferFailed::Dial));
                }
            }
            _ = &mut settled => {}
        }
        loop {
            let settled = self.done.notified();
            tokio::pin!(settled);
            settled.as_mut().enable();
            if let Some(outcome) = self.lock().outcome.take() {
                return outcome;
            }
            settled.await;
        }
    }

    /// One of [`tools`](Self::tools), called by the briefing agent: `connect_to_caller` moves
    /// the human agent into the caller's room and settles the transfer, `decline_transfer` and
    /// `voicemail_detected` settle it as failed.
    #[host(method)]
    pub async fn call_tool(
        self: &Arc<Self>,
        name: String,
        arguments: Value,
    ) -> Result<(), TransferFailed> {
        match name.as_str() {
            p::CONNECT_TOOL => self.connect().await,
            p::DECLINE_TOOL => {
                let reason = arguments.get("reason").and_then(Value::as_str).unwrap_or_default();
                self.settle(Err(TransferFailed::Declined(reason.to_owned())));
                Ok(())
            }
            p::VOICEMAIL_TOOL => {
                self.settle(Err(TransferFailed::Voicemail));
                Ok(())
            }
            _ => Err(TransferFailed::UnknownTool(name)),
        }
    }
}

/// The run's `finally`: the human agent's session closed without drain if still open, the hold
/// stopped, the io restored.
struct Restore(Arc<HostWarmTransfer>);

impl Drop for Restore {
    fn drop(&mut self) {
        let host = {
            let mut running = self.0.lock();
            if let Some(watcher) = running.watcher.take() {
                watcher.abort();
            }
            let close = std::mem::take(&mut running.human_open);
            running.settled = true;
            running.caller = None;
            running.host.take().map(|host| (host, close, running.hold.take()))
        };
        if let Some((host, close, hold)) = host {
            if close {
                host.close_human(false);
            }
            if let Some(hold) = hold {
                host.stop_hold(hold);
            }
            self.0.set_io(&host, true);
        }
    }
}

impl HostWarmTransfer {
    fn lock(&self) -> std::sync::MutexGuard<'_, Running> {
        self.running.lock().unwrap_or_else(|p| p.into_inner())
    }

    fn set_io(&self, host: &Arc<dyn TransferHost>, enabled: bool) {
        let state = self.lock().io.desired(enabled, host.io_state());
        host.set_io(state);
    }

    /// Settle once: the human agent's session closed after its speech, the hold stopped, the io
    /// restored, then the outcome handed to the run. A later settle is ignored.
    fn settle(&self, outcome: Result<String, TransferFailed>) {
        let host = {
            let mut running = self.lock();
            if running.settled || running.host.is_none() {
                return;
            }
            running.settled = true;
            running.outcome = Some(outcome);
            let close = std::mem::take(&mut running.human_open);
            running.host.clone().map(|host| (host, close, running.hold.take()))
        };
        if let Some((host, close, hold)) = host {
            if close {
                host.close_human(true);
            }
            if let Some(hold) = hold {
                host.stop_hold(hold);
            }
            self.set_io(&host, true);
        }
        self.done.notify_waiters();
    }

    #[allow(clippy::disallowed_methods, reason = "S0: migrate live execution/state to runtime")]
    /// The human agent's room connected and watched, their session started, and the call
    /// placed; returns once they answered.
    async fn dial(
        self: &Arc<Self>,
        host: &Arc<dyn TransferHost>,
        caller: &Arc<dyn RtcRoom>,
    ) -> Result<(), DialError> {
        let room_name = human_agent_room_name(&caller.name());
        let room =
            host.connect_human(room_name.clone(), caller.local().identity().to_string()).await?;
        let events = room.events();
        let this = self.clone();
        {
            let mut running = self.lock();
            running.human_open = true;
            running.watcher = Some(tokio::spawn(async move { this.watch_human(events).await }));
        }
        host.start_human().await?;
        match &self.origination {
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
                    ringing_timeout: self.ringing_timeout.map(|d| d.as_secs_f64()),
                };
                host.create_sip_participant(request).await?;
            }
            Origination::Twilio(dial) => {
                twilio_originate(host, dial, &room_name, &room, self.ringing_timeout).await?;
            }
        }
        Ok(())
    }

    /// A failed dial: the human agent's room no longer watched, their session closed without
    /// drain and the room left.
    async fn abort_dial(&self, host: &Arc<dyn TransferHost>) {
        let close = {
            let mut running = self.lock();
            if let Some(watcher) = running.watcher.take() {
                watcher.abort();
            }
            std::mem::take(&mut running.human_open)
        };
        if close {
            host.close_human(false);
            if let Err(err) = host.disconnect_human().await {
                log::debug!(target: LOG, "leaving the human agent's room failed: {err}");
            }
        }
    }

    /// The human agent's room closing settles the transfer, unless they are being moved.
    async fn watch_human(&self, mut events: lk_agents_rtc::Receiver<RoomEvent>) {
        while let Some(event) = events.recv().await {
            if let RoomEvent::Close { reason } = event {
                log::debug!(target: LOG, "human agent's room closed: {reason:?}");
                if !self.lock().merging {
                    self.settle(Err(TransferFailed::RoomClosed(reason_name(reason))));
                }
                return;
            }
        }
    }

    #[allow(clippy::disallowed_methods, reason = "S0: migrate live execution/state to runtime")]
    /// `connect_to_caller`: the human agent moved into the caller's room, the transfer settled,
    /// and the caller's room deleted once the caller or the human agent leaves it.
    async fn connect(self: &Arc<Self>) -> Result<(), TransferFailed> {
        let (host, caller) = {
            let mut running = self.lock();
            let (Some(host), Some(caller)) = (running.host.clone(), running.caller.clone()) else {
                return Err(TransferFailed::NotRunning);
            };
            running.merging = true;
            (host, caller)
        };
        let caller_room = caller.name();
        log::debug!(target: LOG, "moving {HUMAN_AGENT_IDENTITY} to caller room {caller_room}");
        host.move_participant(
            human_agent_room_name(&caller_room),
            HUMAN_AGENT_IDENTITY.to_owned(),
            caller_room,
        )
        .await?;
        self.settle(Ok(HUMAN_AGENT_IDENTITY.to_owned()));
        let mut events = caller.events();
        tokio::spawn(async move {
            while let Some(event) = events.recv().await {
                let RoomEvent::ParticipantDisconnected { participant, .. } = event else {
                    continue;
                };
                if matches!(
                    participant.kind(),
                    ParticipantKind::Connector | ParticipantKind::Sip | ParticipantKind::Standard
                ) {
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
    #[error("Twilio request failed (HTTP {status}, code {})", code.map_or("unknown".to_owned(), |c| c.to_string()))]
    Twilio { status: u16, code: Option<i64> },
    #[error("Twilio call creation returned no call SID")]
    NoCallSid,
    #[error("supervisor did not answer")]
    NoAnswer,
    #[error(transparent)]
    Rtc(#[from] lk_agents_rtc::RtcError),
}

#[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
/// A Twilio request: a form `POST` under the account, with basic auth; the JSON body, `{}` when it
/// is not an object.
async fn twilio_request(
    host: &Arc<dyn TransferHost>,
    dial: &TwilioDial,
    path: &str,
    form: Vec<(String, String)>,
) -> Result<serde_json::Map<String, Value>, DialError> {
    use base64::Engine as _;
    let auth = base64::engine::general_purpose::STANDARD
        .encode(format!("{}:{}", dial.account_sid, dial.auth_token));
    let url = format!("{TWILIO_API_BASE}/Accounts/{}{path}", quote(&dial.account_sid));
    let headers = HashMap::from([("Authorization".to_owned(), format!("Basic {auth}"))]);
    let (status, body) = host.http_post(url, headers, form).await?;
    let body = match serde_json::from_str(&body) {
        Ok(Value::Object(map)) => map,
        _ => serde_json::Map::new(),
    };
    if !(200..300).contains(&status) {
        return Err(DialError::Twilio { status, code: body.get("code").and_then(Value::as_i64) });
    }
    Ok(body)
}

/// Create the outbound call, streaming into the connector: its SID.
async fn twilio_create_call(
    host: Arc<dyn TransferHost>,
    dial: TwilioDial,
    from: String,
    twiml: String,
    call_token: Option<String>,
) -> Result<String, DialError> {
    let mut form = vec![
        ("To".to_owned(), dial.phone_number.clone()),
        ("From".to_owned(), from),
        ("Twiml".to_owned(), twiml),
    ];
    form.extend(call_token.map(|t| ("CallToken".to_owned(), t)));
    let call = twilio_request(&host, &dial, "/Calls.json", form).await?;
    match call.get("sid").and_then(Value::as_str) {
        Some(sid) if !sid.is_empty() => Ok(sid.to_owned()),
        _ => Err(DialError::NoCallSid),
    }
}

/// A call the transfer gave up on, cancelled; one answered meanwhile (error 21220) is completed
/// instead.
async fn twilio_cancel(host: Arc<dyn TransferHost>, dial: TwilioDial, sid: String) {
    let path = format!("/Calls/{}.json", quote(&sid));
    let status = |s: &str| vec![("Status".to_owned(), s.to_owned())];
    let cancelled = match twilio_request(&host, &dial, &path, status("canceled")).await {
        Err(DialError::Twilio { status: 400, code: Some(21220) }) => {
            twilio_request(&host, &dial, &path, status("completed")).await.map(drop)
        }
        other => other.map(drop),
    };
    if cancelled.is_err() {
        log::warn!(target: LOG, "Failed to cancel Twilio call");
    }
}

/// The outbound call while the dial waits on it: still being created, or created.
enum Call {
    Creating(tokio::task::JoinHandle<Result<String, DialError>>),
    Created(String),
}

/// Cancels the call when the dial does not get past it, however it ends (a failure or the run
/// dropped): Twilio can create the call after the transfer gave up.
struct CancelOnDrop {
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
    #[allow(clippy::disallowed_methods, reason = "S0: migrate live execution/state to runtime")]
    fn drop(&mut self) {
        let (host, dial) = (self.host.clone(), self.dial.clone());
        match self.call.take() {
            Some(Call::Creating(task)) => {
                tokio::spawn(async move {
                    if let Ok(Ok(sid)) = task.await {
                        twilio_cancel(host, dial, sid).await;
                    }
                });
            }
            Some(Call::Created(sid)) => {
                tokio::spawn(twilio_cancel(host, dial, sid));
            }
            None => {}
        }
    }
}

#[allow(clippy::disallowed_methods, reason = "S0: migrate live execution/state to runtime")]
/// Dial through Twilio: the connector's stream URL, the call created from the caller's number
/// when a call token preserves it (from the business number again if Twilio rejects that caller
/// ID, errors 21210 and 21212, which create no call), then the human agent's audio track awaited
/// in their room, within `ringing_timeout`.
async fn twilio_originate(
    host: &Arc<dyn TransferHost>,
    dial: &TwilioDial,
    room_name: &str,
    room: &Arc<dyn RtcRoom>,
    ringing_timeout: Option<Duration>,
) -> Result<(), DialError> {
    let connect_url =
        host.connect_twilio_call(room_name.to_owned(), HUMAN_AGENT_IDENTITY.to_owned()).await?;
    let twiml = format!(
        "<Response><Connect><Stream url={}/></Connect></Response>",
        quoteattr(&connect_url)
    );
    let token = dial.call_token.clone().filter(|t| !t.is_empty());
    let from = match &token {
        Some(_) => dial.original_caller_number.clone().unwrap_or_default(),
        None => dial.from_number.clone(),
    };
    let create = |from: String, token: Option<String>| {
        tokio::spawn(twilio_create_call(host.clone(), dial.clone(), from, twiml.clone(), token))
    };
    let mut guard = CancelOnDrop {
        host: host.clone(),
        dial: dial.clone(),
        call: Some(Call::Creating(create(from, token.clone()))),
    };
    let mut created = guard.created().await;
    if let Err(DialError::Twilio { status: 400, code: Some(21210 | 21212) }) = &created
        && token.is_some()
    {
        log::warn!(target: LOG, "Twilio rejected preserved caller ID; retrying with business caller ID");
        guard.call = Some(Call::Creating(create(dial.from_number.clone(), None)));
        created = guard.created().await;
    }
    created?;
    let answered = lk_agents_rtc::wait_for_track_publication(
        room.clone(),
        Some(HUMAN_AGENT_IDENTITY.to_owned()),
        Some(vec![TRACK_KIND_AUDIO]),
        false,
        false,
    );
    match ringing_timeout {
        Some(limit) => {
            tokio::time::timeout(limit, answered).await.map_err(|_| DialError::NoAnswer)??
        }
        None => answered.await?,
    };
    guard.call = None;
    Ok(())
}

/// `urllib.parse.quote(s, safe="")`.
fn quote(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        if b.is_ascii_alphanumeric() || b"-._~".contains(&b) {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

/// `xml.sax.saxutils.quoteattr`: escaped and quoted, in single quotes when the value holds a
/// double quote but no single one.
fn quoteattr(s: &str) -> String {
    let escaped = s
        .replace('&', "&amp;")
        .replace('>', "&gt;")
        .replace('<', "&lt;")
        .replace('\n', "&#10;")
        .replace('\r', "&#13;")
        .replace('\t', "&#9;");
    match (escaped.contains('"'), escaped.contains('\'')) {
        (true, true) => format!("\"{}\"", escaped.replace('"', "&quot;")),
        (true, false) => format!("'{escaped}'"),
        _ => format!("\"{escaped}\""),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sip(
        sip_call_to: Option<&str>,
        has_sip_connection: bool,
        env_trunk: Option<&str>,
    ) -> Result<HostWarmTransfer, WarmTransferRefused> {
        HostWarmTransfer::new(
            sip_call_to.map(str::to_owned),
            Maybe::NotGiven,
            has_sip_connection,
            Maybe::NotGiven,
            None,
            None,
            None,
            Maybe::NotGiven,
            None,
            env_trunk.map(str::to_owned),
            String::new(),
            None,
        )
    }

    // Catches: the trunk fallback losing its middle arm (an env trunk overriding an explicit
    // `sip_connection`), or a transfer built with nothing to dial.
    #[test]
    fn an_explicit_sip_connection_beats_the_env_trunk() {
        let t = sip(Some("+15550100"), true, Some("env")).unwrap();
        assert!(matches!(t.origination, Origination::Sip { sip_trunk_id: None, .. }));
        let t = sip(Some("+15550100"), false, Some("env")).unwrap();
        assert!(
            matches!(t.origination, Origination::Sip { sip_trunk_id: Some(ref id), .. } if id == "env")
        );
        assert_eq!(sip(Some("+15550100"), false, None).err(), Some(WarmTransferRefused::NoTrunk));
        assert_eq!(sip(None, false, Some("env")).err(), Some(WarmTransferRefused::NoSipCallTo));
    }

    // Catches: the briefing losing the caller conversation or the persona.
    #[test]
    fn the_briefing_carries_the_conversation() {
        let i = briefing(
            None,
            Some(vec![("user".into(), "my order is late".into()), ("system".into(), "x".into())]),
        );
        assert!(i.contains("Caller: my order is late\n"), "{i}");
        assert!(i.contains("# Identity"));
        assert!(!i.contains("x\n## End"));
    }

    #[test]
    fn twiml_attributes_are_quoted_as_saxutils() {
        assert_eq!(quoteattr("wss://x/y?a=1&b=2"), "\"wss://x/y?a=1&amp;b=2\"");
        assert_eq!(quoteattr("a\"b"), "'a\"b'");
        assert_eq!(quoteattr("a\"b'c"), "\"a&quot;b'c\"");
        assert_eq!(quote("AC12/3"), "AC12%2F3");
    }

    // Catches: a restore that re-enables a leg the caller had off before the transfer.
    #[test]
    fn the_io_gate_restores_what_was_there() {
        let mut gate = IoGate::default();
        let before =
            IoEnableState { audio_input: true, audio_output: false, transcription_output: true };
        let off = gate.desired(false, before);
        assert_eq!(
            off,
            IoEnableState { audio_input: false, audio_output: false, transcription_output: false }
        );
        assert_eq!(gate.desired(true, off), before);
    }
}
