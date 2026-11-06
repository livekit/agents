//! The host bridge and public contract of a warm transfer.

use super::{InstructionsArg, IoEnableState, TwilioDial};
use crate::errors::HostError;
use crate::{HostRef, Maybe};
use std::collections::BTreeMap;
use std::sync::Arc;

/// The SIP participant's identity, the second session's `participant_identity`, and what
/// `move_participant` names.
#[crate::host(export(name = "WARM_TRANSFER_IDENTITY"))]
pub const HUMAN_AGENT_IDENTITY: &str = "human-agent-sip";

/// Twilio reports no-answer and failure only through status webhooks, which nothing here reads,
/// so an unanswered call is given up after this long unless `ringing_timeout` says otherwise.
#[crate::host(export(name = "TWILIO_RINGING_TIMEOUT"))]
pub const TWILIO_RINGING_TIMEOUT: f64 = 30.0;

/// A transfer the reference refuses to build (`ValueError`).
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[crate::host(error(kind = "ValueError"))]
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
    #[error("invalid sip_connection: {0}")]
    InvalidSipConnection(String),
    #[error(transparent)]
    #[host(delegate)]
    Host(#[from] HostError),
}

/// Why a transfer did not connect: the person's agent hears it as the tool's error.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[crate::host(error(kind = "livekit.agents.llm.tool_context.ToolError"))]
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

/// What the transfer executor asks of the SDK (rooms, sessions, hold music), each entered through
/// HostInvoker. Its LiveKit and Twilio requests are the runtime's own.
#[crate::host(proxy(host = "livekit.agents.beta.workflows._TransferHost"))]
pub trait TransferHost: Send + Sync {
    #[host(await, name = "_core_start_hold")]
    fn start_hold(
        &self,
        cx: &crate::telemetry::Cx,
    ) -> crate::BoxFuture<'_, Result<Option<HostRef>, HostError>>;
    #[host(notify, name = "_core_stop_hold")]
    fn stop_hold(&self, hold: HostRef);
    #[host(read_once, name = "_core_io_state")]
    fn io_state(&self) -> IoEnableState;
    #[host(notify, name = "_core_set_io")]
    fn set_io(&self, state: IoEnableState);
    #[host(await, name = "_core_connect_human")]
    fn connect_human(
        &self,
        cx: &crate::telemetry::Cx,
        room_name: String,
        identity: String,
    ) -> crate::BoxFuture<
        '_,
        Result<std::sync::Arc<dyn crate::voice::room_owner::SessionRoom>, HostError>,
    >;
    #[host(await, name = "_core_start_human")]
    fn start_human(&self, cx: &crate::telemetry::Cx)
    -> crate::BoxFuture<'_, Result<(), HostError>>;
    #[host(notify, name = "_core_close_human")]
    fn close_human(&self, drain: bool);
    #[host(await, name = "_core_disconnect_human")]
    fn disconnect_human(
        &self,
        cx: &crate::telemetry::Cx,
    ) -> crate::BoxFuture<'_, Result<(), HostError>>;
    /// The server's settings by variable name: the transfer's LiveKit requests authenticate with
    /// its `LIVEKIT_URL`, `LIVEKIT_API_KEY` and `LIVEKIT_API_SECRET`.
    #[host(call, getter = "_job._server_env")]
    fn server_env(&self) -> Result<BTreeMap<String, String>, HostError>;
}

/// A transfer plan and the runtime ownership of its current dial.
#[crate::host(class(name = "WarmTransfer"))]
pub trait WarmTransfer: Sized + Send + Sync + 'static {
    /// What the hosting process lends the class.
    #[host(env)]
    type Env;

    /// `sip_connection` is the serialized `SIPOutboundConfig` dialled through when no trunk id
    /// names one.
    #[host(new)]
    #[allow(clippy::too_many_arguments, reason = "Preserve the public Python signature")]
    fn new(
        env: &Self::Env,
        #[host(kw, default = None)] sip_call_to: Option<String>,
        sip_trunk_id: Maybe<Option<String>>,
        #[host(default = None)] sip_connection: Option<Vec<u8>>,
        sip_number: Maybe<String>,
        #[host(default = None)] sip_headers: Option<BTreeMap<String, String>>,
        #[host(default = None)] dtmf: Option<String>,
        #[host(default = None)] ringing_timeout: Option<f64>,
        instructions: Maybe<InstructionsArg>,
        #[host(default = None)] conversation: Option<Vec<(String, String)>>,
        #[host(default = None)] env_sip_trunk: Option<String>,
        #[host(default = "")] env_sip_number: String,
        #[host(default = None)] twilio: Option<TwilioDial>,
    ) -> Result<Self, WarmTransferRefused>;
    #[host(getter)]
    fn instructions(&self) -> String;
    #[host(call)]
    fn tools(&self) -> Vec<BTreeMap<String, serde_json::Value>>;
    #[host(await)]
    async fn run(
        &self,
        host: Arc<dyn TransferHost>,
        caller: std::sync::Arc<dyn crate::voice::room_owner::SessionRoom>,
    ) -> Result<String, TransferFailed>;
    #[host(await)]
    async fn call_tool(
        &self,
        name: String,
        arguments: serde_json::Value,
    ) -> Result<(), TransferFailed>;
}
