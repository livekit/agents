use std::sync::Arc;

use lk_agents_abi::a2a::remote::{
    DelegationHost, DelegationInput, RemoteEvent, RemoteInteraction as _,
};
use lk_agents_abi::agent::Invocation;
use lk_agents_abi::errors::HostError;
use lk_agents_abi::invoke::CallContext;
use lk_agents_abi::llm::provider_format::tools::UPDATE_TEMPLATE;
use lk_agents_abi::llm::tool_executor::{ToolOutcome, ToolReport};
use lk_agents_abi::session::{InputRequestRef, SessionCommands};
use lk_agents_abi::telemetry::Cx;
use lk_agents_abi::voice::session::VoiceSession as _;
use lk_agents_abi::{BoxFuture, ChatItem, FunctionCall, SpawnerExt as _, ToolSpec};
use lk_agents_runtime::delegation::remote::{RemoteInteraction, RemoteInvocation};
use lk_agents_runtime::net::a2a_client::AgentClient;
use lk_agents_runtime::voice::VoiceSession;
use parking_lot::Mutex;
use serde_json::{Map, Value};

use super::Registry;
use super::a2a_peer::{ENDPOINT, RemotePeer};
use crate::record::RemoteStep;

// What `RemoteAgent(...).as_tool()` runs on a call (`a2a.py`'s `_RemoteTool._invoke` and its
// `_Delegation` host): the real `RemoteInteraction` over the peer's socket boundary, owned by the
// live session's conversation commands. A cancelled body stops its local wait and joins the
// delegation's cleanup; the remote task runs on.

/// The frontend's remote tool, delegating to [`ENDPOINT`].
pub const REMOTE_TOOL: &str = "desk";

/// What the remote tool declares: the request, stated in full (`_core_delegation_tool`).
pub fn spec() -> ToolSpec {
    ToolSpec {
        name: REMOTE_TOOL.into(),
        provider_tool: None,
        description: "Book with the desk".into(),
        json_schema: serde_json::json!({
            "type": "object",
            "properties": {"request": {"type": "string"}},
            "required": ["request"],
        }),
        flags: Default::default(),
    }
}

/// `_DELEGATED`: the fact that releases the tool's step while the expert works.
const DELEGATED: &str =
    "The remote request is running. Its progress, questions and result arrive separately.";

/// The tool's native identity, as `RunContext` holds it.
#[derive(Clone)]
struct Origin {
    session: Arc<VoiceSession>,
    invocation_id: String,
    operation: u64,
    call: FunctionCall,
}

impl Origin {
    fn report(&self, report: ToolReport) -> Result<(), HostError> {
        self.session
            .tool_update(
                self.invocation_id.clone(),
                self.operation,
                self.call.call_id.clone(),
                report,
            )
            .map_err(|refused| HostError::new(refused.to_string()))
    }
}

/// `_Delegation`: progress, exact output and questions reach the session as the tool's reports.
struct Delegation {
    origin: Origin,
    registry: Arc<Registry>,
    extra: Mutex<Map<String, Value>>,
}

impl DelegationHost for Delegation {
    fn progress<'a>(
        &'a self,
        _cx: &Cx,
        _context: &'a CallContext,
        text: Option<String>,
        extra: Map<String, Value>,
        reply_required: bool,
    ) -> BoxFuture<'a, Result<(), HostError>> {
        self.extra.lock().extend(extra);
        Box::pin(async move {
            let Some(text) = text else { return Ok(()) };
            let origin = &self.origin;
            let message = UPDATE_TEMPLATE
                .replace("{function_name}", &origin.call.name)
                .replace("{message}", &text);
            let outcome = ToolOutcome::Update { message, raw: text, reply_required };
            let receipt = origin
                .session
                .tool_progress(
                    origin.invocation_id.clone(),
                    origin.operation,
                    origin.call.call_id.clone(),
                    outcome.into(),
                )
                .map_err(|refused| HostError::new(refused.to_string()))?;
            receipt.await.map_err(|error| HostError::new(error.to_string()))
        })
    }

    fn say<'a>(
        &'a self,
        _cx: &Cx,
        _context: &'a CallContext,
        text: String,
        extra: Map<String, Value>,
    ) -> BoxFuture<'a, Result<(), HostError>> {
        self.extra.lock().extend(extra);
        let origin = &self.origin;
        // Admission at this hop is enough; the say owner keeps its own delivery.
        let said = origin
            .session
            .tool_say(origin.invocation_id.clone(), origin.operation, origin.call.call_id.clone(), text)
            .map(drop)
            .map_err(|refused| HostError::new(refused.to_string()));
        Box::pin(async move { said })
    }

    fn delegated<'a>(
        &'a self,
        _cx: &Cx,
        _context: &'a CallContext,
    ) -> BoxFuture<'a, Result<(), HostError>> {
        let outcome =
            ToolOutcome::Update { message: DELEGATED.into(), raw: String::new(), reply_required: true };
        let reported = self.origin.report(outcome.into());
        Box::pin(async move { reported })
    }

    fn question<'a>(
        &'a self,
        _cx: &Cx,
        _context: &'a CallContext,
        question: String,
        verbatim: bool,
        input_request: Option<InputRequestRef>,
    ) -> BoxFuture<'a, Result<(), HostError>> {
        self.registry.record_remote(RemoteStep::Presented {
            call: self.origin.call.id.clone(),
            text: question.clone(),
        });
        let reported = self.origin.report(ToolReport {
            outcome: ToolOutcome::InputRequired { question, verbatim },
            input_request: input_request.map(Box::new),
            transition: None,
        });
        Box::pin(async move { reported })
    }

    fn ask<'a>(
        &'a self,
        _cx: &Cx,
        _context: &'a CallContext,
        _question: String,
        _verbatim: bool,
        _input_request: Option<InputRequestRef>,
    ) -> BoxFuture<'a, Result<DelegationInput, HostError>> {
        Box::pin(async { Err(HostError::new("remote questions return through the task driver")) })
    }
}

/// Dropping a body before its delegation returned is the asyncio task's cancellation:
/// `interaction.cancelled()`, then `await interaction.settled()`.
struct Running {
    interaction: Arc<RemoteInteraction>,
    registry: Arc<Registry>,
    call: String,
    returned: bool,
    settled: bool,
}

impl Drop for Running {
    fn drop(&mut self) {
        if self.settled {
            return;
        }
        if !self.returned {
            let call = self.call.clone();
            self.registry.record_remote(RemoteStep::Returned { call, outcome: None });
        }
        if let Err(error) = self.interaction.cancelled() {
            panic!("a remote body's cancellation was refused: {error:?}");
        }
        let (interaction, registry, call) =
            (self.interaction.clone(), self.registry.clone(), self.call.clone());
        self.registry.world.spawn_detached(async move {
            interaction.settled().await;
            registry.record_remote(RemoteStep::BodySettled { call });
        });
    }
}

/// The remote tool's body: its report and the call's transport correlation.
pub(super) async fn run(
    registry: &Arc<Registry>,
    peer: Arc<RemotePeer>,
    invocation: &Invocation,
    call: &FunctionCall,
) -> (ToolOutcome, Map<String, Value>) {
    let session = registry.session().expect("a tool runs in a started session").clone();
    registry.record_remote(RemoteStep::BodyStarted { call: call.id.clone(), call_id: call.call_id.clone() });
    let interaction = match RemoteInteraction::new(
        ENDPOINT.into(),
        ChatItem::FunctionCall(call.clone()),
        String::new(),
        String::new(),
        String::new(),
        None,
    ) {
        Ok(interaction) => Arc::new(interaction),
        Err(error) => {
            registry.record_remote(RemoteStep::BodySettled { call: call.id.clone() });
            return (ToolOutcome::ToolError(error.to_string()), call.extra.clone());
        }
    };
    let clock = registry.clock();
    let env = crate::drive::case_process(clock.clone());
    let client = AgentClient::with_transport(
        ENDPOINT.into(),
        None,
        Default::default(),
        peer,
        (env.spawner)().expect("the case scheduler lends the process spawner"),
        clock,
    )
    .expect("the peer's endpoint is valid");
    let arguments = serde_json::from_str(&call.arguments).unwrap_or(Value::Object(Map::new()));
    let host = Arc::new(Delegation {
        origin: Origin {
            session: session.clone(),
            invocation_id: invocation.invocation_id.clone(),
            operation: invocation.operation,
            call: call.clone(),
        },
        registry: registry.clone(),
        extra: Mutex::new(Map::new()),
    });
    let owner: Arc<dyn SessionCommands> = session;
    let request = RemoteInvocation {
        arguments,
        structured: false,
        history: Vec::new(),
        shared: None,
        configured_token: None,
        live: true,
        host: host.clone(),
        context: CallContext::default(),
        cx: Cx::NONE,
        owner: Some(owner),
        origin: None,
        conn_options: Default::default(),
    };
    let mut running = Running {
        interaction: interaction.clone(),
        registry: registry.clone(),
        call: call.id.clone(),
        returned: false,
        settled: false,
    };
    let event = interaction.run_with_client(&env, client, request).await;
    let outcome = match event {
        Ok(RemoteEvent::Outcome { outcome }) | Ok(RemoteEvent::InputRequired { outcome, .. }) => {
            outcome
        }
        Ok(other) => ToolOutcome::ToolError(format!("the delegation ended on {other:?}")),
        Err(error) => ToolOutcome::ToolError(error.to_string()),
    };
    registry.record_remote(RemoteStep::Returned {
        call: call.id.clone(),
        outcome: Some(format!("{outcome:?}")),
    });
    running.returned = true;
    interaction.settled().await;
    running.settled = true;
    registry.record_remote(RemoteStep::BodySettled { call: call.id.clone() });
    let mut extra = call.extra.clone();
    extra.extend(host.extra.lock().clone());
    if let Ok(more) = interaction.extra() {
        extra.extend(more);
    }
    (outcome, extra)
}
