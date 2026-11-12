//! An agent tool's child, run in process as its call's delegate outside a TextSession. Core's
//! `agent_tool` decides what the child starts from and what it is to the call; this runs it on
//! the driver `agent.run()` uses.

use std::sync::Arc;

use parking_lot::Mutex;
use serde_json::{Map, Value};
use tokio_util::sync::CancellationToken;

use lk_agents_abi::a2a::remote::DelegationHost;
use lk_agents_abi::agent::Invocation;
use lk_agents_abi::agent::run::{Output, RunError, RunOutcome, RunResult, RunUpdate, StreamEvent};
use lk_agents_abi::errors::Invalid;
use lk_agents_abi::invoke::CallContext;
use lk_agents_abi::llm::tool_executor::ToolOutcome;
use lk_agents_abi::resources::{LlmRef, ResourceId};
use lk_agents_abi::text_session::{LocalAgentTool, LocalInteraction as Port, TextSessionAgent};
use lk_agents_abi::usage::Usage;
use lk_agents_abi::{
    ApiConnectOptions, ChatItem, ChatMessage, Clock, FunctionCall, HOST_TEARDOWN_TIMEOUT, Role,
    Spawner, WallAnchor,
};
use lk_agents_core::agent_tool::{self, Step};
use lk_agents_core::tool_outcome::Mode;
use lk_agents_core::usage::Ledger;
use lk_agents_core::{Opening, RunInput};
use lk_agents_net::env::ProcessEnv;
use lk_agents_net::signal;

use crate::agent_run::{Progress, RunSetup};
use crate::telemetry;

/// Updates the delegate may lag behind by before the child waits for it.
const UPDATE_QUEUE: usize = 16;

pub struct LocalInteraction {
    child: TextSessionAgent,
    tool: LocalAgentTool,
    call: FunctionCall,
    depth: usize,
    clock: Arc<dyn Clock>,
    shared: Arc<Mutex<Shared>>,
    /// The running delegation's stop and settlement.
    running: Mutex<Option<(CancellationToken, signal::Cell<Option<ToolOutcome>>)>>,
}

#[derive(Default)]
struct Shared {
    extra: Map<String, Value>,
    /// The child's ended invocations.
    usage: Ledger,
    /// The running invocation: its id, progress port and usage.
    live: Option<(String, Progress, signal::Cell<Usage>)>,
}

impl Port for LocalInteraction {
    type Env = ProcessEnv;

    fn new(
        env: &ProcessEnv,
        child: TextSessionAgent,
        tool: LocalAgentTool,
        function_call: ChatItem,
        depth: usize,
    ) -> Result<Self, Invalid> {
        let ChatItem::FunctionCall(call) = function_call else {
            return Err(Invalid("an agent tool's child needs its parent function call".into()));
        };
        lk_agents_core::agent::validate(&child.definition).map_err(|e| Invalid(e.to_string()))?;
        Ok(Self {
            child,
            tool,
            call,
            depth,
            clock: env.clock.clone(),
            shared: Arc::default(),
            running: Mutex::new(None),
        })
    }

    async fn run(
        &self,
        env: &ProcessEnv,
        host: Arc<dyn DelegationHost>,
        arguments: Value,
        history: Vec<ChatItem>,
        live: bool,
        conn_options: ApiConnectOptions,
        context: CallContext,
        cx: telemetry::Cx,
    ) -> Result<ToolOutcome, Invalid> {
        if let Err(refused) = agent_tool::admit(self.depth) {
            return Ok(refused);
        }
        let spawner = (env.spawner)().map_err(|e| Invalid(e.to_string()))?;
        let stop = CancellationToken::new();
        let (done, settled) = signal::once();
        *self.running.lock() = Some((stop.clone(), settled.clone()));
        let delegate = Delegate {
            child: self.child.clone(),
            tool: self.tool.clone(),
            call: self.call.clone(),
            clock: self.clock.clone(),
            shared: self.shared.clone(),
            spawner: spawner.clone(),
            host,
            context,
            cx,
            conn_options,
            stop,
        };
        // A process task: a dropped `run` leaves it to the host, which stops it in the same turn
        // (`cancelled()`), and `settled()` waits for the child's settlement: the task's end, however
        // it ends, resolves or drops `done`.
        env.tasks.spawn_settling(&*spawner, "agent tool", done, async move {
            delegate.drive(arguments, history, live).await
        });
        settled.wait().await.map_err(|_| Invalid("the agent tool's child ended unsettled".into()))
    }

    fn cancelled(&self) {
        if let Some((stop, _)) = &*self.running.lock() {
            stop.cancel();
        }
    }

    async fn settled(&self) {
        let running = self.running.lock().as_ref().map(|(_, settled)| settled.clone());
        if let Some(settled) = running
            && tokio::time::timeout(HOST_TEARDOWN_TIMEOUT, settled.wait()).await.is_err()
        {
            log::error!(target: lk_agents_abi::LOG_TARGET, "agent tool cleanup incomplete");
        }
    }

    fn extra(&self) -> Map<String, Value> {
        self.shared.lock().extra.clone()
    }

    fn usage(&self) -> Usage {
        let shared = self.shared.lock();
        let mut usage = shared.usage.clone();
        if let Some((_, _, live)) = &shared.live {
            usage.add(&live.get());
        }
        usage.snapshot()
    }

    fn progress(
        &self,
        invocation_id: &str,
        operation: u64,
        message: String,
    ) -> Result<(), Invalid> {
        let shared = self.shared.lock();
        match &shared.live {
            Some((id, progress, _)) if id == invocation_id => {
                // A full queue drops the update, as for a direct run's.
                let _ = progress.report(operation, message);
                Ok(())
            }
            _ => Err(Invalid(format!("invocation {invocation_id} is not running"))),
        }
    }
}

/// One running delegation: the child's invocations until it settles for the call.
struct Delegate {
    child: TextSessionAgent,
    tool: LocalAgentTool,
    call: FunctionCall,
    clock: Arc<dyn Clock>,
    shared: Arc<Mutex<Shared>>,
    spawner: Arc<dyn Spawner>,
    host: Arc<dyn DelegationHost>,
    context: CallContext,
    cx: telemetry::Cx,
    conn_options: ApiConnectOptions,
    stop: CancellationToken,
}

impl Delegate {
    async fn drive(&self, arguments: Value, parent: Vec<ChatItem>, live: bool) -> ToolOutcome {
        let agent = self.child.definition.name.clone();
        let mut history = agent_tool::opening(&parent, self.tool.chat_history);
        let opening = history.len();
        history.extend(agent_tool::retained(&parent, &self.call, &agent));
        let mut message = agent_tool::task(self.id("task"), &arguments, self.stamp());
        if live && let Err(error) = self.host.delegated(&self.cx, &self.context).await {
            return ToolOutcome::Failed(error.to_string());
        }
        loop {
            let (result, output) = match self.invoke(history, message).await {
                Ok(settled) => settled,
                Err(error) => return ToolOutcome::Failed(error),
            };
            let step = agent_tool::settle(&result, output.as_ref());
            let all_messages = result.map(|r| r.all_messages).unwrap_or_default();
            match step {
                Step::Ask { question, verbatim } if live => {
                    let asked = self.host.ask(&self.cx, &self.context, question, verbatim);
                    let answer = tokio::select! {
                        biased;
                        () = self.stop.cancelled() => return ToolOutcome::Cancelled,
                        answer = asked => answer,
                    };
                    let answer = match answer {
                        Ok(answer) => answer,
                        Err(error) => return ToolOutcome::Failed(error.to_string()),
                    };
                    history = all_messages;
                    message =
                        ChatMessage::text(self.id("answer"), Role::User, answer, self.stamp());
                }
                step => {
                    let waiting = matches!(step, Step::Ask { .. })
                        .then(|| all_messages.get(opening..).unwrap_or_default().to_vec());
                    agent_tool::record(&mut self.shared.lock().extra, &agent, waiting);
                    return step.outcome();
                }
            }
        }
    }

    /// One invocation of the child until it settles, under its own `agent_run` span; a stop
    /// cancels it and waits for its settlement.
    async fn invoke(
        &self,
        history: Vec<ChatItem>,
        mut message: ChatMessage,
    ) -> Result<(Result<RunResult, RunError>, Option<Value>), String> {
        let invocation_id = lk_agents_net::strings::shortuuid("run_");
        let execution_id = lk_agents_core::agent_run::execution::bind(
            &mut message,
            &history,
            None,
            &invocation_id,
        );
        let agent_name = self.child.definition.name.clone();
        let entry = Invocation {
            operation: 0,
            speech: None,
            invocation_id: invocation_id.clone(),
            agent_name: agent_name.clone(),
            execution_id: Some(execution_id),
            frame_id: None,
            history: history.iter().cloned().chain([ChatItem::Message(message.clone())]).collect(),
            function_call: None,
            approval: None,
            host_context: Some(self.child.host_context.clone()),
        };
        let input = RunInput {
            invocation_id: invocation_id.clone(),
            id_seed: telemetry::random_seed(),
            opening: Opening::Message(Box::new(message)),
            history,
            answer_tool: None,
            delegated: true,
            mode: Mode::Text,
            llm: Some(LlmRef { id: ResourceId(0), provider: self.child.llm.id() }),
            wall: WallAnchor::of(&*self.clock),
            cleanup_timeout: HOST_TEARDOWN_TIMEOUT,
        };
        let setup = RunSetup {
            llm: self.child.llm.clone(),
            tools_host: self.child.tools_host.clone(),
            host: self.child.host.clone(),
            clock: self.clock.clone(),
            conn_options: self.conn_options,
            entry,
        };
        let operation = telemetry::GenAiOperation::InvokeAgent;
        let span = telemetry::start(&self.cx, telemetry::AgentRun { operation, agent_name });
        let definition = self.child.definition.clone();
        let (active, mut updates) =
            crate::agent_run::start(&self.spawner, definition, input, setup, span, UPDATE_QUEUE)
                .map_err(|e| e.to_string())?;
        self.shared.lock().live = Some((invocation_id, active.progress(), active.usage()));
        let mut cancelling = false;
        let settled = loop {
            let update = tokio::select! {
                biased;
                () = self.stop.cancelled(), if !cancelling => {
                    cancelling = true;
                    active.cancel();
                    continue;
                }
                update = updates.recv() => update,
            };
            match update {
                Some(RunUpdate::Stream(StreamEvent::Progress { message, .. })) => {
                    let extra = self.shared.lock().extra.clone();
                    let reported =
                        self.host.progress(&self.cx, &self.context, Some(message), extra);
                    if let Err(error) = reported.await {
                        log::warn!(target: lk_agents_abi::LOG_TARGET, "an agent tool's progress was not delivered: {error}");
                    }
                }
                Some(RunUpdate::Finished(settled)) => break Ok(*settled),
                Some(RunUpdate::Failed { error, .. }) => break Err(error),
                Some(_) => {}
                None => break Err(RunError::Cancelled),
            }
        };
        let output = match &settled {
            Ok(settled) => match settled.result.outcome {
                RunOutcome::Completed { output: Output::Validated(validation) } => {
                    active.accepted(validation)
                }
                _ => None,
            },
            Err(_) => None,
        };
        let usage = match &settled {
            Ok(settled) => settled.result.usage.clone(),
            Err(_) => active.usage().get(),
        };
        let mut shared = self.shared.lock();
        shared.live = None;
        shared.usage.add(&usage);
        Ok((settled.map(|settled| settled.result), output))
    }

    fn id(&self, kind: &str) -> String {
        format!("{}/{kind}_{}", self.call.call_id, lk_agents_net::strings::shortuuid(""))
    }

    fn stamp(&self) -> std::time::SystemTime {
        let seconds = WallAnchor::of(&*self.clock).seconds(self.clock.now()).max(0.0);
        std::time::UNIX_EPOCH + std::time::Duration::try_from_secs_f64(seconds).unwrap_or_default()
    }
}
