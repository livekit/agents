//! [`AgentRunner`], the host's class: each `run` starts one invocation ([`start`]) on the process
//! runtime and hands its updates out as a stream whose drop cancels it.

use std::sync::{Arc, Mutex};

use lk_agents_abi::agent::run::{AgentRunner, InvalidRun, RunError, RunRequest, RunUpdate};
use lk_agents_abi::agent::{AgentCallbacks, AgentDefinition, Invocation};
use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::resources::{LlmRef, ResourceId};
use lk_agents_abi::utils::stream::StreamSource;
use lk_agents_abi::{ChatItem, Clock, Closed, Llm, Ts, WallAnchor};
use lk_agents_core::tool_outcome::Mode;
use lk_agents_core::{Opening, RunInput};
use tokio::sync::mpsc;

use super::{ActiveRun, PROGRESS_QUEUE, Progress, ProgressRefused, RunSetup, start};
use lk_agents_net::env::ProcessEnv;

/// Updates the consumer may lag behind by before the run waits for it.
const UPDATE_QUEUE: usize = 16;

pub struct Runner {
    definition: AgentDefinition,
    llm: Arc<dyn Llm>,
    host: Arc<dyn AgentCallbacks>,
    clock: Arc<dyn Clock>,
    /// Each running invocation's progress port, by invocation id; ended ones are pruned.
    running: Mutex<Vec<(String, Progress)>>,
    runtime: fn() -> Result<tokio::runtime::Handle, String>,
}

impl AgentRunner for Runner {
    type Updates = Updates;
    type Env = ProcessEnv;

    fn new(
        env: &ProcessEnv,
        definition: AgentDefinition,
        llm: Arc<dyn Llm>,
        host: Arc<dyn AgentCallbacks>,
    ) -> Result<Self, InvalidRun> {
        lk_agents_core::agent::validate(&definition).map_err(|e| InvalidRun(e.to_string()))?;
        Ok(Runner {
            definition,
            llm,
            host,
            clock: env.clock.clone(),
            running: Mutex::new(Vec::new()),
            runtime: env.runtime,
        })
    }

    fn run(&self, request: RunRequest) -> Result<Updates, InvalidRun> {
        let ChatItem::Message(mut message) = request.message else {
            return Err(InvalidRun("a run's message is a chat message".into()));
        };
        let runtime = (self.runtime)().map_err(InvalidRun)?;
        let execution_id = lk_agents_core::agent_run::execution::bind(
            &mut message,
            &request.history,
            request.execution_id.as_deref(),
            &request.invocation_id,
        );
        let input = RunInput {
            invocation_id: request.invocation_id.clone(),
            opening: Opening::Message(message),
            history: request.history,
            answer_tool: request.answer_tool,
            delegated: request.delegated,
            mode: Mode::Text,
            llm: Some(LlmRef { id: ResourceId(0), provider: self.llm.id() }),
            wall: WallAnchor::of(&*self.clock),
            cleanup_timeout: lk_agents_abi::HOST_TEARDOWN_TIMEOUT,
        };
        let setup = RunSetup {
            tools_host: request.tools_host,
            llm: self.llm.clone(),
            host: self.host.clone(),
            clock: self.clock.clone(),
            conn_options: request.conn_options,
            entry: Invocation {
                operation: 0,
                speech: None,
                invocation_id: request.invocation_id,
                agent_name: self.definition.name.clone(),
                execution_id: Some(execution_id),
                frame_id: None,
                context_scope: None,
                application_context: None,
                history: input
                    .history
                    .iter()
                    .cloned()
                    .chain(input.opening.message().cloned().map(ChatItem::Message))
                    .collect(),
                function_call: None,
                approval: None,
                host_context: Some(request.host_context),
            },
        };
        let invocation_id = input.invocation_id.clone();
        let (handle, updates) =
            start(&runtime, self.definition.clone(), input, setup, UPDATE_QUEUE)
                .map_err(|e| InvalidRun(e.to_string()))?;
        let mut running = self.running.lock().unwrap_or_else(|e| e.into_inner());
        running.retain(|(_, progress)| !progress.ended());
        running.push((invocation_id, handle.progress()));
        Ok(Updates { run: handle, updates, clock: self.clock.clone() })
    }

    fn progress(
        &self,
        invocation_id: &str,
        operation: u64,
        message: String,
    ) -> Result<(), InvalidRun> {
        let running = self.running.lock().unwrap_or_else(|e| e.into_inner());
        let progress = running.iter().find(|(id, _)| id == invocation_id).map(|(_, p)| p);
        match progress.map(|p| p.report(operation, message)) {
            Some(Ok(())) => Ok(()),
            Some(Err(ProgressRefused::Full)) => {
                log::warn!(target: "livekit.agents", "invocation {invocation_id}: a progress update was dropped, {PROGRESS_QUEUE} are waiting");
                Ok(())
            }
            None | Some(Err(ProgressRefused::Ended)) => {
                Err(InvalidRun(format!("invocation {invocation_id} is not running")))
            }
        }
    }
}

/// One invocation's updates. Dropping it cancels the run; its work still settles.
pub struct Updates {
    run: ActiveRun,
    updates: mpsc::Receiver<RunUpdate>,
    clock: Arc<dyn Clock>,
}

impl StreamSource<RunUpdate> for Updates {
    /// A run that failed on its model raises the provider's own error.
    fn next(&mut self) -> lk_agents_abi::BoxFuture<'_, Result<Option<RunUpdate>, ProviderError>> {
        Box::pin(async move {
            let update = self.updates.recv().await;
            if let Some(RunUpdate::Failed(RunError::Model(_))) = &update
                && let Some(error) = self.run.model_error()
            {
                return Err(error);
            }
            Ok(update)
        })
    }

    fn cancel_safe(&self) -> bool {
        true
    }

    /// Cancels the run and waits, until `deadline`, for its work to settle.
    fn close(mut self: Box<Self>, deadline: Ts) -> lk_agents_abi::BoxFuture<'static, Closed> {
        Box::pin(async move {
            self.run.cancel();
            let clock = self.clock.clone();
            Closed::bounded(&*clock, deadline, async move {
                while self.updates.recv().await.is_some() {}
            })
            .await
        })
    }
}
