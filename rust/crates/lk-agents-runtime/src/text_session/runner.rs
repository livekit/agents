//! Declared runner and its cancellable, bounded update source.
use super::*;
use lk_agents_abi::agent::run::InvalidRun;
use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::resources::{LlmRef, ResourceId};
use lk_agents_abi::text_session::{TextSessionDelivery, TextSessionRunner};
use lk_agents_abi::utils::clock::RealClock;
use lk_agents_abi::{ChatItem, Closed, StreamSource, Ts};
use lk_agents_core::text_session::SessionAgent;
use lk_agents_net::env::ProcessEnv;
use lk_agents_net::task::{TaskSet, TokioSpawner};
use std::sync::Mutex;

#[derive(Clone)]
struct Control {
    release: mpsc::Sender<Release>,
    progress: mpsc::Sender<(String, u64, String)>,
}

pub struct Runner {
    config: TextSessionConfig,
    agents: BTreeMap<String, TextSessionAgent>,
    control: Mutex<Option<Control>>,
    runtime: fn() -> Result<tokio::runtime::Handle, String>,
}

impl TextSessionRunner for Runner {
    type Updates = Updates;
    type Env = ProcessEnv;

    fn new(
        env: &ProcessEnv,
        root_agent: String,
        agents: Vec<TextSessionAgent>,
        max_depth: usize,
    ) -> Result<Self, InvalidRun> {
        let config = TextSessionConfig::from_agents(
            root_agent,
            agents.iter().map(|agent| SessionAgent {
                definition: agent.definition.clone(),
                llm: Some(LlmRef { id: ResourceId(0), provider: agent.llm.id() }),
                output_type_explicit: agent.output_type_explicit,
                agent_tools: agent.agent_tools.clone(),
            }),
            max_depth,
        )
        .map_err(|e| InvalidRun(e.to_string()))?;
        let resources =
            agents.into_iter().map(|agent| (agent.definition.name.clone(), agent)).collect();
        Ok(Self { config, agents: resources, control: Mutex::new(None), runtime: env.runtime })
    }

    fn run(&self, request: TextSessionRequest) -> Result<Updates, InvalidRun> {
        let runtime = (self.runtime)().map_err(InvalidRun)?;
        let mut current = self.control.lock().unwrap_or_else(|e| e.into_inner());
        if current.as_ref().is_some_and(|c| !c.release.is_closed()) {
            return Err(InvalidRun("this runner already has an active request".into()));
        }
        let (updates, control) = start(
            &runtime,
            self.config.clone(),
            self.agents.clone(),
            request,
            Arc::new(RealClock::new()),
        )?;
        *current = Some(control);
        Ok(updates)
    }

    fn release(&self, delivery: TextSessionDelivery) -> Result<(), InvalidRun> {
        let current = self.control.lock().unwrap_or_else(|e| e.into_inner());
        let control =
            current.as_ref().ok_or_else(|| InvalidRun("session request is not running".into()))?;
        control
            .release
            .try_send(Release {
                invocation_id: delivery.invocation_id,
                delivered: delivery.messages,
            })
            .map_err(|e| InvalidRun(e.to_string()))
    }

    fn progress(
        &self,
        invocation_id: &str,
        operation: u64,
        message: String,
    ) -> Result<(), InvalidRun> {
        let current = self.control.lock().unwrap_or_else(|e| e.into_inner());
        let control =
            current.as_ref().ok_or_else(|| InvalidRun("session request is not running".into()))?;
        control
            .progress
            .try_send((invocation_id.into(), operation, message))
            .map_err(|e| InvalidRun(e.to_string()))
    }
}

fn start(
    runtime: &tokio::runtime::Handle,
    config: TextSessionConfig,
    agents: BTreeMap<String, TextSessionAgent>,
    request: TextSessionRequest,
    clock: Arc<dyn Clock>,
) -> Result<(Updates, Control), InvalidRun> {
    let ChatItem::Message(message) = request.message.clone() else {
        return Err(InvalidRun("a session message must be a ChatMessage".into()));
    };
    let session = TextSession::new(
        config,
        TextSessionInput {
            session_id: request.session_id.clone(),
            execution_id: request.execution_id.clone(),
            task_id: request.task_id.clone(),
            root_scope: request.root_scope.clone(),
            available_scopes: request.contexts.keys().cloned().collect(),
            separate_contexts: request.separate_contexts,
            state: request.state.clone(),
            wall: WallAnchor::of(&*clock),
            cleanup_timeout: lk_agents_abi::HOST_TEARDOWN_TIMEOUT,
            delegated: request.delegated,
        },
    )
    .map_err(|e| InvalidRun(e.to_string()))?
    .with_checkpoints();
    let (cancel, cancelled) = oneshot::channel();
    let (updates, receiver) = mpsc::channel(UPDATE_QUEUE);
    let (events_tx, events) = mpsc::channel(64);
    let (progress_tx, progress) = mpsc::channel(32);
    let (release_tx, releases) = mpsc::channel(1);
    let driver = Driver {
        session,
        agents,
        request,
        clock: clock.clone(),
        runtime: runtime.clone(),
        events_tx,
        updates,
        work: BTreeMap::new(),
        pending: BTreeMap::new(),
        releases: VecDeque::new(),
        output: None,
        finished: false,
    };
    let driving = Arc::new(TaskSet::default());
    let spawner = TokioSpawner::new(runtime.clone());
    driving.spawn(
        &spawner,
        "text session",
        driver.drive(message, events, progress, releases, cancelled),
    );
    Ok((
        Updates { cancel: Some(cancel), receiver, clock, _driver: driving },
        Control { release: release_tx, progress: progress_tx },
    ))
}

pub struct Updates {
    cancel: Option<oneshot::Sender<()>>,
    receiver: mpsc::Receiver<Result<TextSessionUpdate, ProviderError>>,
    clock: Arc<dyn Clock>,
    /// The driver's owner: dropping the updates cancels the session through `cancel`, and the
    /// driver ends after its cleanup.
    _driver: Arc<TaskSet>,
}

impl Drop for Updates {
    fn drop(&mut self) {
        if let Some(cancel) = self.cancel.take() {
            let _ = cancel.send(());
        }
    }
}

impl StreamSource<TextSessionUpdate> for Updates {
    fn next(
        &mut self,
    ) -> lk_agents_abi::BoxFuture<'_, Result<Option<TextSessionUpdate>, ProviderError>> {
        Box::pin(async move {
            match self.receiver.recv().await {
                Some(update) => update.map(Some),
                None => Ok(None),
            }
        })
    }
    fn cancel_safe(&self) -> bool {
        true
    }
    fn close(mut self: Box<Self>, deadline: Ts) -> lk_agents_abi::BoxFuture<'static, Closed> {
        Box::pin(async move {
            if let Some(cancel) = self.cancel.take() {
                let _ = cancel.send(());
            }
            let clock = self.clock.clone();
            Closed::bounded(&*clock, deadline, async move {
                while self.receiver.recv().await.is_some() {}
            })
            .await
        })
    }
}
