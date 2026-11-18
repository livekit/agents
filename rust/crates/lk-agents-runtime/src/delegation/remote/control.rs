//! A bound control uses its current serialized owner, or one short state-only fenced owner.

use std::sync::Arc;

use crate::session::{CommandPort, SessionCommand};
use lk_agents_abi::a2a::client::AgentClientError;
use lk_agents_abi::a2a::remote::RemoteConfiguration as _;
use lk_agents_abi::delegation::remote::{
    RemoteCancellationEvidence, RemoteControlScope, RemoteRecovery,
};
use lk_agents_abi::session::SessionCommands;
use lk_agents_abi::session::control::{
    BoundSessionControl, ControlOwner, RemoteCancellation, RemoteCancellationAdmission,
    RemoteControlRef, SessionControlScope,
};
use lk_agents_abi::storage::StorageError;
use lk_agents_abi::{BoxFuture, Closed};
use lk_agents_core::delegation::remote::cancel::{self as policy, CancellationRoute, Control};
use lk_agents_net::a2a_client::AgentClient;
use lk_agents_net::a2a_client::observer::RequestObserver;
use lk_agents_net::env::{ProcessEnv, server_setting};
use lk_agents_net::owned::Owned;
use lk_agents_net::signal;
use lk_agents_telemetry as telemetry;

use super::{DELEGATE, Project, RemoteInteraction, cancel as transport, journal};

pub(super) struct Receipt(signal::Cell<Option<Result<(), AgentClientError>>>);

type CancellationDone = signal::Producer<Option<Result<(), AgentClientError>>>;

impl Receipt {
    pub(super) fn pair() -> (CancellationDone, Arc<dyn RemoteCancellation>) {
        let (done, result) = signal::once();
        (done, Arc::new(Self(result)))
    }
}

impl RemoteCancellation for Receipt {
    fn pending(&self) -> bool {
        self.0.with(Option::is_none) && !self.0.ended()
    }

    fn wait(&self) -> BoxFuture<'_, Result<(), AgentClientError>> {
        Box::pin(async move {
            self.0.wait().await.map_err(|_| {
                unavailable("the native task control owner ended before its receipt")
            })?
        })
    }
}

/// Called synchronously by the generated bound view. Publication contains no I/O; only the
/// selected candidate starts its owned task after the bound store has released its lock.
pub fn cancel(
    env: ProcessEnv,
    bound: Arc<dyn BoundSessionControl>,
    target: RemoteControlRef,
    evidence: Option<RemoteCancellationEvidence>,
) -> Arc<dyn RemoteCancellation> {
    admit(env, bound, target, evidence, None)
}

/// Inject only the HTTP boundary; authority checks, owner admission, core policy, journal
/// checkpoints and resource cleanup run unchanged in deterministic native executor tests.
pub fn cancel_with_transport(
    env: ProcessEnv,
    bound: Arc<dyn BoundSessionControl>,
    target: RemoteControlRef,
    evidence: Option<RemoteCancellationEvidence>,
    http: Arc<dyn lk_agents_net::http::port::HttpExchangePort>,
) -> Arc<dyn RemoteCancellation> {
    admit(env, bound, target, evidence, Some(http))
}

fn admit(
    env: ProcessEnv,
    bound: Arc<dyn BoundSessionControl>,
    target: RemoteControlRef,
    evidence: Option<RemoteCancellationEvidence>,
    http: Option<Arc<dyn lk_agents_net::http::port::HttpExchangePort>>,
) -> Arc<dyn RemoteCancellation> {
    let (done, candidate) = Receipt::pair();
    let selected = match bound.claim_cancellation(&target, candidate.clone()) {
        Ok(selected) => selected,
        Err(error) => {
            done.resolve(Err(storage_error(error)));
            return candidate;
        }
    };
    if !Arc::ptr_eq(&selected, &candidate) {
        return selected;
    }
    let background = match env.background() {
        Ok(background) => background,
        Err(error) => {
            done.resolve(Err(unavailable(error.to_string())));
            return candidate;
        }
    };
    let admitted = candidate.clone();
    let retry_after = policy::retry_after_evidence(evidence.as_ref());
    background.spawn("bound remote task cancellation", async move {
        // Keep the weak published attempt eligible while concrete admitted work owns it,
        // including when every application waiter was canceled or lost its receipt.
        let _admitted = admitted.clone();
        let result =
            execute(&env, &bound, &target, admitted.clone(), retry_after.as_deref(), http).await;
        done.resolve(result);
    });
    candidate
}

async fn execute(
    env: &ProcessEnv,
    bound: &Arc<dyn BoundSessionControl>,
    target: &RemoteControlRef,
    candidate: Arc<dyn RemoteCancellation>,
    retry_after: Option<&str>,
    http: Option<Arc<dyn lk_agents_net::http::port::HttpExchangePort>>,
) -> Result<(), AgentClientError> {
    let current = CurrentScope::read(bound.as_ref(), &target.endpoint)?;
    let acquiring = bound.acquire();
    tokio::pin!(acquiring);
    let owner = tokio::select! {
        biased;
        owner = &mut acquiring => owner.map_err(storage_error)?,
        () = bound.ended() => return Err(unavailable("the bound context store ended")),
    };
    match owner {
        ControlOwner::Active(owner) => {
            let admission = owner
                .cancel_remote(
                    target,
                    current.as_ref().map(|current| &current.proof),
                    candidate,
                    retry_after,
                )
                .await?;
            match admission {
                RemoteCancellationAdmission::Confirmed => Ok(()),
                RemoteCancellationAdmission::Pending(receipt) => {
                    // Its original native driver owns enqueue, transport and cleanup.
                    tokio::select! {
                        biased;
                        result = receipt.wait() => result,
                        () = owner.ended() => Err(unavailable(
                            "the conversation owner ended before cancellation was confirmed",
                        )),
                        () = bound.ended() => Err(unavailable("the bound context store ended")),
                    }
                }
                RemoteCancellationAdmission::Retained { recovery } => {
                    let current = current.ok_or_else(|| {
                        unavailable("the current bound project cannot address this retained task")
                    })?;
                    perform(env, bound.ended(), target, current, recovery, owner, http).await
                }
            }
        }
        ControlOwner::Fenced(scope) => {
            let policy = (|| {
                let remote = lk_agents_core::session::remote::state(
                    scope.snapshot(),
                    &target.remote_id,
                    &target.endpoint,
                )
                .map_err(storage_error)?
                .ok_or_else(|| {
                    unavailable("this bound partition does not retain the remote task")
                })?;
                policy::route(
                    remote,
                    &target.task,
                    false,
                    current.as_ref().map(|current| &current.proof),
                    retry_after,
                )
                .map_err(storage_error)
            })();
            match policy {
                Ok(CancellationRoute::Retained { recovery }) => {
                    let current = current.expect("the core admitted the supplied bound scope");
                    let (port, commands) = CommandPort::channel(16, None);
                    let (finish, finishing) = signal::once();
                    let owner = scope_owner(scope, commands, finishing, bound.clone());
                    let writing = perform(
                        env,
                        bound.ended(),
                        target,
                        current,
                        recovery,
                        Arc::new(port),
                        http,
                    );
                    tokio::pin!(owner, writing);
                    let (result, closed) = tokio::select! {
                        biased;
                        result = &mut writing => {
                            finish.resolve(());
                            (result, owner.await)
                        }
                        closed = &mut owner => {
                            // A failed owner drops its receiver. The pinned client loop
                            // learns that lifetime signal, drains writes and joins cleanup.
                            drop(finish);
                            (writing.await, closed)
                        }
                    };
                    result.and(closed.map_err(storage_error))
                }
                Ok(CancellationRoute::Confirmed) => scope.close().await.map_err(storage_error),
                Ok(CancellationRoute::Configured) => unreachable!("no client was supplied"),
                Err(error) => {
                    let closed = scope.close().await.map_err(storage_error);
                    Err::<(), AgentClientError>(error).and(closed)
                }
            }
        }
    }
}

/// Current worker/server credentials are read for every control. The grant source is the
/// bound store's thread, never an untrusted context ID copied from a retained remote task.
struct CurrentScope {
    project: Project,
    proof: RemoteControlScope,
}

impl CurrentScope {
    fn read(
        bound: &dyn BoundSessionControl,
        endpoint: &str,
    ) -> Result<Option<Self>, AgentClientError> {
        let Some(thread_id) = bound.thread_id().map_err(storage_error)? else { return Ok(None) };
        let project = Project::new(
            server_setting("LIVEKIT_URL").unwrap_or_default(),
            server_setting("LIVEKIT_API_KEY").unwrap_or_default(),
            server_setting("LIVEKIT_API_SECRET").unwrap_or_default(),
        );
        Self::for_thread(&project, endpoint, thread_id)
    }

    /// The project's grant scope for `thread_id`, when `endpoint` is on the project's origin.
    fn for_thread(
        project: &Project,
        endpoint: &str,
        thread_id: String,
    ) -> Result<Option<Self>, AgentClientError> {
        let Some(project) = project.serving(endpoint)? else { return Ok(None) };
        let origin = url::Url::parse(&project.url)
            .map_err(|error| unavailable(error.to_string()))?
            .origin()
            .ascii_serialization();
        Ok(Some(Self {
            proof: RemoteControlScope {
                project_origin: origin,
                project_issuer: project.api_key.clone(),
                caller_identity: DELEGATE.into(),
                thread_id,
            },
            project,
        }))
    }

    fn token(self, endpoint: &str) -> Result<String, AgentClientError> {
        RemoteInteraction::_core_remote_token(
            endpoint.into(),
            self.project.url,
            self.project.api_key,
            self.project.api_secret,
            None,
            Some(self.proof.thread_id),
        )?
        .ok_or_else(|| unavailable("the current project does not serve this task endpoint"))
    }
}

/// The frontend model's cancel, inside a tool call of the conversation `owner` serializes. The
/// owner resolves the task and shares any pending attempt; a retained task is controlled with
/// the current project's grant for the call's own thread, as a bound control would be.
pub(super) async fn cancel_for_model(
    env: &ProcessEnv,
    owner: Arc<dyn SessionCommands>,
    remote_id: &str,
    endpoint: &str,
    task_id: &str,
    project: &Project,
    thread_id: Option<String>,
) -> Result<lk_agents_abi::delegation::remote::RemoteCancelOutcome, AgentClientError> {
    let background = env.background().map_err(|error| unavailable(error.to_string()))?;
    let current = match thread_id {
        Some(thread_id) => CurrentScope::for_thread(project, endpoint, thread_id)?,
        None => None,
    };
    let (done, candidate) = Receipt::pair();
    let (target, admission) = owner
        .cancel_task(
            remote_id,
            endpoint,
            task_id,
            current.as_ref().map(|current| &current.proof),
            candidate.clone(),
        )
        .await?;
    // Admitted work outlives this tool call: other callers may share this published attempt.
    let (control_env, control_owner, control_target) = (env.clone(), owner.clone(), target.clone());
    background.spawn("model remote task cancellation", async move {
        let result = match admission {
            RemoteCancellationAdmission::Confirmed => Ok(()),
            RemoteCancellationAdmission::Pending(receipt) => tokio::select! {
                biased;
                result = receipt.wait() => result,
                () = control_owner.ended() => Err(unavailable(
                    "the conversation owner ended before cancellation was confirmed",
                )),
            },
            RemoteCancellationAdmission::Retained { recovery } => match current {
                Some(current) => {
                    perform(
                        &control_env,
                        std::future::pending(),
                        &control_target,
                        current,
                        recovery,
                        control_owner,
                        None,
                    )
                    .await
                }
                None => Err(unavailable("the current project cannot address this retained task")),
            },
        };
        done.resolve(result);
    });
    let settled = candidate.wait().await;
    // The outcome is judged from checkpointed facts rather than the settled error, which only
    // matters when the owner can no longer answer.
    match owner.cancel_outcome(&target).await {
        Ok(outcome) => Ok(outcome),
        Err(error) => settled.and(Err(storage_error(error))),
    }
}

async fn perform(
    env: &ProcessEnv,
    ended: impl std::future::Future<Output = ()>,
    target: &RemoteControlRef,
    current: CurrentScope,
    recovery: Option<RemoteRecovery>,
    owner: Arc<dyn SessionCommands>,
    http: Option<Arc<dyn lk_agents_net::http::port::HttpExchangePort>>,
) -> Result<(), AgentClientError> {
    let background = env.background().map_err(|error| unavailable(error.to_string()))?;
    let operation =
        policy::operation(&target.task, recovery, lk_agents_net::strings::shortuuid("request_"))
            .map_err(AgentClientError::from)?;
    let token = current.token(&target.endpoint)?;
    let client = Arc::new(
        match http {
            None => AgentClient::new(env, target.endpoint.clone(), Some(token), Default::default()),
            Some(http) => AgentClient::with_transport(
                target.endpoint.clone(),
                Some(token),
                Default::default(),
                http,
                (env.spawner)().map_err(|error| unavailable(error.to_string()))?,
                env.clock.clone(),
            ),
        }
        .map_err(AgentClientError::from)?,
    );
    let close_clock = background.clock.clone();
    let owned = Owned::new(
        &background,
        client.clone(),
        Box::new(move |client, deadline| {
            Box::pin(async move { Closed::bounded(&*close_clock, deadline, client.aclose()).await })
        }),
    );
    let cx = telemetry::Cx::NONE;
    let writing = transport::execute(&client, operation, &cx, |step| {
        let request_id = match step {
            Control::Dispatch { request_id, .. } => request_id.clone(),
            Control::Observe { .. } | Control::Reconcile { .. } => {
                lk_agents_net::strings::shortuuid("request_")
            }
        };
        Arc::new(journal::Journal::new(
            owner.clone(),
            target.endpoint.clone(),
            target.remote_id.clone(),
            request_id,
            None,
            None,
        )) as Arc<dyn RequestObserver>
    });
    tokio::pin!(writing, ended);
    let outcome = tokio::select! {
        biased;
        outcome = &mut writing => outcome,
        () = owner.ended() => { client.aclose().await; writing.await },
        () = &mut ended => { client.aclose().await; writing.await },
    };
    // Every peer observation has already been checkpointed. There is no live business-call
    // observer on this path; the explicit control receives the core's confirmation outcome.
    let confirmation = outcome.confirmation.map_err(AgentClientError::from);
    let closed = owned.close().await;
    confirmation?;
    match closed {
        Closed::Clean => Ok(()),
        Closed::Failed | Closed::TimedOut => {
            Err(unavailable("the task control client did not close cleanly"))
        }
    }
}

enum ScopeNext {
    Command(Option<Box<SessionCommand>>),
    Closing,
    Close,
}

/// The existing native command queue drives this short owner's journal writes. Its pinned
/// future retains the scope through every checkpoint, and closes it before its receipt ends.
async fn scope_owner(
    mut scope: Box<dyn SessionControlScope>,
    mut commands: tokio::sync::mpsc::Receiver<SessionCommand>,
    finish: signal::Cell<Option<()>>,
    bound: Arc<dyn BoundSessionControl>,
) -> Result<(), StorageError> {
    let mut closing = false;
    let mut result = loop {
        let next = tokio::select! {
            biased;
            command = commands.recv() => ScopeNext::Command(command.map(Box::new)),
            () = finish.until(Option::is_some) => ScopeNext::Close,
            () = bound.ended(), if !closing => ScopeNext::Closing,
        };
        match next {
            ScopeNext::Command(Some(command)) => {
                if let Err(error) = scope_command(&mut *scope, *command, closing).await {
                    break Err(error);
                }
            }
            ScopeNext::Command(None) | ScopeNext::Close => break Ok(()),
            ScopeNext::Closing => closing = true,
        }
    };
    commands.close();
    while let Some(command) = commands.recv().await {
        match &result {
            Ok(()) => {
                if let Err(error) = scope_command(&mut *scope, command, true).await {
                    result = Err(error);
                }
            }
            Err(error) => command.fail(error.clone()),
        }
    }
    let closed = scope.close().await;
    result.and(closed)
}

async fn scope_command(
    scope: &mut dyn SessionControlScope,
    command: SessionCommand,
    closing: bool,
) -> Result<(), StorageError> {
    let command = match command {
        SessionCommand::Recover { remote_id, endpoint, target, done } => {
            // This read follows the observation's checkpoint and uses the same short owner;
            // it cannot acquire a lease, dispatch, or restore a separate control capability.
            let result =
                lk_agents_core::session::remote::state(scope.snapshot(), &remote_id, &endpoint)
                    .and_then(|remote| {
                        remote
                            .map(|remote| lk_agents_core::session::remote::recover(remote, &target))
                            .transpose()
                    })
                    .map(Option::flatten);
            done.resolve(result);
            return Ok(());
        }
        command => command,
    };
    let SessionCommand::Apply { command, done } = command else {
        command.fail(StorageError::Invalid(
            "this short owner accepts only task-control journal I/O".into(),
        ));
        return Ok(());
    };
    let mut next = scope.snapshot().clone();
    let outcome = lk_agents_core::session::remote::admit(&command.input, closing, false)
        .and_then(|()| lk_agents_core::session::remote::apply(&mut next, *command));
    let saved = scope.checkpoint(next).await;
    // Resolve the exact journal producer even when its checkpoint failed. A failed writer
    // then closes admission; its transport loop observes owner death and joins its client.
    let result = saved.clone();
    done.resolve(saved.and(outcome));
    result
}

fn storage_error(error: StorageError) -> AgentClientError {
    match error {
        StorageError::Invalid(message) => unavailable(message),
        error => AgentClientError::Protocol(error.to_string()),
    }
}

fn unavailable(message: impl Into<String>) -> AgentClientError {
    AgentClientError::InvalidInput(message.into())
}
