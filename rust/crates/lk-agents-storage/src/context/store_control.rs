//! A bound task control borrows one short fenced owner and writes only framework state.

use super::*;
use crate::Unapplied;
use lk_agents_abi::session::SessionCommands;
use lk_agents_abi::session::control::{
    BoundSessionControl, ControlOwner, RemoteCancellation, RemoteControlRef, SessionControlScope,
};
use lk_agents_abi::storage::session::PersistedSessionState as _;
use lk_agents_abi::text_session::SessionSnapshot;
use lk_agents_abi::{BoxFuture, Closed};
use lk_agents_net::owned::Owned;
use lk_agents_net::signal;

/// One explicitly associated view and its existing store; no endpoint or credential registry.
struct ControlBinding {
    store: ContextStore,
    view: crate::session_state::SessionState,
}

impl std::ops::Deref for ControlBinding {
    type Target = ContextStore;
    fn deref(&self) -> &ContextStore {
        &self.store
    }
}

impl ContextStore {
    pub(crate) fn control_binding(
        &self,
        view: crate::session_state::SessionState,
    ) -> Arc<dyn BoundSessionControl> {
        Arc::new(ControlBinding { store: self.clone(), view })
    }
}

struct ControlScope {
    owner: Owned<ControlResource>,
    released: signal::Cell<Option<Result<(), StorageError>>>,
    snapshot: SessionSnapshot,
    value: Value,
    expected: Option<Value>,
    thread_id: String,
}

struct ControlResource {
    store: ContextStore,
    view: crate::session_state::SessionState,
    thread: Thread,
    run: Option<crate::Run>,
    _operation: tokio::sync::OwnedMutexGuard<()>,
    done: signal::Producer<Option<Result<(), StorageError>>>,
    completed: bool,
}

impl BoundSessionControl for ControlBinding {
    fn thread_id(&self) -> Result<Option<String>, StorageError> {
        self.ensure_control_process()?;
        self.view.ensure_control_process()?;
        self.activation.lock().ensure_open()?;
        Ok(self.database.as_ref().map(|database| database.thread_id.clone()))
    }

    fn ended(&self) -> BoxFuture<'_, ()> {
        if self.ensure_control_process().is_err() || self.view.ensure_control_process().is_err() {
            return Box::pin(async {});
        }
        let closed = self.control_closed.cell();
        Box::pin(async move { closed.until(|closed| *closed).await })
    }

    fn commands(&self) -> Result<Option<Arc<dyn SessionCommands>>, StorageError> {
        self.ensure_control_process()?;
        self.view.ensure_control_process()?;
        self.activation.lock().ensure_open()?;
        self.view.commands()?.map_or_else(|| self.store.session_commands(), |owner| Ok(Some(owner)))
    }

    fn claim_cancellation(
        &self,
        target: &RemoteControlRef,
        candidate: Arc<dyn RemoteCancellation>,
    ) -> Result<Arc<dyn RemoteCancellation>, StorageError> {
        self.ensure_control_process()?;
        self.view.ensure_control_process()?;
        self.activation.lock().ensure_open()?;
        let mut inspected = Vec::new();
        let mut attempts = self.control_attempts.lock();
        attempts.retain(|_, receipt| {
            receipt.upgrade().is_some_and(|receipt| {
                let pending = receipt.pending();
                inspected.push(receipt);
                pending
            })
        });
        if let Some(receipt) = attempts.get(target).and_then(std::sync::Weak::upgrade) {
            return Ok(receipt);
        }
        attempts.insert(target.clone(), Arc::downgrade(&candidate));
        drop(attempts);
        drop(inspected);
        Ok(candidate)
    }

    fn acquire(&self) -> BoxFuture<'_, Result<ControlOwner, StorageError>> {
        Box::pin(async move {
            if let Some(owner) = self.commands()? {
                return Ok(ControlOwner::Active(owner));
            }
            let operation = self.lock_managed().await;
            if let Some(owner) = self.commands()? {
                return Ok(ControlOwner::Active(owner));
            }
            self.ensure_control_process()?;
            self.view.ensure_control_process()?;
            self.activation.lock().ensure_open()?;
            let managed = self.managed.load(Ordering::Acquire);
            let attached = self.session_sink.lock().is_some();
            let owns_run = self.owned_session.lock().await.is_some();
            if managed || attached || owns_run {
                return Err(StorageError::Invalid(
                    "the bound conversation requires its current execution owner".into(),
                ));
            }
            let database = self.database.as_ref().ok_or_else(|| {
                StorageError::Invalid("idle task control requires a bound thread database".into())
            })?;
            let expected = self.session_state();
            let background = database.storage.background();
            let thread = database.open().await?;
            let (done, released) = signal::once();
            // Retain the thread before acquiring its partition. A canceled acquire still
            // closes the connection, and any acquired run is installed before another await.
            let mut owner = Owned::new(
                background,
                ControlResource {
                    store: self.store.clone(),
                    view: self.view.clone(),
                    thread,
                    run: None,
                    _operation: operation,
                    done,
                    completed: false,
                },
                Box::new(|mut scope, _deadline| {
                    Box::pin(async move {
                        // Run release has its own storage bound. Never time out a control write
                        // or cut lease release with a second outer timeout.
                        let result = match &mut scope.run {
                            Some(run) => {
                                run.abandon(
                                    if scope.completed {
                                        RunOutcome::Completed
                                    } else {
                                        RunOutcome::Canceled
                                    },
                                    "remote task control owner ended",
                                )
                                .await
                            }
                            None => Ok(()),
                        };
                        scope.thread.close().await;
                        let closed = if result.is_ok() { Closed::Clean } else { Closed::Failed };
                        scope.done.resolve(result);
                        closed
                    })
                }),
            );
            let loaded = async {
                let mut options = RunOptions::new(shortuuid("control_"), "");
                options.channel = Some("control".into());
                options.kind = crate::PartitionKind::Context;
                let run = owner
                    .thread
                    .partition(&database.partition_key)
                    .open(options, &database.holder)
                    .await?;
                owner.run = Some(run);
                let value = owner
                    .run
                    .as_ref()
                    .expect("the acquired run was retained")
                    .state()
                    .cloned()
                    .ok_or_else(|| {
                        StorageError::Invalid(
                            "this bound partition has no saved conversation".into(),
                        )
                    })?;
                let checked = crate::session_state::SessionState::from_value(value.clone())?;
                let snapshot = checked._snapshot_data()?.ok_or_else(|| {
                    StorageError::Invalid("this bound partition has no saved conversation".into())
                })?;
                Ok::<_, StorageError>((value, checked, snapshot))
            }
            .await;
            let (value, checked, snapshot) = match loaded {
                Ok(loaded) => loaded,
                Err(error) => {
                    release_control(owner, released, false).await?;
                    return Err(error);
                }
            };
            self.view.publish_control(&self.store, &checked);
            Ok(ControlOwner::Fenced(Box::new(ControlScope {
                owner,
                released,
                snapshot,
                value,
                expected,
                thread_id: database.thread_id.clone(),
            })))
        })
    }
}

impl SessionControlScope for ControlScope {
    fn thread_id(&self) -> &str {
        &self.thread_id
    }
    fn snapshot(&self) -> &SessionSnapshot {
        &self.snapshot
    }

    fn checkpoint(&mut self, state: SessionSnapshot) -> BoxFuture<'_, Result<(), StorageError>> {
        Box::pin(async move {
            let mut value = crate::session_state::snapshot_value(state.clone())?;
            if let Some(pending) = self.value.get("pending_user_turns") {
                value["pending_user_turns"] = pending.clone();
            }
            let run = self.owner.run.as_mut().expect("the control scope retains its run");
            // Preserve the partition's delivered rows. Task control edits neither app data
            // nor voice/text dialogue, and must not synthesize a replacement history.
            let history = run.history().iter().map(|row| row.payload.clone()).collect();
            run.checkpoint_session(value.clone(), history).await?;
            self.owner.store.checkpointed_session(self.expected.clone(), value.clone())?;
            self.owner.view.publish_control(
                &self.owner.store,
                &crate::session_state::SessionState::from_value(value.clone())?,
            );
            self.expected = Some(value.clone());
            self.value = value;
            self.snapshot = state;
            Ok(())
        })
    }

    fn close(self: Box<Self>) -> BoxFuture<'static, Result<(), StorageError>> {
        let Self { owner, released, .. } = *self;
        Box::pin(release_control(owner, released, true))
    }
}

async fn release_control(
    mut owner: Owned<ControlResource>,
    released: signal::Cell<Option<Result<(), StorageError>>>,
    completed: bool,
) -> Result<(), StorageError> {
    owner.completed = completed;
    let closed = owner.close().await;
    released.wait().await.map_err(|_| {
        StorageError::Unavailable(Unapplied::local("the task control release owner ended"))
    })??;
    match closed {
        Closed::Clean => Ok(()),
        Closed::Failed | Closed::TimedOut => Err(StorageError::Unavailable(Unapplied::local(
            "the task control owner did not close cleanly",
        ))),
    }
}
