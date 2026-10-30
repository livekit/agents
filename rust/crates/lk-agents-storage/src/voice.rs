//! One call's native lane lease, transport reload and fenced history/context commit.

use lk_agents_net::clock::RealClock;
use lk_agents_net::task::{TaskSet, TokioSpawner};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use parking_lot::Mutex;

use crate::run::{ItemRow, RunOutcome};
use lk_agents_abi::llm::chat_context::protocol::{decode_chat_item, encode_chat_item};
use lk_agents_abi::storage::StorageError;
use lk_agents_abi::storage::context::ContextActivation as _;
use lk_agents_abi::storage::voice::{
    StorageWork, VoiceStorage, VoiceStorageCommand, VoiceStorageEvent, VoiceStore as HostVoiceStore,
};
use lk_agents_abi::{ChatContent, ChatItem, Role};

use crate::context::store::ContextStore;
use crate::{CommitOptions, Lease, Run, RunOptions, Storage, Thread, shortuuid};

pub struct VoiceStore {
    backend: Arc<dyn Storage>,
    thread_id: String,
    lane: String,
    activation: ContextStore,
    initial: Option<Vec<ChatItem>>,
    state: tokio::sync::Mutex<State>,
    lease: Mutex<Option<Lease>>,
    generation: tokio::sync::watch::Sender<u64>,
    lost: Arc<AtomicBool>,
    /// Settles, aborts and loss observations, which `aclose` waits out.
    tasks: Arc<TaskSet>,
    closing: Mutex<bool>,
}

struct State {
    known: BTreeSet<String>,
    version: i64,
    explicit: bool,
    active: Option<Active>,
    uncertain_fields: Option<tokio::sync::OwnedMutexGuard<()>>,
}

struct Active {
    thread: Thread,
    run: Run,
    speeches: BTreeSet<u64>,
}

fn conversational(item: &ChatItem) -> bool {
    !item.as_message().is_some_and(|message| matches!(message.role, Role::System | Role::Developer))
}

impl VoiceStore {
    pub async fn bound(
        backend: Arc<dyn Storage>,
        thread_id: String,
        lane: String,
        explicit_history: Option<Vec<ChatItem>>,
    ) -> Result<Self, StorageError> {
        let thread = backend.open(&thread_id, "voice-load").await?;
        let snapshot = async {
            let lane = thread.lane(&lane);
            let history =
                if explicit_history.is_none() { lane.history().await? } else { Vec::new() };
            let version = lane.info().await?.map_or(0, |lane| lane.version);
            Ok::<_, StorageError>((history, version))
        }
        .await;
        thread.close().await;
        let (history, version) = snapshot?;
        let items = history
            .iter()
            .map(|row| decode_chat_item(row.payload.clone()).map_err(StorageError::Unavailable))
            .collect::<Result<Vec<_>, _>>()?;
        let known = explicit_history
            .as_ref()
            .unwrap_or(&items)
            .iter()
            .map(|item| item.id().to_owned())
            .collect();
        let explicit = explicit_history.is_some();
        let activation =
            ContextStore::bound(backend.clone(), thread_id.clone(), lane.clone()).await?;
        Ok(Self {
            backend,
            thread_id,
            lane,
            activation,
            initial: (!explicit).then_some(items),
            state: tokio::sync::Mutex::new(State {
                known,
                version,
                explicit,
                active: None,
                uncertain_fields: None,
            }),
            lease: Mutex::new(None),
            generation: tokio::sync::watch::channel(0).0,
            lost: Arc::new(AtomicBool::new(false)),
            tasks: Arc::default(),
            closing: Mutex::new(false),
        })
    }

    fn released(&self) {
        *self.lease.lock() = None;
        self.generation.send_modify(|generation| *generation += 1);
    }

    async fn begin(
        &self,
        speech: u64,
        local: Vec<ChatItem>,
    ) -> Result<Vec<ChatItem>, StorageError> {
        let mut state = self.state.lock().await;
        if *self.closing.lock() {
            return Err(StorageError::Invalid("the voice store has closed".into()));
        }
        if self.lease_lost() {
            return Err(StorageError::LeaseLost("the voice lane lease was lost".into()));
        }
        if let Some(active) = &mut state.active {
            active.speeches.insert(speech);
            return Ok(local);
        }
        // Exclude direct saves before taking the lane, avoiding the inverse lock/lease order.
        let fields = self.activation.lock_managed().await;
        let thread = self.backend.open(&self.thread_id, "voice").await?;
        let mut options = RunOptions::new(shortuuid("voice_"), "");
        options.channel = Some("voice".into());
        let mut run = match thread.lane(&self.lane).open(options).await {
            Ok(run) => run,
            Err(error) => {
                thread.close().await;
                return Err(error);
            }
        };
        let prepared = (|| {
            let merged =
                if state.explicit && run.version() == state.version {
                    local
                } else {
                    let mut stored = run
                        .history
                        .iter()
                        .map(|row| {
                            decode_chat_item(row.payload.clone()).map_err(StorageError::Unavailable)
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    let ids: BTreeSet<_> = stored.iter().map(|item| item.id().to_owned()).collect();
                    let (mut instructions, pending): (Vec<_>, Vec<_>) =
                        local.into_iter().partition(|item| !conversational(item));
                    stored.extend(pending.into_iter().filter(|item| {
                        !state.known.contains(item.id()) && !ids.contains(item.id())
                    }));
                    instructions.extend(stored);
                    instructions
                };
            self.activation.begin_managed(
                run.context_scopes.clone(),
                run.state.clone(),
                &fields,
            )?;
            Ok::<_, StorageError>(merged)
        })();
        let merged = match prepared {
            Ok(merged) => merged,
            Err(error) => {
                let _ = run.abandon(RunOutcome::Failed, "voice-load").await;
                thread.close().await;
                return Err(error);
            }
        };
        state.known = merged
            .iter()
            .filter(|item| conversational(item))
            .map(|item| item.id().to_owned())
            .collect();
        state.explicit = false;
        *self.lease.lock() = run.lease().cloned();
        state.active = Some(Active { thread, run, speeches: BTreeSet::from([speech]) });
        Ok(merged)
    }

    async fn finish(&self, speech: u64, history: Vec<ChatItem>) -> Result<(), StorageError> {
        let mut state = self.state.lock().await;
        let Some(active) = state.active.as_mut() else { return Ok(()) };
        active.speeches.remove(&speech);
        if !active.speeches.is_empty() {
            return Ok(());
        }
        let fields = self.activation.lock_managed().await;
        let mut active = state.active.take().expect("the active voice turn");
        let message_ids: BTreeMap<_, _> =
            active.run.history.iter().map(|row| (row.id.clone(), row.message_id.clone())).collect();
        let history = history
            .into_iter()
            .filter(conversational)
            .map(|item| {
                if item.as_message().is_some_and(|message| {
                    message.content.iter().any(|content| !matches!(content, ChatContent::Text(_)))
                }) {
                    return Err(StorageError::NotSerializable(
                        item.id().into(),
                        "thread chat content must be durable text".into(),
                    ));
                }
                Ok(ItemRow {
                    id: item.id().to_owned(),
                    message_id: message_ids.get(item.id()).cloned().flatten(),
                    payload: encode_chat_item(&item),
                })
            })
            .collect::<Result<Vec<_>, StorageError>>();
        let result = match history {
            Ok(history) => {
                active.run.history = history;
                active.run.scope_writes = self.activation.writes();
                active.run.app_writes = self.activation.statements();
                active.run.commit(CommitOptions::default()).await.map(|committed| committed.version)
            }
            Err(error) => {
                let _ = active.run.abandon(RunOutcome::Failed, "voice-encode").await;
                Err(error)
            }
        };
        if matches!(result, Err(StorageError::Uncertain(_))) {
            // Retain the resolver and frozen staged fields. Closing observes its real outcome.
            state.active = Some(active);
            state.uncertain_fields = Some(fields);
            self.lost.store(true, Ordering::Release);
            return result.map(|_| ());
        }
        if result.is_err() {
            if matches!(result, Err(StorageError::VersionConflict { .. })) || self.lease_lost() {
                self.lost.store(true, Ordering::Release);
            }
            if let Some(lease) = active.run.lease() {
                let _ = lease.release(None).await;
            }
        }
        self.released();
        active.thread.close().await;
        self.activation.end_managed(result.is_ok(), fields);
        if let Ok(version) = result {
            state.version = version;
            state.known = active.run.history.iter().map(|row| row.id.clone()).collect();
            Ok(())
        } else {
            result.map(|_| ())
        }
    }

    async fn abandon(&self, speech: u64) -> Result<(), StorageError> {
        let mut state = self.state.lock().await;
        let Some(active) = state.active.as_mut() else { return Ok(()) };
        active.speeches.remove(&speech);
        if !active.speeches.is_empty() {
            return Ok(());
        }
        if state.uncertain_fields.is_some() {
            return Err(StorageError::Uncertain("the voice commit is still unresolved".into()));
        }
        let fields = self.activation.lock_managed().await;
        let mut active = state.active.take().expect("the active voice turn");
        let result = active.run.abandon(RunOutcome::Failed, "voice-generation").await;
        self.released();
        active.thread.close().await;
        self.activation.end_managed(false, fields);
        result
    }

    fn observe_loss(
        &self,
    ) -> impl std::future::Future<Output = Result<bool, StorageError>> + Send + 'static {
        let mut generation = self.generation.subscribe();
        let lease = self.lease.lock().clone();
        let lost = self.lost.clone();
        async move {
            let Some(lease) = lease else { return Ok(false) };
            tokio::select! {
                _ = lease.lost() => {
                    lost.store(true, Ordering::Release);
                    Ok(true)
                }
                _ = generation.changed() => Ok(false),
            }
        }
    }
}

impl VoiceStorage for VoiceStore {
    fn dispatch(
        self: Arc<Self>,
        command: VoiceStorageCommand,
        events: Arc<dyn lk_agents_abi::StreamSink<VoiceStorageEvent>>,
    ) -> Result<(), StorageError> {
        let closing = self.closing.lock();
        if *closing {
            return Err(StorageError::Invalid("the voice store has closed".into()));
        }
        match command {
            VoiceStorageCommand::Settle { speech, history } => {
                let store = self.clone();
                self.tasks.spawn(&TokioSpawner::current(), "voice storage settle", async move {
                    let result = store.finish(speech, history).await;
                    let _ = events.push(VoiceStorageEvent::Settled {
                        speech,
                        result,
                        lease_lost: store.lease_lost(),
                    });
                });
            }
            VoiceStorageCommand::Abort { speech } => {
                let store = self.clone();
                self.tasks.spawn(&TokioSpawner::current(), "voice storage abort", async move {
                    let result = store.abandon(speech).await;
                    let _ = events.push(VoiceStorageEvent::Abandoned { speech, result });
                });
            }
            VoiceStorageCommand::ObserveLoss { speech } => {
                let observation = self.observe_loss();
                self.tasks.spawn(&TokioSpawner::current(), "voice lease loss", async move {
                    if observation.await.unwrap_or(false) {
                        let _ = events.push(VoiceStorageEvent::LeaseLost { speech });
                    }
                });
            }
        }
        Ok(())
    }

    fn acquire(&self, speech: u64, local: Vec<ChatItem>) -> StorageWork<'_, Vec<ChatItem>> {
        Box::pin(self.begin(speech, local))
    }
    fn lease_lost(&self) -> bool {
        if self.lease.lock().as_ref().is_some_and(Lease::is_lost) {
            self.lost.store(true, Ordering::Release);
        }
        self.lost.load(Ordering::Acquire)
    }
}

impl HostVoiceStore for VoiceStore {
    type ContextActivation = ContextStore;
    fn initial_history(&self) -> Option<Vec<ChatItem>> {
        self.initial.clone()
    }
    fn activation(&self) -> ContextStore {
        self.activation.clone()
    }

    async fn aclose(&self) -> Result<(), StorageError> {
        *self.closing.lock() = true;
        let mut state = self.state.lock().await;
        let mut result = Ok(());
        if let Some(mut active) = state.active.take() {
            let uncertain = state.uncertain_fields.take();
            let resolving = uncertain.is_some();
            let fields = match uncertain {
                Some(fields) => fields,
                None => self.activation.lock_managed().await,
            };
            result = if resolving && !active.run.closed {
                match active.run.resolve_later().await {
                    Some(result) => result.map(|_| ()),
                    None => Err(StorageError::Invalid("the voice commit has no resolver".into())),
                }
            } else {
                active.run.abandon(RunOutcome::Canceled, "voice-close").await
            };
            self.released();
            active.thread.close().await;
            self.activation.end_managed(result.is_ok() && resolving, fields);
        } else {
            self.released();
        }
        self.activation.close();
        drop(state);
        self.tasks.wait_empty(&RealClock::new(), None).await;
        result
    }
}
