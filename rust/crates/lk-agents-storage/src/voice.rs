use lk_agents_net::owned::Background;
use lk_agents_net::signal::{self, Producer};
use lk_agents_net::task::TaskSet;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use parking_lot::Mutex;

use crate::thread::run::{ItemRow, RunOutcome};
use lk_agents_abi::llm::chat_context::protocol::{decode_chat_item, encode_chat_item};
use lk_agents_abi::storage::context::ContextActivation as _;
use lk_agents_abi::storage::thread::AbandonReason;
use lk_agents_abi::storage::voice::{
    StorageWork, VoiceStorage, VoiceStorageCommand, VoiceStorageEvent, VoiceStore as HostVoiceStore,
};
use lk_agents_abi::storage::{StorageError, Unacknowledged};
use lk_agents_abi::{ChatContent, ChatItem, Role};

use crate::backends::with_thread;
use crate::context::store::ContextStore;
use crate::telemetry::Cx;
use crate::{CommitOptions, Holder, Lease, Run, RunOptions, Storage, Thread, shortuuid};

pub struct VoiceStore {
    backend: Arc<dyn Storage>,
    thread_id: String,
    partition_key: String,
    /// Every turn's lease owner, so a turn whose acquire was canceled leaves the next one
    /// nothing to wait for.
    holder: Holder,
    activation: ContextStore,
    initial: Option<Vec<ChatItem>>,
    state: tokio::sync::Mutex<State>,
    lease: Mutex<Option<Lease>>,
    /// Bumped as each turn releases its lease, ending that lease's loss observation.
    generation: Producer<u64>,
    lost: Arc<AtomicBool>,
    /// Settles, aborts and loss observations, which `aclose` waits out.
    tasks: Arc<TaskSet>,
    /// The backend's executor and clock, which they run on.
    background: Background,
    closing: Mutex<bool>,
    /// The session the store serves: its turns' health metrics record under it.
    cx: Cx,
}

struct State {
    /// Each conversational item as this store last read it from the partition or committed it.
    known: BTreeMap<String, ChatItem>,
    version: i64,
    active: Option<Active>,
    /// The frozen staged fields of a commit left unresolved, and why its outcome is unknown.
    uncertain_fields: Option<(tokio::sync::OwnedMutexGuard<()>, Unacknowledged)>,
}

struct Active {
    thread: Thread,
    run: Run,
    speeches: BTreeSet<u64>,
    checkpoint_sequence: Option<u64>,
}

fn conversational(item: &ChatItem) -> bool {
    !item.as_message().is_some_and(|message| matches!(message.role, Role::System | Role::Developer))
}

impl VoiceStore {
    pub async fn bound(
        backend: Arc<dyn Storage>,
        thread_id: String,
        partition_key: String,
        explicit_history: Option<Vec<ChatItem>>,
        cx: Cx,
    ) -> Result<Self, StorageError> {
        Self::bound_with_activation(backend, thread_id, partition_key, explicit_history, cx, None)
            .await
    }

    pub(crate) async fn bound_with_activation(
        backend: Arc<dyn Storage>,
        thread_id: String,
        partition_key: String,
        explicit_history: Option<Vec<ChatItem>>,
        cx: Cx,
        activation: Option<ContextStore>,
    ) -> Result<Self, StorageError> {
        if let Some(activation) = &activation
            && (activation.closed() || !activation.matches_partition(&thread_id, &partition_key))
        {
            return Err(StorageError::Invalid(
                "voice context belongs to a different or closed partition".into(),
            ));
        }
        let (history, version) = with_thread(&*backend, &thread_id, async |thread| {
            let partition_key = thread.partition(&partition_key);
            let history = if explicit_history.is_none() {
                partition_key.history().await?
            } else {
                Vec::new()
            };
            let version =
                partition_key.info().await?.map_or(0, |partition_key| partition_key.version);
            Ok::<_, StorageError>((history, version))
        })
        .await?;
        let items = history
            .iter()
            .map(|row| decode_chat_item(row.payload.clone()).map_err(StorageError::from))
            .collect::<Result<Vec<_>, _>>()?;
        if explicit_history.is_none() {
            let answered: BTreeSet<_> = items
                .iter()
                .filter_map(|item| match item {
                    ChatItem::FunctionCallOutput(output) => Some(output.call_id.as_str()),
                    _ => None,
                })
                .collect();
            if items.iter().any(|item| {
                matches!(item,
                ChatItem::FunctionCall(call) if !answered.contains(call.call_id.as_str()))
            }) {
                return Err(StorageError::Invalid(
                    "a stored tool has no recorded result; reconcile its outcome before resuming the voice session".into(),
                ));
            }
        }
        let known = explicit_history
            .as_ref()
            .unwrap_or(&items)
            .iter()
            .map(|item| (item.id().to_owned(), item.clone()))
            .collect();
        let activation = match activation {
            Some(activation) => activation,
            None => {
                ContextStore::bound(
                    backend.clone(),
                    thread_id.clone(),
                    partition_key.clone(),
                    cx.clone(),
                )
                .await?
            }
        };
        Ok(Self {
            thread_id,
            partition_key,
            holder: Holder::new("voice"),
            activation,
            initial: explicit_history.is_none().then_some(items),
            state: tokio::sync::Mutex::new(State {
                known,
                version,
                active: None,
                uncertain_fields: None,
            }),
            lease: Mutex::new(None),
            generation: signal::signal(0).0,
            lost: Arc::new(AtomicBool::new(false)),
            tasks: Arc::default(),
            background: backend.background().clone(),
            closing: Mutex::new(false),
            cx,
            backend,
        })
    }

    pub fn session_sink(self: &Arc<Self>) -> Arc<dyn lk_agents_abi::a2a::turn::HandlerPort> {
        Arc::new(VoiceSessionSink(Arc::downgrade(self)))
    }

    fn released(&self) {
        *self.lease.lock() = None;
        self.generation.modify(|generation| *generation += 1);
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
            return Err(StorageError::LeaseLost("the voice partition_key lease was lost".into()));
        }
        if let Some(active) = &mut state.active {
            active.speeches.insert(speech);
            return Ok(local);
        }
        // Exclude direct saves before taking the partition, avoiding the inverse lock/lease order.
        let fields = self.activation.lock_managed().await;
        let thread = self.backend.open(&self.thread_id).await?.with_telemetry(self.cx.clone());
        let mut options = RunOptions::new(shortuuid("voice_"), "");
        options.channel = Some("voice".into());
        options.kind = crate::thread::run::PartitionKind::Voice;
        let mut run = match thread.partition(&self.partition_key).open(options, &self.holder).await
        {
            Ok(run) => run,
            Err(error) => {
                thread.close().await;
                return Err(error);
            }
        };
        let prepared = (|| {
            // The partition as this store last saw it: the turn's own reading, edits included, is
            // the newest.
            let (merged, written) = if run.version() == state.version {
                // The turn's new items are known from now on: if it is abandoned, a later reload
                // drops them rather than committing them as another turn's.
                for item in local.iter().filter(|item| conversational(item)) {
                    state.known.entry(item.id().to_owned()).or_insert_with(|| item.clone());
                }
                // `local` is the turn's view (the model's), not the partition's history: the
                // partition keeps its rows (history-only items, other agents' items), with the
                // turn's edits. Its new items are the session's to write, in the session's order.
                let stored = run
                    .history()
                    .iter()
                    .map(|row| decode_chat_item(row.payload.clone()).map_err(StorageError::from))
                    .collect::<Result<Vec<_>, _>>()?;
                let mine: BTreeMap<&str, &ChatItem> =
                    local.iter().map(|item| (item.id(), item)).collect();
                let written: Vec<ChatItem> = stored
                    .into_iter()
                    .map(|row| mine.get(row.id()).map_or(row, |item| (*item).clone()))
                    .collect();
                (local, written)
            } else {
                let stored = run
                    .history()
                    .iter()
                    .map(|row| decode_chat_item(row.payload.clone()).map_err(StorageError::from))
                    .collect::<Result<Vec<_>, _>>()?;
                let ids: BTreeSet<_> = stored.iter().map(|item| item.id().to_owned()).collect();
                let (mut instructions, pending): (Vec<_>, Vec<_>) =
                    local.into_iter().partition(|item| !conversational(item));
                let mut edited: BTreeMap<String, ChatItem> = BTreeMap::new();
                let mut added = Vec::new();
                for item in pending {
                    if ids.contains(item.id()) {
                        edited.insert(item.id().to_owned(), item);
                    } else if !state.known.contains_key(item.id()) {
                        added.push(item);
                    }
                }
                // A row the other writer left as this store last saw it keeps the turn's own
                // edit (a question answered since the last commit); one it changed is its own.
                let mut merged: Vec<ChatItem> = stored
                    .iter()
                    .map(|row| match edited.remove(row.id()) {
                        Some(mine) if state.known.get(row.id()) == Some(row) => mine,
                        _ => row.clone(),
                    })
                    .collect();
                state.known = stored.into_iter().map(|item| (item.id().to_owned(), item)).collect();
                state.known.extend(added.iter().map(|item| (item.id().to_owned(), item.clone())));
                merged.extend(added);
                instructions.extend(merged);
                let written = instructions.clone();
                (instructions, written)
            };
            let session = self.activation.session_state();
            self.activation.begin_managed(
                run.context_scopes().to_vec(),
                run.state().cloned(),
                &fields,
            )?;
            if let Some(session) = session {
                self.activation.stage_session(session)?;
            }
            Ok::<_, StorageError>((merged, written))
        })();
        let (merged, written) = match prepared {
            Ok(prepared) => prepared,
            Err(error) => {
                let _ = run.abandon(RunOutcome::Failed, AbandonReason::VoiceLoad.as_str()).await;
                thread.close().await;
                return Err(error);
            }
        };
        *self.lease.lock() = run.lease().cloned();
        state.active = Some(Active {
            thread,
            run,
            speeches: BTreeSet::from([speech]),
            checkpoint_sequence: None,
        });
        if let Some(routing) = self.activation.session_state() {
            state
                .active
                .as_mut()
                .expect("turn was installed")
                .run
                .checkpoint_session(routing, written.iter().map(encode_chat_item).collect())
                .await?;
        }
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
        let message_ids: BTreeMap<_, _> = active
            .run
            .history()
            .iter()
            .map(|row| (row.id.clone(), row.message_id.clone()))
            .collect();
        let history: Vec<ChatItem> = history.into_iter().filter(conversational).collect();
        let committed: BTreeMap<String, ChatItem> =
            history.iter().map(|item| (item.id().to_owned(), item.clone())).collect();
        let history = history
            .into_iter()
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
                *active.run.history_mut() = history;
                if let Some(routing) = self.activation.session_state() {
                    active.run.set_state(routing);
                }
                active.run.stage_scopes(self.activation.writes());
                active.run.stage_statements(self.activation.statements());
                active.run.commit(CommitOptions::default()).await.map(|committed| committed.version)
            }
            Err(error) => {
                let _ = active
                    .run
                    .abandon(RunOutcome::Failed, AbandonReason::VoiceEncode.as_str())
                    .await;
                Err(error)
            }
        };
        if let Err(StorageError::Uncertain(why)) = &result {
            // Retain the resolver and frozen staged fields. Closing observes its real outcome.
            state.active = Some(active);
            state.uncertain_fields = Some((fields, why.clone()));
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
            state.known = committed;
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
        if let Some((_, why)) = &state.uncertain_fields {
            return Err(StorageError::Uncertain(why.clone()));
        }
        let fields = self.activation.lock_managed().await;
        let mut active = state.active.take().expect("the active voice turn");
        let result =
            active.run.abandon(RunOutcome::Failed, AbandonReason::VoiceGeneration.as_str()).await;
        self.released();
        active.thread.close().await;
        self.activation.end_managed(false, fields);
        result
    }

    fn observe_loss(
        &self,
    ) -> impl std::future::Future<Output = Result<bool, StorageError>> + Send + 'static {
        let mut generation = self.generation.cell();
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

    fn settle_in_background(self: &Arc<Self>, speech: u64, history: Vec<ChatItem>, events: Events) {
        let store = self.clone();
        self.tasks.spawn(&*self.background.spawner, "voice storage settle", async move {
            let result = store.finish(speech, history).await;
            let lease_lost = store.lease_lost();
            let _ = events.push(VoiceStorageEvent::Settled { speech, result, lease_lost });
        });
    }

    fn abandon_in_background(self: &Arc<Self>, speech: u64, events: Events) {
        let store = self.clone();
        self.tasks.spawn(&*self.background.spawner, "voice storage abort", async move {
            let result = store.abandon(speech).await;
            let _ = events.push(VoiceStorageEvent::Abandoned { speech, result });
        });
    }

    fn report_lease_loss(&self, speech: u64, events: Events) {
        let observation = self.observe_loss();
        self.tasks.spawn(&*self.background.spawner, "voice lease loss", async move {
            if observation.await.unwrap_or(false) {
                let _ = events.push(VoiceStorageEvent::LeaseLost { speech });
            }
        });
    }
}

type Events = Arc<dyn lk_agents_abi::StreamInput<VoiceStorageEvent>>;

impl VoiceStorage for VoiceStore {
    fn dispatch(
        self: Arc<Self>,
        command: VoiceStorageCommand,
        events: Arc<dyn lk_agents_abi::StreamInput<VoiceStorageEvent>>,
    ) -> Result<(), StorageError> {
        let closing = self.closing.lock();
        if *closing {
            return Err(StorageError::Invalid("the voice store has closed".into()));
        }
        // Each task is spawned while `closing` is held, so none starts after close.
        match command {
            VoiceStorageCommand::Settle { speech, history } => {
                self.settle_in_background(speech, history, events)
            }
            VoiceStorageCommand::Abort { speech } => self.abandon_in_background(speech, events),
            VoiceStorageCommand::ObserveLoss { speech } => self.report_lease_loss(speech, events),
        }
        drop(closing);
        Ok(())
    }

    fn acquire(&self, speech: u64, local: Vec<ChatItem>) -> StorageWork<'_, Vec<ChatItem>> {
        Box::pin(self.begin(speech, local))
    }
    fn checkpoint(
        &self,
        sequence: u64,
        agent: String,
        history: Vec<ChatItem>,
    ) -> StorageWork<'_, ()> {
        Box::pin(async move {
            let mut state = self.state.lock().await;
            let active = state
                .active
                .as_mut()
                .ok_or_else(|| StorageError::Invalid("no voice turn owns the partition".into()))?;
            if active.checkpoint_sequence.is_some_and(|accepted| sequence <= accepted) {
                return Ok(());
            }
            let routing = crate::session_state::voice_value(
                agent,
                history
                    .iter()
                    .filter(|item| conversational(item))
                    .map(|item| item.id().to_owned())
                    .collect(),
            )?;
            active
                .run
                .checkpoint_session(routing.clone(), history.iter().map(encode_chat_item).collect())
                .await?;
            active.checkpoint_sequence = Some(sequence);
            self.activation.stage_session(routing)?;
            self.activation.acknowledge_session_state();
            Ok(())
        })
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
            let uncertain = state.uncertain_fields.take().map(|(fields, _)| fields);
            let resolving = uncertain.is_some();
            let fields = match uncertain {
                Some(fields) => fields,
                None => self.activation.lock_managed().await,
            };
            result = if resolving && !active.run.closed() {
                match active.run.resolve_later().await {
                    Some(result) => result.map(|_| ()),
                    None => Err(StorageError::Invalid("the voice commit has no resolver".into())),
                }
            } else {
                active.run.abandon(RunOutcome::Canceled, AbandonReason::VoiceClose.as_str()).await
            };
            self.released();
            active.thread.close().await;
            self.activation.end_managed(result.is_ok() && resolving, fields);
        } else {
            self.released();
        }
        self.activation.close();
        drop(state);
        let clock = &*self.background.clock;
        self.tasks.wait_empty(clock, None).await;
        let (voice, context) =
            tokio::join!(self.holder.wait_released(clock), self.activation.wait_released());
        if !(voice && context) {
            log::warn!(target: lk_agents_abi::LOG_TARGET, partition_key = self.partition_key.as_str(); "the voice store closed with a lease release in flight");
        }
        result
    }
}

#[cfg(test)]
#[path = "voice_tests.rs"]
mod tests;

/// The activation can outlive a turn without keeping the call's storage alive.
struct VoiceSessionSink(std::sync::Weak<VoiceStore>);

impl lk_agents_abi::a2a::turn::HandlerPort for VoiceSessionSink {
    fn checkpoint(&self) -> lk_agents_abi::BoxFuture<'_, bool> {
        Box::pin(async { false })
    }
    fn progress(&self, _text: String) -> lk_agents_abi::BoxFuture<'_, ()> {
        Box::pin(async {})
    }
    fn token(&self, _delta: String) -> lk_agents_abi::BoxFuture<'_, ()> {
        Box::pin(async {})
    }
    fn save_session(
        &self,
        mut routing: serde_json::Value,
        history: Vec<ChatItem>,
    ) -> lk_agents_abi::BoxFuture<'_, Result<(), StorageError>> {
        Box::pin(async move {
            let store = self
                .0
                .upgrade()
                .ok_or_else(|| StorageError::Invalid("the voice session has ended".into()))?;
            let mut state = store.state.lock().await;
            let active = state
                .active
                .as_mut()
                .ok_or_else(|| StorageError::Invalid("no voice turn owns the partition".into()))?;
            // A host ledger may not yet contain a delivered partial item. A durability barrier
            // must retain that accepted prefix; final turn settlement owns deletions/truncation.
            let mut merged = active
                .run
                .history()
                .iter()
                .map(|row| decode_chat_item(row.payload.clone()).map_err(StorageError::from))
                .collect::<Result<Vec<_>, _>>()?;
            for item in history {
                if let Some(existing) =
                    merged.iter_mut().find(|existing| existing.id() == item.id())
                {
                    *existing = item;
                } else {
                    merged.push(item);
                }
            }
            routing
                .as_object_mut()
                .ok_or_else(|| StorageError::Invalid("voice routing must be an object".into()))?
                .insert(
                    "history_ids".into(),
                    serde_json::json!(
                        merged
                            .iter()
                            .filter(|item| conversational(item))
                            .map(|item| item.id())
                            .collect::<Vec<_>>()
                    ),
                );
            active
                .run
                .checkpoint_session(routing.clone(), merged.iter().map(encode_chat_item).collect())
                .await?;
            store.activation.stage_session(routing)?;
            Ok(())
        })
    }
}
