use lk_agents_net::owned::Background;
use lk_agents_net::signal::{self, Producer};
use lk_agents_net::task::TaskSet;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use parking_lot::Mutex;

use crate::thread::run::{ItemRow, RunOutcome};
use livekit_protocol::agent::chat_context::ChatItem as StoredItem;
use lk_agents_abi::llm::chat_context::protocol::{decode_chat_item, encode_chat_item};
use lk_agents_abi::storage::StorageError;
use lk_agents_abi::storage::context::ContextActivation as _;
use lk_agents_abi::storage::thread::AbandonReason;
use lk_agents_abi::storage::voice::{
    StorageWork, VoiceStorage, VoiceStorageCommand, VoiceStorageEvent, VoiceStore as HostVoiceStore,
};
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
    /// Each conversational item as this store last read it from the partition or committed it,
    /// in its stored form: what the partition keeps of an item is all a reading can compare.
    known: BTreeMap<String, StoredItem>,
    active: Option<Active>,
}

struct Active {
    thread: Thread,
    run: Run,
    speeches: BTreeSet<u64>,
    history_sequence: Option<u64>,
    input_sequence: Option<u64>,
    /// Recognition progress is durable but not yet accepted model-visible conversation.
    pending_user_turns: Vec<lk_agents_abi::ChatMessage>,
}

fn conversational(item: &ChatItem) -> bool {
    !item.as_message().is_some_and(|message| matches!(message.role, Role::System | Role::Developer))
}

impl State {
    /// Reconcile against rows actually read or written, never treating a local omission as
    /// deletion of an unseen item. Only the database baseline can retire a previously known row.
    fn reconcile(
        &mut self,
        stored: Vec<ChatItem>,
        local: Vec<ChatItem>,
    ) -> (Vec<ChatItem>, Vec<ChatItem>) {
        let rows: BTreeMap<_, _> = stored.iter().map(|item| (item.id(), item)).collect();
        let (mut instructions, pending): (Vec<_>, Vec<_>) =
            local.into_iter().partition(|item| !conversational(item));
        let mut merged = Vec::new();
        for item in pending {
            match rows.get(item.id()) {
                Some(row) if self.known.get(item.id()) != Some(&encode_chat_item(row)) => {
                    merged.push((*row).clone());
                }
                Some(_) => merged.push(item),
                None if self.known.get(item.id()) != Some(&encode_chat_item(&item)) => {
                    merged.push(item);
                }
                // A previously observed row absent from the owned baseline was removed by
                // another writer; its unchanged stale copy cannot resurrect it. An explicit
                // new revision under that identity is still a local edit to persist.
                None => {}
            }
        }
        if !merged.iter().any(|item| rows.contains_key(item.id())) {
            // With no shared anchor, the database rows precede equally stamped local
            // additions: they existed before this snapshot was introduced.
            let additions = std::mem::replace(&mut merged, stored.clone());
            for item in additions {
                lk_agents_abi::llm::chat_context::ordering::commit_by_instant(&mut merged, item);
            }
        }
        // A model view omits history-only records (handoffs, another agent's items). Preserve
        // those rows' stored neighbors instead of moving them behind timestamp ties in the
        // narrower view. Local new items keep the session ledger's relative order.
        for (index, row) in stored.iter().enumerate() {
            if !merged.iter().any(|item| item.id() == row.id()) {
                let next = stored[index + 1..]
                    .iter()
                    .find_map(|neighbor| merged.iter().position(|item| item.id() == neighbor.id()));
                let previous = stored[..index].iter().rev().find_map(|neighbor| {
                    merged.iter().position(|item| item.id() == neighbor.id()).map(|at| at + 1)
                });
                if let Some(at) = next.or(previous) {
                    merged.insert(at, row.clone());
                } else {
                    lk_agents_abi::llm::chat_context::ordering::commit_by_instant(
                        &mut merged,
                        row.clone(),
                    );
                }
            }
        }
        self.known.extend(stored.iter().map(|item| (item.id().to_owned(), encode_chat_item(item))));
        let written = instructions
            .iter()
            .chain(&merged)
            .filter(|item| rows.contains_key(item.id()))
            .cloned()
            .collect();
        instructions.extend(merged);
        (instructions, written)
    }

    /// The partition holds `rows`, the stored forms of `items`, as this store wrote them.
    fn wrote(&mut self, items: &[ChatItem], rows: &[StoredItem]) {
        for (item, row) in items.iter().zip(rows).filter(|(item, _)| conversational(item)) {
            self.known.insert(item.id().to_owned(), row.clone());
        }
    }
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
        let history = with_thread(&*backend, &thread_id, async |thread| {
            let partition_key = thread.partition(&partition_key);
            let history = if explicit_history.is_none() {
                partition_key.history().await?
            } else {
                Vec::new()
            };
            Ok::<_, StorageError>(history)
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
        // Only rows actually read from storage establish a database baseline. Supplied
        // history is a local view, never proof that unseen rows were deleted or edited.
        let known =
            items.iter().map(|item| (item.id().to_owned(), encode_chat_item(item))).collect();
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
            state: tokio::sync::Mutex::new(State { known, active: None }),
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
        speech: Option<u64>,
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
            active.speeches.extend(speech);
            let stored = active
                .run
                .history()
                .iter()
                .map(|row| decode_chat_item(row.payload.clone()).map_err(StorageError::from))
                .collect::<Result<Vec<_>, _>>()?;
            return Ok(state.reconcile(stored, local).0);
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
            let (merged, written) = {
                let rows = run.history();
                let mut stored = rows
                    .iter()
                    .map(|row| decode_chat_item(row.payload.clone()).map_err(StorageError::from))
                    .collect::<Result<Vec<_>, _>>()?;
                // A previous voice activation ended mid-input. Its latest durable words are
                // retained as one user item, just as a lost recognition connection retains them.
                if let Some(routing) = run.state() {
                    for pending in crate::session_state::pending_voice_inputs(routing)? {
                        if !stored.iter().any(|item| item.id() == pending.id) {
                            stored.push(ChatItem::Message(pending));
                        }
                    }
                }
                state.reconcile(stored, local)
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
                let abandoned =
                    run.abandon(RunOutcome::Failed, AbandonReason::VoiceLoad.as_str()).await;
                crate::telemetry::cleanup_failed(
                    &self.partition_key,
                    "abandoning the run",
                    abandoned,
                );
                thread.close().await;
                return Err(error);
            }
        };
        *self.lease.lock() = run.lease().cloned();
        state.active = Some(Active {
            thread,
            run,
            speeches: speech.into_iter().collect(),
            history_sequence: None,
            input_sequence: None,
            pending_user_turns: Vec::new(),
        });
        if let Some(routing) = self.activation.session_state() {
            let rows: Vec<StoredItem> = written.iter().map(encode_chat_item).collect();
            state
                .active
                .as_mut()
                .expect("turn was installed")
                .run
                .checkpoint_session(routing, rows.clone())
                .await?;
            state.wrote(&written, &rows);
        }
        Ok(merged)
    }

    async fn finish(
        &self,
        speech: Option<u64>,
        sequence: u64,
        history: Vec<ChatItem>,
    ) -> Result<(), StorageError> {
        let mut state = self.state.lock().await;
        let Some(active) = state.active.as_mut() else { return Ok(()) };
        if let Some(speech) = speech {
            active.speeches.remove(&speech);
        }
        let newer = active.history_sequence.is_none_or(|held| sequence > held);
        let history = if newer {
            history
        } else {
            active
                .run
                .history()
                .iter()
                .map(|row| decode_chat_item(row.payload.clone()).map_err(StorageError::from))
                .collect::<Result<Vec<_>, _>>()?
        };
        if !active.speeches.is_empty() {
            if !newer {
                return Ok(());
            }
            // Another speech keeps the database run open, but this speech's successful receipt
            // must already protect its final item revisions. A subsequent abort retains this
            // checkpoint, including a cut prefix and its interruption flag.
            let agent = self
                .activation
                .session_state()
                .and_then(|state| {
                    state.get("agent").and_then(|agent| agent.as_str()).map(str::to_owned)
                })
                .unwrap_or_default();
            self.checkpoint_history(&mut state, agent, &history).await?;
            state.active.as_mut().expect("checkpoint kept its run").history_sequence =
                Some(sequence);
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
        let committed: BTreeMap<String, StoredItem> =
            history.iter().map(|item| (item.id().to_owned(), encode_chat_item(item))).collect();
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
        let prepared = history.and_then(|history| {
            *active.run.history_mut() = history;
            active.pending_user_turns.retain(|message| !committed.contains_key(&message.id));
            let agent = self
                .activation
                .session_state()
                .and_then(|state| {
                    state.get("agent").and_then(|agent| agent.as_str()).map(str::to_owned)
                })
                .unwrap_or_default();
            let routing = crate::session_state::voice_progress_value(
                agent,
                active.run.history().iter().map(|item| item.id.clone()).collect(),
                &active.pending_user_turns,
            )?;
            self.activation.stage_session(routing.clone())?;
            active.run.set_state(routing);
            active.run.stage_scopes(self.activation.writes());
            active.run.stage_statements(self.activation.statements());
            Ok(())
        });
        let result = match prepared {
            Ok(()) => {
                active.run.commit(CommitOptions::default()).await.map(|committed| committed.version)
            }
            Err(error) => {
                let abandoned = active
                    .run
                    .abandon(RunOutcome::Failed, AbandonReason::VoiceEncode.as_str())
                    .await;
                crate::telemetry::cleanup_failed(
                    &self.partition_key,
                    "abandoning the run",
                    abandoned,
                );
                Err(error)
            }
        };
        if result.is_err() {
            // A commit still unknown at its deadline is the host's to learn: what this store
            // knows of the partition no longer holds, and the release fences out a commit that
            // never applied.
            let unknown = matches!(
                result,
                Err(StorageError::VersionConflict { .. } | StorageError::Uncertain(_))
            );
            if unknown || self.lease_lost() {
                self.lost.store(true, Ordering::Release);
            }
            if let Some(lease) = active.run.lease() {
                let released = lease.release(None).await;
                crate::telemetry::cleanup_failed(
                    &self.partition_key,
                    "releasing the lease",
                    released,
                );
            }
        }
        self.released();
        active.thread.close().await;
        self.activation.end_managed(result.is_ok(), fields);
        if result.is_ok() {
            state.known.extend(committed);
            Ok(())
        } else {
            result.map(|_| ())
        }
    }

    /// Durably publish conversation items without ending the shared database run. Used by
    /// progress barriers and by an individual speech settling while another still owns it.
    async fn checkpoint_history(
        &self,
        state: &mut State,
        agent: String,
        history: &[ChatItem],
    ) -> Result<(), StorageError> {
        let stored = state
            .active
            .as_ref()
            .expect("the caller holds the active run")
            .run
            .history()
            .iter()
            .map(|row| decode_chat_item(row.payload.clone()).map_err(StorageError::from))
            .collect::<Result<Vec<_>, _>>()?;
        let history: Vec<_> = state
            .reconcile(stored, history.to_vec())
            .0
            .into_iter()
            .filter(conversational)
            .collect();
        state
            .active
            .as_mut()
            .expect("the caller holds the active run")
            .pending_user_turns
            .retain(|pending| !history.iter().any(|item| item.id() == pending.id));
        let routing = crate::session_state::voice_progress_value(
            agent,
            history
                .iter()
                .filter(|item| conversational(item))
                .map(|item| item.id().to_owned())
                .collect(),
            &state.active.as_ref().expect("the caller holds the active run").pending_user_turns,
        )?;
        let rows: Vec<StoredItem> =
            history.iter().filter(|item| conversational(item)).map(encode_chat_item).collect();
        state
            .active
            .as_mut()
            .expect("the caller holds the active run")
            .run
            .checkpoint_session(routing.clone(), rows.clone())
            .await?;
        state.wrote(&history, &rows);
        self.activation.stage_session(routing)?;
        self.activation.acknowledge_session_state();
        Ok(())
    }

    async fn abandon(&self, speech: u64) -> Result<(), StorageError> {
        let mut state = self.state.lock().await;
        let Some(active) = state.active.as_mut() else { return Ok(()) };
        active.speeches.remove(&speech);
        if !active.speeches.is_empty() {
            return Ok(());
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

    fn settle_in_background(
        self: &Arc<Self>,
        speech: u64,
        sequence: u64,
        history: Vec<ChatItem>,
        events: Events,
    ) {
        let store = self.clone();
        self.tasks.spawn(&*self.background.spawner, "voice storage settle", async move {
            // Closing may cancel the speech's original acquire before it installs a run.
            // A successful final receipt still protects these concrete items, so establish
            // ownership and reconcile first instead of treating an absent run as a save.
            let result = match store.begin(Some(speech), history).await {
                Ok(history) => store.finish(Some(speech), sequence, history).await,
                Err(error) => Err(error),
            };
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
            VoiceStorageCommand::Settle { speech, sequence, history } => {
                self.settle_in_background(speech, sequence, history, events)
            }
            VoiceStorageCommand::Abort { speech } => self.abandon_in_background(speech, events),
            VoiceStorageCommand::ObserveLoss { speech } => self.report_lease_loss(speech, events),
        }
        drop(closing);
        Ok(())
    }

    fn acquire(&self, speech: u64, local: Vec<ChatItem>) -> StorageWork<'_, Vec<ChatItem>> {
        Box::pin(self.begin(Some(speech), local))
    }
    fn conversation(
        &self,
        progress: lk_agents_abi::storage::voice::VoiceConversationProgress,
    ) -> StorageWork<'_, Vec<ChatItem>> {
        Box::pin(async move {
            use lk_agents_abi::storage::voice::VoiceConversationBoundary;
            let mut history = self.begin(None, progress.history).await?;
            let mut state = self.state.lock().await;
            let active = state
                .active
                .as_mut()
                .ok_or_else(|| StorageError::Invalid("the voice conversation has ended".into()))?;
            if active.history_sequence.is_some_and(|held| progress.sequence <= held) {
                history = active
                    .run
                    .history()
                    .iter()
                    .map(|row| decode_chat_item(row.payload.clone()).map_err(StorageError::from))
                    .collect::<Result<Vec<_>, _>>()?;
            }
            if active.input_sequence.is_none_or(|held| progress.sequence > held) {
                active.pending_user_turns = progress.pending_user_turns;
                active.input_sequence = Some(progress.sequence);
            }
            if progress.boundary == VoiceConversationBoundary::Progress {
                self.checkpoint_history(&mut state, progress.agent, &history).await?;
                let active = state.active.as_mut().expect("checkpoint kept its run");
                active.history_sequence = Some(
                    active
                        .history_sequence
                        .map_or(progress.sequence, |held| held.max(progress.sequence)),
                );
            }
            drop(state);
            if progress.boundary == VoiceConversationBoundary::UserTurnCompleted {
                self.finish(None, progress.sequence, history.clone()).await?;
            }
            Ok(history)
        })
    }
    fn checkpoint(
        &self,
        speech: u64,
        sequence: u64,
        agent: String,
        history: Vec<ChatItem>,
    ) -> StorageWork<'_, ()> {
        Box::pin(async move {
            let mut state = self.state.lock().await;
            let active = state
                .active
                .as_mut()
                .filter(|active| active.speeches.contains(&speech))
                .ok_or_else(|| {
                StorageError::Invalid("the turn does not own the partition".into())
            })?;
            if active.history_sequence.is_some_and(|accepted| sequence <= accepted) {
                return Ok(());
            }
            self.checkpoint_history(&mut state, agent, &history).await?;
            state.active.as_mut().expect("checkpoint kept its run").history_sequence =
                Some(sequence);
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
            let fields = self.activation.lock_managed().await;
            result =
                active.run.abandon(RunOutcome::Canceled, AbandonReason::VoiceClose.as_str()).await;
            self.released();
            active.thread.close().await;
            self.activation.end_managed(false, fields);
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
            let rows: Vec<StoredItem> = merged.iter().map(encode_chat_item).collect();
            active.run.checkpoint_session(routing.clone(), rows.clone()).await?;
            state.wrote(&merged, &rows);
            store.activation.stage_session(routing)?;
            Ok(())
        })
    }
}
