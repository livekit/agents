use lk_agents_net::clock::RealClock;
use lk_agents_net::signal::{self, Producer};
use lk_agents_net::task::{TaskSet, TokioSpawner};
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
    lane: String,
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
    closing: Mutex<bool>,
    /// The session the store serves: its turns' health metrics record under it.
    cx: Cx,
}

struct State {
    /// Each conversational item as this store last read it from the lane or committed it.
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
        cx: Cx,
    ) -> Result<Self, StorageError> {
        let (history, version) = with_thread(&*backend, &thread_id, async |thread| {
            let lane = thread.lane(&lane);
            let history =
                if explicit_history.is_none() { lane.history().await? } else { Vec::new() };
            let version = lane.info().await?.map_or(0, |lane| lane.version);
            Ok::<_, StorageError>((history, version))
        })
        .await?;
        let items = history
            .iter()
            .map(|row| decode_chat_item(row.payload.clone()).map_err(StorageError::from))
            .collect::<Result<Vec<_>, _>>()?;
        let known = explicit_history
            .as_ref()
            .unwrap_or(&items)
            .iter()
            .map(|item| (item.id().to_owned(), item.clone()))
            .collect();
        let activation =
            ContextStore::bound(backend.clone(), thread_id.clone(), lane.clone(), cx.clone())
                .await?;
        Ok(Self {
            backend,
            thread_id,
            lane,
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
            closing: Mutex::new(false),
            cx,
        })
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
            return Err(StorageError::LeaseLost("the voice lane lease was lost".into()));
        }
        if let Some(active) = &mut state.active {
            active.speeches.insert(speech);
            return Ok(local);
        }
        // Exclude direct saves before taking the lane, avoiding the inverse lock/lease order.
        let fields = self.activation.lock_managed().await;
        let thread = self.backend.open(&self.thread_id).await?.with_telemetry(self.cx.clone());
        let mut options = RunOptions::new(shortuuid("voice_"), "");
        options.channel = Some("voice".into());
        options.kind = crate::thread::run::LaneKind::Voice;
        let mut run = match thread.lane(&self.lane).open(options, &self.holder).await {
            Ok(run) => run,
            Err(error) => {
                thread.close().await;
                return Err(error);
            }
        };
        let prepared = (|| {
            // The lane as this store last saw it: the turn's own reading, edits included, is
            // the newest.
            let merged = if run.version() == state.version {
                // The turn's new items are known from now on: if it is abandoned, a later reload
                // drops them rather than committing them as another turn's.
                for item in local.iter().filter(|item| conversational(item)) {
                    state.known.entry(item.id().to_owned()).or_insert_with(|| item.clone());
                }
                local
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
                instructions
            };
            self.activation.begin_managed(
                run.context_scopes().to_vec(),
                run.state().cloned(),
                &fields,
            )?;
            Ok::<_, StorageError>(merged)
        })();
        let merged = match prepared {
            Ok(merged) => merged,
            Err(error) => {
                let _ = run.abandon(RunOutcome::Failed, AbandonReason::VoiceLoad.as_str()).await;
                thread.close().await;
                return Err(error);
            }
        };
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
        self.tasks.spawn(&TokioSpawner::current(), "voice storage settle", async move {
            let result = store.finish(speech, history).await;
            let lease_lost = store.lease_lost();
            let _ = events.push(VoiceStorageEvent::Settled { speech, result, lease_lost });
        });
    }

    fn abandon_in_background(self: &Arc<Self>, speech: u64, events: Events) {
        let store = self.clone();
        self.tasks.spawn(&TokioSpawner::current(), "voice storage abort", async move {
            let result = store.abandon(speech).await;
            let _ = events.push(VoiceStorageEvent::Abandoned { speech, result });
        });
    }

    fn report_lease_loss(&self, speech: u64, events: Events) {
        let observation = self.observe_loss();
        self.tasks.spawn(&TokioSpawner::current(), "voice lease loss", async move {
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
        self.tasks.wait_empty(&RealClock::new(), None).await;
        let (voice, context) =
            tokio::join!(self.holder.wait_released(), self.activation.wait_released());
        if !(voice && context) {
            log::warn!(target: lk_agents_abi::LOG_TARGET, lane = self.lane.as_str(); "the voice store closed with a lease release in flight");
        }
        result
    }
}

#[cfg(test)]
#[path = "voice_tests.rs"]
mod tests;
