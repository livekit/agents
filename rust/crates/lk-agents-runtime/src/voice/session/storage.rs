use crate::telemetry;
use crate::voice::VoiceRuntime;
use crate::voice::session::Tx;
use lk_agents_abi::StreamInput;
use lk_agents_abi::errors::{ApiError, ProviderError};
use lk_agents_abi::storage::Unapplied;
use lk_agents_abi::storage::context::ContextActivation as _;
use lk_agents_abi::storage::voice::VoiceStore as _;
use lk_agents_abi::storage::voice::{VoiceStorageCommand, VoiceStorageEvent};
use lk_agents_abi::{ChatItem, storage::StorageError};
use lk_agents_net::env::ProcessEnv;
use lk_agents_storage::{
    context::store::{ContextStore, SessionSinkGuard},
    storage::Namespace,
    voice::VoiceStore,
};
use lk_agents_voice::agent_session::machine::activity::scheduling::SpeechId;
use lk_agents_voice::agent_session::{StorageRequest, VoiceEvent};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use tokio::sync::OnceCell;

/// The call's partition on the supplied thread or, without one, in memory.
pub struct SessionStorage {
    namespace: Option<Arc<Namespace>>,
    thread_id: Option<String>,
    partition_key: String,
    agent: String,
    initial_history: Option<Vec<ChatItem>>,
    /// Where the partition's health metrics go before a session names its own span.
    process: telemetry::Cx,
    opened: OnceCell<Opened>,
    supplied_activation: parking_lot::Mutex<Option<ContextStore>>,
}

pub(super) struct Opened {
    partition_key: Option<Partition>,
    scope_id: String,
    activation: ContextStore,
    /// Only the in-memory activation created by this owner is closed here.
    owns_activation: bool,
    session_sink: parking_lot::Mutex<Option<SessionSinkGuard>>,
    history: Option<Vec<ChatItem>>,
}

/// The supplied thread's partition, as it loaded.
struct Partition {
    namespace: Arc<Namespace>,
    thread_id: String,
    voice_store: Arc<VoiceStore>,
}

impl SessionStorage {
    /// The thread the call was given.
    pub(super) fn supplied_thread(&self) -> Option<&str> {
        self.thread_id.as_deref()
    }

    /// The call's agent, which the spans of a start that failed before its session ran name.
    pub(super) fn agent(&self) -> &str {
        &self.agent
    }

    /// The existing host ABI exposes infallible getters after `open`. Native startup uses the
    /// returned ready payload directly instead of relying on this protocol-only entry point.
    fn opened(&self) -> &Opened {
        self.opened.get().expect("session storage must be opened before reading it")
    }

    /// Native startup receives the initialized payload, so it cannot read a pre-open handle.
    /// The partition's health metrics record under `cx`, the session's span.
    pub(super) async fn open_ready(&self, cx: &telemetry::Cx) -> Result<&Opened, StorageError> {
        self.opened.get_or_try_init(|| self.load(cx)).await
    }

    /// The supplied thread's partition, whose failure is the call's; without one, the call runs in
    /// memory on the history it was given.
    async fn load(&self, cx: &telemetry::Cx) -> Result<Opened, StorageError> {
        let scope_id = format!("root:{}", self.partition_key);
        let activation = self.supplied_activation.lock().clone();
        if activation.is_some() && self.initial_history.is_some() {
            return Err(StorageError::Invalid(
                "explicit history cannot be combined with a bound context".into(),
            ));
        }
        let history = match &activation {
            Some(activation) => activation.bound_history()?,
            None => self.initial_history.clone(),
        };
        if activation.is_some()
            && let Some(history) = &history
        {
            lk_agents_core::session::validate_resume(history)
                .map_err(|error| StorageError::Invalid(error.to_string()))?;
        }
        let (Some(namespace), Some(thread_id)) = (&self.namespace, &self.thread_id) else {
            return Ok(Opened {
                partition_key: None,
                scope_id,
                owns_activation: activation.is_none(),
                session_sink: parking_lot::Mutex::new(None),
                activation: activation
                    .map(Ok)
                    .unwrap_or_else(|| ContextStore::new(Vec::new(), None, false))?,
                history,
            });
        };
        let store = namespace
            .voice_store(
                thread_id.clone(),
                self.partition_key.clone(),
                history,
                cx.clone(),
                activation,
            )
            .await?;
        let store = Arc::new(store);
        let session_sink = store.activation().attach_session_sink(store.session_sink())?;
        Ok(Opened {
            scope_id,
            activation: store.activation(),
            owns_activation: false,
            session_sink: parking_lot::Mutex::new(Some(session_sink)),
            history: store.initial_history(),
            partition_key: Some(Partition {
                namespace: Arc::clone(namespace),
                thread_id: thread_id.clone(),
                voice_store: store,
            }),
        })
    }
}

impl Opened {
    pub(super) fn voice_store(&self) -> Option<Arc<VoiceStore>> {
        self.partition_key.as_ref().map(|partition_key| Arc::clone(&partition_key.voice_store))
    }

    /// The call's thread: the conversation its spans record.
    pub(super) fn thread(&self) -> Option<&str> {
        self.partition_key.as_ref().map(|partition_key| partition_key.thread_id.as_str())
    }

    pub(super) fn history(&self) -> Option<Vec<ChatItem>> {
        self.history.clone()
    }
}

impl lk_agents_abi::voice::session_storage::SessionStorage for SessionStorage {
    type Env = ProcessEnv;
    type Namespace = Namespace;
    type Activation = ContextStore;
    type VoiceStore = VoiceStore;

    fn new(
        env: &ProcessEnv,
        namespace: Option<Arc<Namespace>>,
        thread_id: Option<String>,
        partition_key: String,
        agent: String,
        initial_history: Option<Vec<ChatItem>>,
    ) -> Self {
        let process = env.tracer.root();
        Self {
            namespace,
            thread_id,
            partition_key,
            agent,
            initial_history,
            process,
            opened: OnceCell::new(),
            supplied_activation: parking_lot::Mutex::new(None),
        }
    }
    fn use_activation(&self, activation: Arc<ContextStore>) -> Result<(), StorageError> {
        if self.opened.get().is_some() {
            return Err(StorageError::Invalid("voice storage is already open".into()));
        }
        *self.supplied_activation.lock() = Some((*activation).clone());
        Ok(())
    }
    async fn open(&self) -> Result<(), StorageError> {
        self.open_ready(&self.process).await.map(|_| ())
    }
    fn namespace(&self) -> Option<Arc<Namespace>> {
        self.opened()
            .partition_key
            .as_ref()
            .map(|partition_key| Arc::clone(&partition_key.namespace))
    }
    fn thread_id(&self) -> Option<String> {
        self.opened().thread().map(str::to_owned)
    }
    fn scope_id(&self) -> String {
        self.opened().scope_id.clone()
    }
    fn activation(&self) -> ContextStore {
        self.opened().activation.clone()
    }
    fn voice_store(&self) -> Option<Arc<VoiceStore>> {
        self.opened().voice_store()
    }
    fn history(&self) -> Option<Vec<ChatItem>> {
        self.opened().history()
    }
    async fn aclose(&self) {
        let Some(opened) = self.opened.get() else { return };
        if let Some(partition_key) = &opened.partition_key
            && let Err(error) = partition_key.voice_store.aclose().await
        {
            log::warn!(target: lk_agents_abi::LOG_TARGET, "the voice store could not close: {error}");
        }
        let session_sink = opened.session_sink.lock().take();
        drop(session_sink);
        if opened.owns_activation {
            opened.activation.close();
        }
    }
}

/// Typed storage receipts, over the session's existing event queue. A reporter that owes a
/// request's answer gives it exactly once: dropped unanswered (its task died, or the store dropped
/// it), it answers with that death, so the speech waiting on it never waits on nothing.
pub(in crate::voice) struct StorageReporter {
    tx: Tx,
    owed: Option<(SpeechId, StorageRequest)>,
    read: Vec<ChatItem>,
    answered: AtomicBool,
}

impl StorageReporter {
    /// Watches the partition's loss: no answer is owed.
    pub(in crate::voice) fn observer(tx: Tx) -> Self {
        Self { tx, owed: None, read: Vec::new(), answered: AtomicBool::new(false) }
    }

    /// Owes the answer to `speech`'s `request`.
    pub(in crate::voice) fn answering(
        tx: Tx,
        speech: SpeechId,
        request: StorageRequest,
        read: Vec<ChatItem>,
    ) -> Self {
        Self { tx, owed: Some((speech, request)), read, answered: AtomicBool::new(false) }
    }

    fn answer(
        &self,
        speech: u64,
        request: StorageRequest,
        result: Result<(), StorageError>,
    ) -> Result<(), ProviderError> {
        self.answered.store(true, Ordering::Release);
        let speech = SpeechId(speech);
        self.tx.send(VoiceEvent::StorageAnswered { speech, request, result }).map_err(|_| {
            ProviderError::Local(ApiError::connection(Some("the voice session has closed".into())))
        })
    }

    /// Answers the request this reporter owes, if any, here rather than from the store.
    fn answer_owed(&self, result: Result<(), StorageError>) {
        if let Some((speech, request)) = self.owed {
            let _ = self.answer(speech.0, request, result);
        }
    }
}

impl Drop for StorageReporter {
    fn drop(&mut self) {
        if let Some((speech, request)) = self.owed
            && !self.answered.load(Ordering::Acquire)
        {
            let result = Err(StorageError::Unavailable(Unapplied::local("the storage task ended")));
            let _ = self.tx.send(VoiceEvent::StorageAnswered { speech, request, result });
        }
    }
}

impl StreamInput<VoiceStorageEvent> for StorageReporter {
    fn push(&self, event: VoiceStorageEvent) -> Result<(), ProviderError> {
        match event {
            VoiceStorageEvent::Settled { speech, result, lease_lost } => {
                if lease_lost {
                    let _ = self.tx.send(VoiceEvent::StorageLost(SpeechId(speech)));
                }
                let result = result.map(|history| {
                    let _ = self
                        .tx
                        .send(VoiceEvent::StorageReloaded { read: self.read.clone(), history });
                });
                self.answer(speech, StorageRequest::Commit, result)
            }
            VoiceStorageEvent::Abandoned { speech, result } => {
                self.answer(speech, StorageRequest::Abort, result)
            }
            VoiceStorageEvent::LeaseLost { speech } => {
                self.tx.send(VoiceEvent::StorageLost(SpeechId(speech))).map_err(|_| {
                    ProviderError::Local(ApiError::connection(Some(
                        "the voice session has closed".into(),
                    )))
                })
            }
        }
    }
    fn flush(&self) -> Result<(), ProviderError> {
        Ok(())
    }
    fn end_input(&self) {}
}

impl VoiceRuntime {
    pub(in crate::voice) fn storage_command(&self, command: VoiceStorageCommand) {
        let tx = self.reports.clone();
        let reporter = Arc::new(match &command {
            VoiceStorageCommand::Settle { speech, history, .. } => StorageReporter::answering(
                tx,
                SpeechId(*speech),
                StorageRequest::Commit,
                history.items.clone(),
            ),
            VoiceStorageCommand::Abort { speech } => {
                StorageReporter::answering(tx, SpeechId(*speech), StorageRequest::Abort, Vec::new())
            }
            VoiceStorageCommand::ObserveLoss { .. } => StorageReporter::observer(tx),
        });
        match &self.storage {
            Some(storage) => {
                if let Err(error) = storage.clone().dispatch(command, reporter.clone()) {
                    reporter.answer_owed(Err(error));
                }
            }
            // No store: the save has nothing to write.
            None => reporter.answer_owed(Ok(())),
        }
    }
}

/// A conversation checkpoint always returns one result, including if its task dies. It belongs
/// to the session rather than any speech, so releasing an interrupted speech cannot drop it.
struct ConversationReceipt {
    tx: Tx,
    sequence: u64,
    read: Option<Vec<ChatItem>>,
}

impl ConversationReceipt {
    fn settle(mut self, result: Result<lk_agents_abi::storage::voice::VoiceHistory, StorageError>) {
        let read = self.read.take().expect("the receipt is owed once");
        let _ =
            self.tx.send(VoiceEvent::ConversationSaved { sequence: self.sequence, read, result });
    }
}

impl Drop for ConversationReceipt {
    fn drop(&mut self) {
        if let Some(read) = self.read.take() {
            let result = Err(StorageError::Unavailable(Unapplied::local(
                "the conversation checkpoint task ended",
            )));
            let _ = self.tx.send(VoiceEvent::ConversationSaved {
                sequence: self.sequence,
                read,
                result,
            });
        }
    }
}

impl VoiceRuntime {
    pub(in crate::voice) fn save_conversation(
        &self,
        progress: lk_agents_abi::storage::voice::VoiceConversationProgress,
    ) {
        let receipt = ConversationReceipt {
            tx: self.reports.clone(),
            sequence: progress.sequence,
            read: Some(progress.history.items.clone()),
        };
        let storage = self.storage.clone();
        self.spawn("voice conversation progress", async move {
            let result = match storage {
                Some(storage) => storage.conversation(progress).await,
                None => Ok(progress.history),
            };
            receipt.settle(result);
        });
    }
}
