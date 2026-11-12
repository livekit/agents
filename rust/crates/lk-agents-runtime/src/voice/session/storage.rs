use crate::telemetry;
use crate::voice::VoiceRuntime;
use crate::voice::session::Tx;
use lk_agents_abi::StreamInput;
use lk_agents_abi::errors::{ApiError, ProviderError};
use lk_agents_abi::storage::context::ContextActivation as _;
use lk_agents_abi::storage::voice::VoiceStore as _;
use lk_agents_abi::storage::voice::{VoiceStorageCommand, VoiceStorageEvent};
use lk_agents_abi::{ChatItem, storage::StorageError};
use lk_agents_net::env::ProcessEnv;
use lk_agents_storage::{context::store::ContextStore, storage::Namespace, voice::VoiceStore};
use lk_agents_voice::agent_session::VoiceEvent;
use lk_agents_voice::agent_session::machine::activity::scheduling::SpeechId;
use std::sync::Arc;
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
        let (Some(namespace), Some(thread_id)) = (&self.namespace, &self.thread_id) else {
            return Ok(Opened {
                partition_key: None,
                scope_id,
                activation: self
                    .supplied_activation
                    .lock()
                    .clone()
                    .map(Ok)
                    .unwrap_or_else(|| ContextStore::new(Vec::new(), None, false))?,
                history: self.initial_history.clone(),
            });
        };
        let history = self.initial_history.clone();
        let activation = self.supplied_activation.lock().clone();
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
        store.activation().set_session_sink(store.session_sink())?;
        Ok(Opened {
            scope_id,
            activation: store.activation(),
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
        opened.activation.close();
    }
}

/// Typed storage receipts, over the session's existing event queue.
pub(in crate::voice) struct StorageReporter(pub Tx);

impl StreamInput<VoiceStorageEvent> for StorageReporter {
    fn push(&self, event: VoiceStorageEvent) -> Result<(), ProviderError> {
        let input = match event {
            VoiceStorageEvent::Settled { speech, result, lease_lost } => {
                if let Err(error) = result {
                    return self
                        .0
                        .send(VoiceEvent::StorageFailed { speech: SpeechId(speech), error })
                        .map_err(|_| {
                            ProviderError::Local(ApiError::connection(Some(
                                "the voice session has closed".into(),
                            )))
                        });
                }
                if lease_lost {
                    let _ = self.0.send(VoiceEvent::StorageLost(SpeechId(speech)));
                }
                VoiceEvent::StorageSettled(SpeechId(speech))
            }
            VoiceStorageEvent::Abandoned { speech, result } => {
                if let Err(error) = result {
                    return self
                        .0
                        .send(VoiceEvent::StorageFailed { speech: SpeechId(speech), error })
                        .map_err(|_| {
                            ProviderError::Local(ApiError::connection(Some(
                                "the voice session has closed".into(),
                            )))
                        });
                }
                VoiceEvent::StorageAbandoned(SpeechId(speech))
            }
            VoiceStorageEvent::LeaseLost { speech } => VoiceEvent::StorageLost(SpeechId(speech)),
        };
        self.0.send(input).map_err(|_| {
            ProviderError::Local(ApiError::connection(Some("the voice session has closed".into())))
        })
    }
    fn flush(&self) -> Result<(), ProviderError> {
        Ok(())
    }
    fn end_input(&self) {}
}

impl VoiceRuntime {
    pub(in crate::voice) fn storage_command(&self, command: VoiceStorageCommand) {
        let event = match &command {
            VoiceStorageCommand::Settle { speech, .. } => Some(VoiceStorageEvent::Settled {
                speech: *speech,
                result: Ok(()),
                lease_lost: false,
            }),
            VoiceStorageCommand::Abort { speech } => {
                Some(VoiceStorageEvent::Abandoned { speech: *speech, result: Ok(()) })
            }
            VoiceStorageCommand::ObserveLoss { .. } => None,
        };
        let reporter = Arc::new(StorageReporter(self.reports.clone()));
        if let Some(storage) = &self.storage {
            if let Err(error) = storage.clone().dispatch(command, reporter.clone())
                && let Some(mut event) = event
            {
                match &mut event {
                    VoiceStorageEvent::Settled { result, .. }
                    | VoiceStorageEvent::Abandoned { result, .. } => *result = Err(error),
                    VoiceStorageEvent::LeaseLost { .. } => {}
                }
                let _ = reporter.push(event);
            }
        } else if let Some(event) = event {
            let _ = reporter.push(event);
        }
    }
}
