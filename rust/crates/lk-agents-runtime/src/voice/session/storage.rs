use crate::voice::VoiceRuntime;
use crate::voice::session::Tx;
use lk_agents_abi::StreamInput;
use lk_agents_abi::errors::{ApiError, ProviderError};
use lk_agents_abi::storage::Storage as _;
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

/// The call's lane on the supplied thread or, without one, in memory.
pub struct SessionStorage {
    namespace: Option<Arc<Namespace>>,
    thread_id: Option<String>,
    lane: String,
    agent: String,
    initial_history: Option<Vec<ChatItem>>,
    opened: OnceCell<Opened>,
}

pub(super) struct Opened {
    lane: Option<Lane>,
    scope_id: String,
    activation: ContextStore,
    history: Option<Vec<ChatItem>>,
}

/// The supplied thread's lane, as it loaded.
struct Lane {
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
    pub(super) async fn open_ready(&self) -> Result<&Opened, StorageError> {
        self.opened.get_or_try_init(|| self.load()).await
    }

    /// The supplied thread's lane, whose failure is the call's; without one, the call runs in
    /// memory on the history it was given.
    async fn load(&self) -> Result<Opened, StorageError> {
        let scope_id = format!("root:{}", self.lane);
        let (Some(namespace), Some(thread_id)) = (&self.namespace, &self.thread_id) else {
            return Ok(Opened {
                lane: None,
                scope_id,
                activation: ContextStore::new(Vec::new(), None, false)?,
                history: self.initial_history.clone(),
            });
        };
        let store = namespace
            ._core_voice_store(thread_id.clone(), self.lane.clone(), self.initial_history.clone())
            .await?;
        Ok(Opened {
            scope_id,
            activation: store.activation(),
            history: store.initial_history(),
            lane: Some(Lane {
                namespace: Arc::clone(namespace),
                thread_id: thread_id.clone(),
                voice_store: Arc::new(store),
            }),
        })
    }
}

impl Opened {
    pub(super) fn voice_store(&self) -> Option<Arc<VoiceStore>> {
        self.lane.as_ref().map(|lane| Arc::clone(&lane.voice_store))
    }

    /// The call's thread: the conversation its spans record.
    pub(super) fn thread(&self) -> Option<&str> {
        self.lane.as_ref().map(|lane| lane.thread_id.as_str())
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
        _env: &ProcessEnv,
        namespace: Option<Arc<Namespace>>,
        thread_id: Option<String>,
        lane: String,
        agent: String,
        initial_history: Option<Vec<ChatItem>>,
    ) -> Self {
        Self { namespace, thread_id, lane, agent, initial_history, opened: OnceCell::new() }
    }
    async fn open(&self) -> Result<(), StorageError> {
        self.open_ready().await.map(|_| ())
    }
    fn namespace(&self) -> Option<Arc<Namespace>> {
        self.opened().lane.as_ref().map(|lane| Arc::clone(&lane.namespace))
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
        if let Some(lane) = &opened.lane
            && let Err(error) = lane.voice_store.aclose().await
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
                    log::warn!(target: lk_agents_abi::LOG_TARGET, "the voice turn could not commit: {error}");
                }
                if lease_lost {
                    let _ = self.0.send(VoiceEvent::StorageLost(SpeechId(speech)));
                }
                VoiceEvent::StorageSettled(SpeechId(speech))
            }
            VoiceStorageEvent::Abandoned { speech, result } => {
                if let Err(error) = result {
                    log::warn!(target: lk_agents_abi::LOG_TARGET, "the voice turn could not abandon: {error}");
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
