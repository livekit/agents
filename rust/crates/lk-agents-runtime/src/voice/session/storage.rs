//! Storage startup: the call's lane, on the supplied thread or a new local one.

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

pub struct SessionStorage {
    env: ProcessEnv,
    namespace: Option<Arc<Namespace>>,
    thread_id: Option<String>,
    lane: String,
    initial_history: Option<Vec<ChatItem>>,
    opened: OnceCell<Opened>,
}

pub(super) struct Opened {
    namespace: Arc<Namespace>,
    thread_id: String,
    scope_id: String,
    activation: ContextStore,
    voice_store: Arc<VoiceStore>,
    history: Option<Vec<ChatItem>>,
}

impl SessionStorage {
    /// The existing host ABI exposes infallible getters after `open`. Native startup uses the
    /// returned ready payload directly instead of relying on this protocol-only entry point.
    fn opened(&self) -> &Opened {
        self.opened.get().expect("session storage must be opened before reading it")
    }

    /// Native startup receives the initialized payload, so it cannot read a pre-open handle.
    pub(super) async fn open_ready(&self) -> Result<&Opened, StorageError> {
        self.opened.get_or_try_init(|| self.load()).await
    }

    /// The supplied thread's lane; without one, or when it does not load, a new thread of the
    /// local SQLite storage.
    async fn load(&self) -> Result<Opened, StorageError> {
        if let (Some(namespace), Some(thread_id)) = (&self.namespace, &self.thread_id) {
            let initial_history = self.initial_history.clone();
            match self.lane(Arc::clone(namespace), thread_id.clone(), initial_history).await {
                Ok(opened) => return Ok(opened),
                Err(error) => {
                    log::warn!(target: lk_agents_abi::LOG_TARGET, threadId = thread_id.as_str(); "the voice lane could not be loaded, the call continues on a new local thread: {error}")
                }
            }
        }
        let local = Arc::new(Namespace::new(&self.env, None, None, None, None, None, None)?);
        let thread = local.create_thread(None, None, 0).await?;
        // A new thread has nothing to load: the call starts on the history it was given.
        let history = self.initial_history.clone().unwrap_or_default();
        self.lane(local, thread.thread_id, Some(history)).await
    }

    async fn lane(
        &self,
        namespace: Arc<Namespace>,
        thread_id: String,
        initial_history: Option<Vec<ChatItem>>,
    ) -> Result<Opened, StorageError> {
        let store = namespace
            ._core_voice_store(thread_id.clone(), self.lane.clone(), initial_history)
            .await?;
        Ok(Opened {
            namespace,
            thread_id,
            scope_id: format!("root:{}", self.lane),
            activation: store.activation(),
            history: store.initial_history(),
            voice_store: Arc::new(store),
        })
    }
}

impl Opened {
    pub(super) fn voice_store(&self) -> Arc<VoiceStore> {
        Arc::clone(&self.voice_store)
    }

    /// The call's thread: the conversation its spans record.
    pub(super) fn thread(&self) -> &str {
        &self.thread_id
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
        lane: String,
        _fallback_lane: String,
        initial_history: Option<Vec<ChatItem>>,
    ) -> Self {
        Self {
            env: env.clone(),
            namespace,
            thread_id,
            lane,
            initial_history,
            opened: OnceCell::new(),
        }
    }
    async fn open(&self) -> Result<(), StorageError> {
        self.open_ready().await.map(|_| ())
    }
    fn namespace(&self) -> Arc<Namespace> {
        Arc::clone(&self.opened().namespace)
    }
    fn thread_id(&self) -> String {
        self.opened().thread_id.clone()
    }
    fn scope_id(&self) -> String {
        self.opened().scope_id.clone()
    }
    fn activation(&self) -> ContextStore {
        self.opened().activation.clone()
    }
    fn voice_store(&self) -> Arc<VoiceStore> {
        self.opened().voice_store()
    }
    fn history(&self) -> Option<Vec<ChatItem>> {
        self.opened().history()
    }
    async fn aclose(&self) {
        let Some(opened) = self.opened.get() else { return };
        if let Err(error) = opened.voice_store.aclose().await {
            log::warn!(target: lk_agents_abi::LOG_TARGET, "the voice store could not close: {error}");
        }
        opened.activation.close();
    }
}

// Typed storage commands and receipts, over the session's existing event queue.

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
