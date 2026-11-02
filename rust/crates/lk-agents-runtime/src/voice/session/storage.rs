//! Storage startup effects; lane versus ephemeral fallback is selected by voice policy.

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
use lk_agents_storage::{context::store::ContextStore, namespace::Namespace, voice::VoiceStore};
use lk_agents_voice::agent_activity::scheduling::SpeechId;
use lk_agents_voice::agent_session::VoiceEvent;
use std::sync::Arc;
use tokio::sync::OnceCell;

pub struct SessionStorage {
    namespace: Option<Arc<Namespace>>,
    thread_id: Option<String>,
    lane: String,
    initial_history: Option<Vec<ChatItem>>,
    opened: OnceCell<Opened>,
}

pub(super) struct Opened {
    namespace: Option<Arc<Namespace>>,
    thread_id: Option<String>,
    scope_id: String,
    activation: ContextStore,
    voice_store: Option<Arc<VoiceStore>>,
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

    async fn load(&self) -> Result<Opened, StorageError> {
        use lk_agents_voice::agent_session::lifecycle::{StorageSource, StorageStart};
        let lane = self.namespace.clone().zip(self.thread_id.clone());
        let plan = StorageStart::new(lane.is_some());
        let loaded = if let Some((namespace, thread_id)) = lane.filter(|_| plan.load_lane()) {
            match namespace
                ._core_voice_store(
                    thread_id.clone(),
                    self.lane.clone(),
                    self.initial_history.clone(),
                )
                .await
            {
                Ok(store) => Some(Opened {
                    namespace: Some(namespace),
                    thread_id: Some(thread_id),
                    scope_id: format!("root:{}", self.lane),
                    activation: store.activation(),
                    history: store.initial_history(),
                    voice_store: Some(Arc::new(store)),
                }),
                Err(error) => {
                    log::warn!("the voice lane could not be loaded: {error}");
                    None
                }
            }
        } else {
            None
        };
        match plan.after_load(loaded) {
            StorageSource::Lane(ready) => return Ok(ready),
            StorageSource::Ephemeral => {}
        }
        let activation = ContextStore::new(Vec::new(), None, false)?;
        Ok(Opened {
            namespace: None,
            thread_id: None,
            scope_id: "root".to_owned(),
            activation,
            voice_store: None,
            history: None,
        })
    }
}

impl Opened {
    pub(super) fn voice_store(&self) -> Option<Arc<VoiceStore>> {
        self.voice_store.clone()
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
        _fallback_lane: String,
        initial_history: Option<Vec<ChatItem>>,
    ) -> Self {
        Self { namespace, thread_id, lane, initial_history, opened: OnceCell::new() }
    }
    async fn open(&self) -> Result<(), StorageError> {
        self.open_ready().await.map(|_| ())
    }
    fn namespace(&self) -> Option<Arc<Namespace>> {
        self.opened().namespace.clone()
    }
    fn thread_id(&self) -> Option<String> {
        self.opened().thread_id.clone()
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
        if let Some(store) = &opened.voice_store
            && let Err(error) = store.aclose().await
        {
            log::warn!("the voice store could not close: {error}");
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
                    log::warn!(target: "livekit.agents", "the voice turn could not commit: {error}");
                }
                if lease_lost {
                    let _ = self.0.send(VoiceEvent::StorageLost(SpeechId(speech)));
                }
                VoiceEvent::StorageSettled(SpeechId(speech))
            }
            VoiceStorageEvent::Abandoned { speech, result } => {
                if let Err(error) = result {
                    log::warn!(target: "livekit.agents", "the voice turn could not abandon: {error}");
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
