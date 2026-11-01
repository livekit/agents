//! Storage startup effects; lane versus ephemeral fallback is selected by voice policy.

use lk_agents_abi::storage::Storage as _;
use lk_agents_abi::storage::context::ContextActivation as _;
use lk_agents_abi::storage::voice::VoiceStore as _;
use lk_agents_abi::{ChatItem, storage::StorageError};
use lk_agents_net::env::ProcessEnv;
use lk_agents_storage::{context::store::ContextStore, namespace::Namespace, voice::VoiceStore};
use std::sync::Arc;
use tokio::sync::OnceCell;

pub struct SessionStorage {
    env: ProcessEnv,
    namespace: Option<Arc<Namespace>>,
    thread_id: Option<String>,
    lane: String,
    fallback_lane: String,
    initial_history: Option<Vec<ChatItem>>,
    opened: OnceCell<Opened>,
}

struct Opened {
    namespace: Arc<Namespace>,
    thread_id: String,
    scope_id: String,
    activation: ContextStore,
    voice_store: Option<Arc<VoiceStore>>,
    history: Option<Vec<ChatItem>>,
    ephemeral: bool,
}

impl SessionStorage {
    fn opened(&self) -> &Opened {
        self.opened.get().expect("session storage must be opened before reading it")
    }

    async fn load(&self) -> Result<Opened, StorageError> {
        use lk_agents_voice::session_start::{StorageSource, StorageStart};
        let plan = StorageStart::new(self.namespace.is_some() && self.thread_id.is_some());
        let loaded = if plan.load_lane() {
            let namespace = self.namespace.as_ref().unwrap();
            match namespace
                ._core_voice_store(
                    self.thread_id.clone().unwrap(),
                    self.lane.clone(),
                    self.initial_history.clone(),
                )
                .await
            {
                Ok(store) => Some(store),
                Err(error) => {
                    log::warn!("the voice lane could not be loaded: {error}");
                    None
                }
            }
        } else {
            None
        };
        match plan.after_load(loaded.is_some()) {
            StorageSource::Lane => {
                let store = loaded.expect("voice selected the loaded lane");
                return Ok(Opened {
                    namespace: self.namespace.clone().unwrap(),
                    thread_id: self.thread_id.clone().unwrap(),
                    scope_id: format!("root:{}", self.lane),
                    activation: store.activation(),
                    history: store.initial_history(),
                    voice_store: Some(Arc::new(store)),
                    ephemeral: false,
                });
            }
            StorageSource::Ephemeral => {}
        }
        let namespace = Arc::new(Namespace::new(&self.env, None, None, None, None, None, None)?);
        let info = namespace.create_thread(None, None, 0).await?;
        let activation = namespace
            ._core_context_store(info.thread_id.clone(), self.fallback_lane.clone())
            .await?;
        Ok(Opened {
            namespace,
            thread_id: info.thread_id,
            scope_id: "root".to_owned(),
            activation,
            voice_store: None,
            history: None,
            ephemeral: true,
        })
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
        fallback_lane: String,
        initial_history: Option<Vec<ChatItem>>,
    ) -> Self {
        Self {
            env: env.clone(),
            namespace,
            thread_id,
            lane,
            fallback_lane,
            initial_history,
            opened: OnceCell::new(),
        }
    }
    async fn open(&self) -> Result<(), StorageError> {
        self.opened.get_or_try_init(|| self.load()).await.map(|_| ())
    }
    fn namespace(&self) -> Arc<Namespace> {
        self.opened().namespace.clone()
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
    fn voice_store(&self) -> Option<Arc<VoiceStore>> {
        self.opened().voice_store.clone()
    }
    fn history(&self) -> Option<Vec<ChatItem>> {
        self.opened().history.clone()
    }
    async fn aclose(&self) {
        let Some(opened) = self.opened.get() else { return };
        if let Some(store) = &opened.voice_store
            && let Err(error) = store.aclose().await
        {
            log::warn!("the voice store could not close: {error}");
        }
        opened.activation.close();
        if opened.ephemeral {
            opened.namespace.aclose().await;
        }
    }
}
