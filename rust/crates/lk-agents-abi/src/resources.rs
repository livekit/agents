//! The providers a session uses, by id. The host registers each one when it hands it over and
//! puts the typed reference in its command: the core decides with the reference (its id and what
//! the provider said about itself at registration) and names the id in the effects that reach it.
//! The table resolves an id to the provider for whoever performs those effects.

use crate::llm::realtime::RealtimeCapabilities;
use crate::{
    Llm, ProviderId, RealtimeSession, Stt, SttCapabilities, Tts, TtsCapabilities, Vad,
    VadCapabilities,
};
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ResourceId(pub u64);

#[derive(Debug, Clone, PartialEq)]
pub struct SttRef {
    pub id: ResourceId,
    pub provider: ProviderId,
    pub capabilities: SttCapabilities,
    /// The object registered: the same recogniser handed back under a new id is not reopened.
    pub instance: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct VadRef {
    pub id: ResourceId,
    pub provider: ProviderId,
    pub capabilities: VadCapabilities,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LlmRef {
    pub id: ResourceId,
    pub provider: ProviderId,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TtsRef {
    pub id: ResourceId,
    pub provider: ProviderId,
    pub capabilities: TtsCapabilities,
    pub sample_rate: u32,
    pub num_channels: u16,
}

/// The session a realtime model opened: what it said it can do when it opened.
#[derive(Debug, Clone, PartialEq)]
pub struct RealtimeRef {
    pub id: ResourceId,
    pub capabilities: RealtimeCapabilities,
}

enum Resource {
    Stt(Arc<dyn Stt>),
    Vad(Arc<dyn Vad>),
    Llm(Arc<dyn Llm>),
    Tts(Arc<dyn Tts>),
    Realtime(Arc<dyn RealtimeSession>),
}

#[derive(Default)]
struct Table {
    next: u64,
    live: BTreeMap<ResourceId, Resource>,
}

/// Shared by the host side that registers and the side that performs, until the runtime owns it.
#[derive(Clone, Default)]
pub struct Resources(Arc<Mutex<Table>>);

impl Resources {
    fn insert(&self, resource: Resource) -> ResourceId {
        let mut table = self.0.lock().unwrap_or_else(|p| p.into_inner());
        table.next += 1;
        let id = ResourceId(table.next);
        table.live.insert(id, resource);
        id
    }

    fn get<T>(&self, id: ResourceId, pick: impl FnOnce(&Resource) -> Option<T>) -> Option<T> {
        let table = self.0.lock().unwrap_or_else(|p| p.into_inner());
        table.live.get(&id).and_then(pick)
    }

    pub fn register_stt(&self, stt: Arc<dyn Stt>) -> SttRef {
        let (provider, capabilities) = (stt.id(), stt.capabilities());
        let instance = Arc::as_ptr(&stt).cast::<()>() as usize;
        SttRef { id: self.insert(Resource::Stt(stt)), provider, capabilities, instance }
    }

    pub fn register_vad(&self, vad: Arc<dyn Vad>) -> VadRef {
        let (provider, capabilities) = (vad.id(), vad.capabilities());
        VadRef { id: self.insert(Resource::Vad(vad)), provider, capabilities }
    }

    pub fn register_llm(&self, llm: Arc<dyn Llm>) -> LlmRef {
        let provider = llm.id();
        LlmRef { id: self.insert(Resource::Llm(llm)), provider }
    }

    pub fn register_tts(&self, tts: Arc<dyn Tts>) -> TtsRef {
        let (provider, capabilities) = (tts.id(), tts.capabilities());
        let (sample_rate, num_channels) = (tts.sample_rate(), tts.num_channels());
        TtsRef {
            id: self.insert(Resource::Tts(tts)),
            provider,
            capabilities,
            sample_rate,
            num_channels,
        }
    }

    pub fn register_realtime(&self, session: Arc<dyn RealtimeSession>) -> RealtimeRef {
        let capabilities = session.capabilities();
        RealtimeRef { id: self.insert(Resource::Realtime(session)), capabilities }
    }

    pub fn realtime(&self, id: ResourceId) -> Option<Arc<dyn RealtimeSession>> {
        self.get(id, |r| if let Resource::Realtime(s) = r { Some(Arc::clone(s)) } else { None })
    }

    pub fn stt(&self, id: ResourceId) -> Option<Arc<dyn Stt>> {
        self.get(id, |r| if let Resource::Stt(s) = r { Some(Arc::clone(s)) } else { None })
    }

    pub fn vad(&self, id: ResourceId) -> Option<Arc<dyn Vad>> {
        self.get(id, |r| if let Resource::Vad(v) = r { Some(Arc::clone(v)) } else { None })
    }

    pub fn llm(&self, id: ResourceId) -> Option<Arc<dyn Llm>> {
        self.get(id, |r| if let Resource::Llm(l) = r { Some(Arc::clone(l)) } else { None })
    }

    pub fn tts(&self, id: ResourceId) -> Option<Arc<dyn Tts>> {
        self.get(id, |r| if let Resource::Tts(t) = r { Some(Arc::clone(t)) } else { None })
    }

    pub fn prewarm(&self, id: ResourceId) {
        let table = self.0.lock().unwrap_or_else(|p| p.into_inner());
        match table.live.get(&id) {
            Some(Resource::Stt(s)) => s.prewarm(),
            Some(Resource::Vad(v)) => v.prewarm(),
            Some(Resource::Llm(l)) => l.prewarm(),
            Some(Resource::Tts(t)) => t.prewarm(),
            Some(Resource::Realtime(_)) | None => {}
        }
    }

    /// Whoever holds a clone (a stream or a reply in flight) keeps the provider alive.
    pub fn release(&self, id: ResourceId) {
        let mut table = self.0.lock().unwrap_or_else(|p| p.into_inner());
        table.live.remove(&id);
    }
}
