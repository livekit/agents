//! The session's provider table: [`ResourceId`]s the core names in its effects, resolved to the
//! providers the host registered, for whoever performs those effects.

use lk_agents_abi::resources::{
    AudioOutputRef, LlmRef, RealtimeRef, ResourceId, SttRef, TextOutputRef, TtsRef, VadRef,
    VideoInputRef,
};
use lk_agents_abi::voice::io::{
    AudioInput, AudioOutput, AudioOutputCapabilities, AudioOutputExt, TextOutput, VideoInput,
};
use lk_agents_abi::{AudioFrame, Llm, ProviderId, RealtimeSession, StreamInput, Stt, Tts, Vad};
use parking_lot::Mutex;
use std::collections::BTreeMap;
use std::sync::Arc;

#[derive(Clone)]
enum Resource {
    Recorder(Arc<dyn lk_agents_abi::recorder::AudioRecorder>),
    Instructions(Arc<dyn lk_agents_abi::agent::HostInstructions>),
    TranscriptionNode(Arc<dyn lk_agents_abi::voice::nodes::NodeFactory>),
    Stt(Arc<dyn Stt>),
    Vad(Arc<dyn Vad>),
    Llm(Arc<dyn Llm>),
    Tts(Arc<dyn Tts>),
    Realtime(Arc<dyn RealtimeSession>),
    AudioOutput(Arc<dyn AudioOutput>),
    TextOutput(Arc<dyn TextOutput>),
    AudioInput(Arc<dyn AudioInput>),
    VideoInput(Arc<dyn VideoInput>),
    AudioSink(Arc<dyn StreamInput<AudioFrame>>),
}

#[derive(Default)]
struct Table {
    next: u64,
    live: BTreeMap<ResourceId, Resource>,
}

/// Shared by the host side that registers and the side that performs.
#[derive(Clone, Default)]
pub struct Resources(Arc<Mutex<Table>>);

impl Resources {
    pub fn register_recorder(
        &self,
        recorder: Arc<dyn lk_agents_abi::recorder::AudioRecorder>,
    ) -> ResourceId {
        self.insert(Resource::Recorder(recorder))
    }
    pub fn recorder(
        &self,
        id: ResourceId,
    ) -> Option<Arc<dyn lk_agents_abi::recorder::AudioRecorder>> {
        self.get(id, |resource| match resource {
            Resource::Recorder(value) => Some(value.clone()),
            _ => None,
        })
    }

    pub fn register_instructions(
        &self,
        callback: Arc<dyn lk_agents_abi::agent::HostInstructions>,
    ) -> ResourceId {
        self.insert(Resource::Instructions(callback))
    }
    pub fn instructions(
        &self,
        id: ResourceId,
    ) -> Option<Arc<dyn lk_agents_abi::agent::HostInstructions>> {
        self.get(id, |resource| match resource {
            Resource::Instructions(value) => Some(value.clone()),
            _ => None,
        })
    }
    pub fn register_transcription_node(
        &self,
        node: Arc<dyn lk_agents_abi::voice::nodes::NodeFactory>,
    ) -> ResourceId {
        self.insert(Resource::TranscriptionNode(node))
    }
    pub fn transcription_node(
        &self,
        id: ResourceId,
    ) -> Option<Arc<dyn lk_agents_abi::voice::nodes::NodeFactory>> {
        self.get(id, |resource| match resource {
            Resource::TranscriptionNode(value) => Some(value.clone()),
            _ => None,
        })
    }

    fn insert(&self, resource: Resource) -> ResourceId {
        let mut table = self.0.lock();
        table.next += 1;
        let id = ResourceId(table.next);
        table.live.insert(id, resource);
        id
    }

    fn get<T>(&self, id: ResourceId, pick: impl FnOnce(&Resource) -> Option<T>) -> Option<T> {
        let table = self.0.lock();
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
        let markup = lk_agents_abi::tts::markup::Markup::from_provider_key(&tts.markup_provider());
        let lowers_markup = tts.lowers_markup();
        TtsRef {
            id: self.insert(Resource::Tts(tts)),
            provider,
            capabilities,
            sample_rate,
            num_channels,
            markup,
            lowers_markup,
        }
    }

    pub fn register_realtime(
        &self,
        session: Arc<dyn RealtimeSession>,
        provider: ProviderId,
    ) -> RealtimeRef {
        let capabilities = session.capabilities();
        RealtimeRef { id: self.insert(Resource::Realtime(session)), provider, capabilities }
    }

    pub fn realtime(&self, id: ResourceId) -> Option<Arc<dyn RealtimeSession>> {
        self.get(id, |r| if let Resource::Realtime(s) = r { Some(Arc::clone(s)) } else { None })
    }

    pub fn register_audio_output(&self, out: Arc<dyn AudioOutput>) -> AudioOutputRef {
        let label = out.label().to_owned();
        let capabilities = AudioOutputCapabilities { pause: out.can_pause() };
        AudioOutputRef { id: self.insert(Resource::AudioOutput(out)), label, capabilities }
    }

    pub fn register_text_output(&self, out: Arc<dyn TextOutput>) -> TextOutputRef {
        let label = out.label().to_owned();
        TextOutputRef { id: self.insert(Resource::TextOutput(out)), label }
    }

    pub fn audio_output(&self, id: ResourceId) -> Option<Arc<dyn AudioOutput>> {
        self.get(id, |r| if let Resource::AudioOutput(o) = r { Some(Arc::clone(o)) } else { None })
    }

    pub fn text_output(&self, id: ResourceId) -> Option<Arc<dyn TextOutput>> {
        self.get(id, |r| if let Resource::TextOutput(o) = r { Some(Arc::clone(o)) } else { None })
    }

    pub fn register_audio_input(&self, input: Arc<dyn AudioInput>) -> ResourceId {
        self.insert(Resource::AudioInput(input))
    }

    pub fn audio_input(&self, id: ResourceId) -> Option<Arc<dyn AudioInput>> {
        self.get(id, |r| if let Resource::AudioInput(i) = r { Some(Arc::clone(i)) } else { None })
    }

    pub fn register_video_input(&self, input: Arc<dyn VideoInput>) -> VideoInputRef {
        let instance = Arc::as_ptr(&input).cast::<()>() as usize;
        VideoInputRef { id: self.insert(Resource::VideoInput(input)), instance }
    }

    pub fn video_input(&self, id: ResourceId) -> Option<Arc<dyn VideoInput>> {
        self.get(id, |r| if let Resource::VideoInput(i) = r { Some(Arc::clone(i)) } else { None })
    }

    /// A stream's input side the microphone feeds: an STT, VAD or detector stream.
    pub fn register_audio_sink(&self, sink: Arc<dyn StreamInput<AudioFrame>>) -> ResourceId {
        self.insert(Resource::AudioSink(sink))
    }

    pub fn audio_sink(&self, id: ResourceId) -> Option<Arc<dyn StreamInput<AudioFrame>>> {
        self.get(id, |r| if let Resource::AudioSink(s) = r { Some(Arc::clone(s)) } else { None })
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

    /// The provider is called after the table lock is released: a host provider's `prewarm` is
    /// a host call.
    pub fn prewarm(&self, id: ResourceId) {
        let resource = self.0.lock().live.get(&id).cloned();
        match resource {
            Some(Resource::Stt(s)) => s.prewarm(),
            Some(Resource::Vad(v)) => v.prewarm(),
            Some(Resource::Llm(l)) => l.prewarm(),
            Some(Resource::Tts(t)) => t.prewarm(),
            Some(
                Resource::Recorder(_)
                | Resource::Instructions(_)
                | Resource::TranscriptionNode(_)
                | Resource::Realtime(_)
                | Resource::AudioOutput(_)
                | Resource::TextOutput(_)
                | Resource::AudioInput(_)
                | Resource::VideoInput(_)
                | Resource::AudioSink(_),
            )
            | None => {}
        }
    }

    /// What the table still holds, by id and kind: nothing once the session has ended.
    pub fn live(&self) -> Vec<(ResourceId, &'static str)> {
        let table = self.0.lock();
        let kind = |r: &Resource| match r {
            Resource::Recorder(_) => "recorder",
            Resource::Instructions(_) => "instructions",
            Resource::TranscriptionNode(_) => "transcription node",
            Resource::Stt(_) => "stt",
            Resource::Vad(_) => "vad",
            Resource::Llm(_) => "llm",
            Resource::Tts(_) => "tts",
            Resource::Realtime(_) => "realtime",
            Resource::AudioOutput(_) => "audio output",
            Resource::TextOutput(_) => "text output",
            Resource::AudioInput(_) => "audio input",
            Resource::VideoInput(_) => "video input",
            Resource::AudioSink(_) => "audio sink",
        };
        table.live.iter().map(|(id, r)| (*id, kind(r))).collect()
    }

    /// The session also owns registrations never submitted to the core (for example a prepared
    /// handoff that was refused). Finalization retires those after owned work settles.
    pub(crate) fn clear(&self) {
        let retired = std::mem::take(&mut self.0.lock().live);
        drop(retired);
    }

    /// Whoever holds a clone (a stream or a reply in flight) keeps the provider alive.
    pub fn release(&self, id: ResourceId) {
        let retired = self.0.lock().live.remove(&id);
        drop(retired);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};

    struct Callback {
        resources: Resources,
        dropped: Arc<AtomicBool>,
    }
    impl Drop for Callback {
        fn drop(&mut self) {
            assert!(
                self.resources.0.try_lock().is_some(),
                "host object dropped under registry lock"
            );
            self.dropped.store(true, Ordering::SeqCst);
        }
    }
    impl lk_agents_abi::agent::HostInstructions for Callback {
        fn render<'a>(
            &'a self,
            _: Arc<dyn lk_agents_abi::voice::speech_handle::Speech>,
        ) -> lk_agents_abi::BoxFuture<'a, Result<String, lk_agents_abi::errors::HostError>>
        {
            Box::pin(async { Ok(String::new()) })
        }
    }

    #[test]
    fn callback_snapshots_survive_retirement_and_drop_outside_the_registry_lock() {
        let resources = Resources::default();
        let dropped = Arc::new(AtomicBool::new(false));
        let id = resources.register_instructions(Arc::new(Callback {
            resources: resources.clone(),
            dropped: dropped.clone(),
        }));
        let snapshot = resources.instructions(id).unwrap();
        resources.release(id);
        assert!(resources.live().is_empty());
        assert!(!dropped.load(Ordering::SeqCst));
        drop(snapshot);
        assert!(dropped.load(Ordering::SeqCst));

        dropped.store(false, Ordering::SeqCst);
        resources.register_instructions(Arc::new(Callback {
            resources: resources.clone(),
            dropped: dropped.clone(),
        }));
        resources.clear();
        assert!(
            dropped.load(Ordering::SeqCst),
            "unsubmitted registration outlived session cleanup"
        );
    }
}
