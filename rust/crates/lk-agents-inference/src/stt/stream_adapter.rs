//! [`StreamAdapter`] (`stt/stream_adapter.py`): a recognize-only STT as a streaming one. The VAD
//! cuts the audio into utterances, each recognized on its own once it ends.

use std::sync::Arc;

use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::tokenize::language::LanguageCode;
use lk_agents_abi::{
    ApiConnectOptions, AudioFrame, Clock, Closed, ProviderId, SpeechEvent, SpeechEventType,
    StreamSource, Stt, SttCapabilities, SttSink, SttSource, Ts, Vad, VadEventType, VadSource,
};
use lk_agents_net::env::ProcessEnv;

/// `E` is the hosting process a host-built adapter reads its clock from.
pub struct StreamAdapter {
    stt: Arc<dyn Stt>,
    vad: Arc<dyn Vad>,
    /// What an utterance's `speech_end_time` is read off.
    clock: Arc<dyn Clock>,
}

impl Clone for StreamAdapter {
    fn clone(&self) -> Self {
        StreamAdapter::new(Arc::clone(&self.stt), Arc::clone(&self.vad), Arc::clone(&self.clock))
    }
}

impl StreamAdapter {
    fn configured(&self, conn_options: Option<ApiConnectOptions>) -> Self {
        let stt = conn_options
            .and_then(|conn| self.stt.with_conn_options(&conn))
            .unwrap_or_else(|| self.stt.clone());
        Self { stt, ..self.clone() }
    }

    pub fn new(stt: Arc<dyn Stt>, vad: Arc<dyn Vad>, clock: Arc<dyn Clock>) -> Self {
        StreamAdapter { stt, vad, clock }
    }
}

impl StreamAdapter {
    /// Itself when it streams; over `vad` when it does not and there is one (`Agent.stt_node`'s
    /// default); as it is otherwise, its stream refused when opened.
    pub fn adapted(
        stt: Arc<dyn Stt>,
        vad: Option<Arc<dyn Vad>>,
        clock: Arc<dyn Clock>,
    ) -> Arc<dyn Stt> {
        match vad {
            Some(vad) if !stt.capabilities().streaming => {
                Arc::new(StreamAdapter::new(stt, vad, clock))
            }
            _ => stt,
        }
    }
}

/// `_core.SttStreamAdapter`: the generated adapter calls this.
impl lk_agents_abi::stt::stream_adapter::SttStreamAdapter for StreamAdapter {
    type Audio = SttSink;
    type Events = SttSource;
    type Env = ProcessEnv;

    fn new(env: &ProcessEnv, stt: Arc<dyn Stt>, vad: Arc<dyn Vad>) -> Self {
        StreamAdapter::new(stt, vad, env.clock.clone())
    }

    async fn recognize(
        &self,
        frames: Vec<AudioFrame>,
        language: Option<LanguageCode>,
        conn_options: Option<ApiConnectOptions>,
    ) -> Result<SpeechEvent, ProviderError> {
        self.configured(conn_options).stt.recognize(&frames, language.as_ref()).await
    }

    async fn stream(
        &self,
        language: Option<LanguageCode>,
        start_time_offset: Option<f64>,
        conn_options: Option<ApiConnectOptions>,
    ) -> Result<(SttSink, SttSource), ProviderError> {
        Stt::stream(&self.configured(conn_options), language.as_ref(), start_time_offset)
            .await
            .map(|opened| opened.expect("the adapter's `stream` is always duplex"))
    }
}

impl Stt for StreamAdapter {
    fn label(&self) -> String {
        format!("StreamAdapter[{}]", self.stt.label())
    }

    /// The adapter is transparent to the metrics.
    fn id(&self) -> ProviderId {
        self.stt.id()
    }

    /// Streaming, no interim results, no diarization (it needs a streaming STT).
    fn capabilities(&self) -> SttCapabilities {
        let inner = self.stt.capabilities();
        SttCapabilities {
            streaming: true,
            interim_results: false,
            diarization: false,
            keyterms: inner.keyterms,
            chat_context: inner.chat_context,
            ..SttCapabilities::default()
        }
    }

    fn prewarm(&self) {
        self.stt.prewarm();
    }

    fn failed(&self, error: ProviderError, recoverable: bool) {
        self.stt.failed(error, recoverable);
    }

    fn update_keyterms(&self, keyterms: &[String]) {
        self.stt.update_keyterms(keyterms);
    }

    fn push_conversation_item(&self, role: &str, text: &str) {
        self.stt.push_conversation_item(role, text);
    }

    /// The options go to the wrapped STT's recognitions; the adapter's stream does not retry.
    fn with_conn_options(&self, conn: &ApiConnectOptions) -> Option<Arc<dyn Stt>> {
        Some(Arc::new(self.configured(Some(*conn))))
    }

    fn recognize(
        &self,
        buffer: &[AudioFrame],
        language: Option<&LanguageCode>,
    ) -> lk_agents_abi::BoxFuture<'_, Result<SpeechEvent, ProviderError>> {
        let buffer = buffer.to_vec();
        let language = language.cloned();
        Box::pin(async move { self.stt.recognize(&buffer, language.as_ref()).await })
    }

    fn stream(
        &self,
        language: Option<&LanguageCode>,
        _start_time_offset: Option<f64>,
    ) -> lk_agents_abi::BoxFuture<'_, Result<Option<(SttSink, SttSource)>, ProviderError>> {
        let language = language.cloned();
        Box::pin(async move {
            let opened = self.vad.stream().await.map(|(sink, vad)| {
                let source = AdapterSource {
                    vad,
                    stt: Arc::clone(&self.stt),
                    language: language.clone(),
                    utterance: None,
                    speech_end: None,
                    clock: Arc::clone(&self.clock),
                };
                (sink as SttSink, Box::new(source) as SttSource)
            });
            opened.map(Some)
        })
    }
}

/// The VAD's events as speech events; an ended utterance is recognized on the next read.
struct AdapterSource {
    vad: VadSource,
    stt: Arc<dyn Stt>,
    language: Option<LanguageCode>,
    /// The utterance the last end of speech closed, not yet recognized.
    utterance: Option<Vec<AudioFrame>>,
    /// When the last utterance ended, wall seconds: the VAD's report less its silence and its
    /// inference time, as the reference reads it.
    speech_end: Option<f64>,
    clock: Arc<dyn Clock>,
}

impl StreamSource<SpeechEvent> for AdapterSource {
    fn next(&mut self) -> lk_agents_abi::BoxFuture<'_, Result<Option<SpeechEvent>, ProviderError>> {
        Box::pin(async move {
            loop {
                if let Some(frames) = self.utterance.take() {
                    let event = self.stt.recognize(&frames, self.language.as_ref()).await?;
                    if let Some(mut transcript) = recognized(event) {
                        transcript.speech_end_time = transcript.speech_end_time.or(self.speech_end);
                        return Ok(Some(transcript));
                    }
                }
                let Some(event) = self.vad.next().await? else { return Ok(None) };
                match event.kind {
                    VadEventType::StartOfSpeech => {
                        return Ok(Some(SpeechEvent::new(SpeechEventType::StartOfSpeech)));
                    }
                    VadEventType::EndOfSpeech => {
                        let ended = lk_agents_abi::unix_secs(self.clock.wall())
                            - event.silence_duration
                            - event.inference_duration;
                        self.speech_end = Some(ended);
                        self.utterance = Some(event.frames);
                        let mut end = SpeechEvent::new(SpeechEventType::EndOfSpeech);
                        end.speech_end_time = Some(ended);
                        return Ok(Some(end));
                    }
                    VadEventType::InferenceDone => {}
                }
            }
        })
    }

    fn close(self: Box<Self>, deadline: Ts) -> lk_agents_abi::BoxFuture<'static, Closed> {
        Box::pin(async move { self.vad.close(deadline).await })
    }
}

/// The final transcript a recognition becomes (its best alternative), or `None` for an empty one.
fn recognized(event: SpeechEvent) -> Option<SpeechEvent> {
    let best = event.alternatives.into_iter().next()?;
    if best.text.is_empty() {
        return None;
    }
    let mut transcript = SpeechEvent::new(SpeechEventType::FinalTranscript);
    transcript.alternatives.push(best);
    transcript.speech_end_time = event.speech_end_time;
    Some(transcript)
}

#[cfg(test)]
mod tests {
    use super::*;
    use lk_agents_abi::{StreamSink, VadCapabilities, VadEvent};
    use std::collections::VecDeque;
    use std::sync::Mutex;

    struct Ignore;

    impl StreamSink<AudioFrame> for Ignore {
        fn push(&self, _: AudioFrame) -> Result<(), ProviderError> {
            Ok(())
        }
        fn flush(&self) -> Result<(), ProviderError> {
            Ok(())
        }
        fn end_input(&self) {}
    }

    struct Events(VecDeque<VadEventType>);

    impl StreamSource<VadEvent> for Events {
        fn next(
            &mut self,
        ) -> lk_agents_abi::BoxFuture<'_, Result<Option<VadEvent>, ProviderError>> {
            Box::pin(async move {
                Ok(self.0.pop_front().map(|kind| VadEvent::new(kind, 0, 0.0, 0.0, 0.0)))
            })
        }
        fn close(self: Box<Self>, _: Ts) -> lk_agents_abi::BoxFuture<'static, Closed> {
            Box::pin(async move { Closed::Clean })
        }
    }

    struct ScriptedVad(Vec<VadEventType>);

    impl Vad for ScriptedVad {
        fn label(&self) -> String {
            "vad".into()
        }
        fn capabilities(&self) -> VadCapabilities {
            VadCapabilities { update_interval: 0.032 }
        }
        fn stream(
            &self,
        ) -> lk_agents_abi::BoxFuture<'_, Result<(lk_agents_abi::VadSink, VadSource), ProviderError>>
        {
            let events = Events(self.0.iter().copied().collect());
            Box::pin(async move {
                Ok((Box::new(Ignore) as lk_agents_abi::VadSink, Box::new(events) as VadSource))
            })
        }
    }

    /// Recognizes each utterance as the next of its texts.
    struct BatchStt(Mutex<VecDeque<&'static str>>);

    impl Stt for BatchStt {
        fn label(&self) -> String {
            "batch".into()
        }
        fn capabilities(&self) -> SttCapabilities {
            SttCapabilities { keyterms: true, ..SttCapabilities::default() }
        }
        fn recognize(
            &self,
            _: &[AudioFrame],
            _: Option<&LanguageCode>,
        ) -> lk_agents_abi::BoxFuture<'_, Result<SpeechEvent, ProviderError>> {
            Box::pin(async move {
                let text = self.0.lock().unwrap().pop_front().unwrap_or_default();
                let mut ev = SpeechEvent::new(SpeechEventType::FinalTranscript);
                ev.alternatives.push(lk_agents_abi::SpeechData::new("en", text));
                Ok(ev)
            })
        }
        fn stream(
            &self,
            _: Option<&LanguageCode>,
            _: Option<f64>,
        ) -> lk_agents_abi::BoxFuture<'_, Result<Option<(SttSink, SttSource)>, ProviderError>>
        {
            Box::pin(async move { Ok(None) })
        }
    }

    // Catches: a transcript before its end of speech, an empty recognition reported, or the
    // VAD's inference ticks leaking out as events.
    #[tokio::test]
    async fn each_utterance_is_recognized_once_it_ends() {
        use VadEventType::{EndOfSpeech, InferenceDone, StartOfSpeech};
        let stt = Arc::new(BatchStt(Mutex::new(VecDeque::from(["hello", ""]))));
        let vad = Arc::new(ScriptedVad(vec![
            StartOfSpeech,
            InferenceDone,
            EndOfSpeech,
            StartOfSpeech,
            EndOfSpeech,
        ]));
        let adapter =
            StreamAdapter::adapted(stt, Some(vad), Arc::new(lk_agents_abi::SimClock::new()));
        let caps = adapter.capabilities();
        assert!(caps.streaming && !caps.interim_results && caps.keyterms);
        let (_sink, mut source) = adapter.stream(None, None).await.unwrap().unwrap();
        let mut seen = Vec::new();
        while let Some(ev) = source.next().await.unwrap() {
            seen.push((ev.kind, ev.best_alternative().map(|a| a.text.clone())));
        }
        use SpeechEventType as T;
        assert_eq!(
            seen,
            vec![
                (T::StartOfSpeech, None),
                (T::EndOfSpeech, None),
                (T::FinalTranscript, Some("hello".into())),
                (T::StartOfSpeech, None),
                (T::EndOfSpeech, None),
            ]
        );
    }
}
