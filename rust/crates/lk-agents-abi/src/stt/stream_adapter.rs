//! [`SttStreamAdapter`] (`stt/stream_adapter.py`): a recognize-only STT as a streaming one. The VAD
//! cuts the audio into utterances, each recognized on its own once it ends.

use std::sync::Arc;

use async_trait::async_trait;

use super::events::{SpeechEvent, SpeechEventType};
use super::stt::{Stt, SttCapabilities, SttSink, SttSource};
use crate::errors::ProviderError;
use crate::tokenize::language::LanguageCode;
use crate::types::identity::ProviderId;
use crate::utils::clock::{Clock, Ts};
use crate::utils::stream::{Closed, StreamSource};
use crate::vad::events::VadEventType;
use crate::vad::vad::{Vad, VadSource};
use crate::{ApiConnectOptions, AudioFrame};

pub struct SttStreamAdapter {
    stt: Arc<dyn Stt>,
    vad: Arc<dyn Vad>,
    /// What an utterance's `speech_end_time` is read off.
    clock: Arc<dyn Clock>,
}

impl SttStreamAdapter {
    pub fn new(stt: Arc<dyn Stt>, vad: Arc<dyn Vad>, clock: Arc<dyn Clock>) -> Self {
        SttStreamAdapter { stt, vad, clock }
    }

    /// Itself when it streams; over `vad` when it does not and there is one (`Agent.stt_node`'s
    /// default); as it is otherwise, its stream refused when opened.
    pub fn adapted(
        stt: Arc<dyn Stt>,
        vad: Option<Arc<dyn Vad>>,
        clock: Arc<dyn Clock>,
    ) -> Arc<dyn Stt> {
        match vad {
            Some(vad) if !stt.capabilities().streaming => {
                Arc::new(SttStreamAdapter::new(stt, vad, clock))
            }
            _ => stt,
        }
    }
}

/// `stt.StreamAdapter`'s engine, as the host builds it; the public class stays Python.
#[crate::host(object(name = "SttStreamAdapter", cell = "arc", as_dyn(Stt = as_stt)))]
impl SttStreamAdapter {
    /// The wrapped STT's recognitions run under the default `conn_options`, as the reference's.
    #[host(new)]
    pub fn for_host(
        stt: Arc<dyn Stt>,
        vad: Arc<dyn Vad>,
        #[host(ambient)] clock: Arc<dyn Clock>,
    ) -> Self {
        let stt = stt.with_conn_options(&ApiConnectOptions::default()).unwrap_or(stt);
        SttStreamAdapter::new(stt, vad, clock)
    }

    /// The adapter as the provider a session holds (not proxied).
    pub fn as_stt(self: &Arc<Self>) -> Arc<dyn Stt> {
        Arc::clone(self) as Arc<dyn Stt>
    }

    /// `_recognize_impl`: the wrapped STT's batch path.
    #[host(method(name = "recognize"))]
    pub async fn recognize_for_host(
        &self,
        frames: Vec<AudioFrame>,
        language: Option<LanguageCode>,
    ) -> Result<SpeechEvent, ProviderError> {
        self.stt.recognize(&frames, language.as_ref()).await
    }

    /// The duplex path: audio into the VAD, its utterances recognized.
    #[host(method(name = "stream", stream = crate::FallbackSttStream))]
    pub async fn stream_for_host(
        &self,
        language: Option<LanguageCode>,
        start_time_offset: Option<f64>,
    ) -> Result<(SttSink, SttSource), ProviderError> {
        Stt::stream(self, language.as_ref(), start_time_offset)
            .await
            .map(|opened| opened.expect("the adapter's `stream` is always duplex"))
    }
}

#[async_trait]
impl Stt for SttStreamAdapter {
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
        let stt = self.stt.with_conn_options(conn).unwrap_or_else(|| Arc::clone(&self.stt));
        Some(Arc::new(SttStreamAdapter::new(stt, Arc::clone(&self.vad), Arc::clone(&self.clock))))
    }

    async fn recognize(
        &self,
        buffer: &[AudioFrame],
        language: Option<&LanguageCode>,
    ) -> Result<SpeechEvent, ProviderError> {
        self.stt.recognize(buffer, language).await
    }

    async fn stream(
        &self,
        language: Option<&LanguageCode>,
        _start_time_offset: Option<f64>,
    ) -> Result<Option<(SttSink, SttSource)>, ProviderError> {
        let opened = self.vad.stream().await.map(|(sink, vad)| {
            let source = AdapterSource {
                vad,
                stt: Arc::clone(&self.stt),
                language: language.cloned(),
                utterance: None,
                speech_end: None,
                clock: Arc::clone(&self.clock),
            };
            (sink as SttSink, Box::new(source) as SttSource)
        });
        opened.map(Some)
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

#[async_trait]
impl StreamSource<SpeechEvent> for AdapterSource {
    async fn next(&mut self) -> Result<Option<SpeechEvent>, ProviderError> {
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
                    let ended = crate::unix_secs(self.clock.wall())
                        - event.silence_duration
                        - event.inference_duration;
                    self.speech_end = Some(ended);
                    self.utterance =
                        Some(event.frames.map(|f| f.audio_frames()).transpose()?.unwrap_or_default());
                    let mut end = SpeechEvent::new(SpeechEventType::EndOfSpeech);
                    end.speech_end_time = Some(ended);
                    return Ok(Some(end));
                }
                VadEventType::InferenceDone => {}
            }
        }
    }

    async fn close(self: Box<Self>, deadline: Ts) -> Closed {
        self.vad.close(deadline).await
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
    use crate::VadEvent;
    use crate::utils::stream::StreamSink;
    use crate::vad::vad::VadCapabilities;
    use std::collections::VecDeque;
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
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

    #[async_trait]
    impl StreamSource<VadEvent> for Events {
        async fn next(&mut self) -> Result<Option<VadEvent>, ProviderError> {
            Ok(self.0.pop_front().map(|kind| VadEvent::new(kind, 0, 0.0, 0.0, 0.0)))
        }
        async fn close(self: Box<Self>, _: Ts) -> Closed {
            Closed::Clean
        }
    }

    struct ScriptedVad(Vec<VadEventType>);

    #[async_trait]
    impl Vad for ScriptedVad {
        fn label(&self) -> String {
            "vad".into()
        }
        fn capabilities(&self) -> VadCapabilities {
            VadCapabilities { update_interval: 0.032 }
        }
        async fn stream(&self) -> Result<(crate::vad::vad::VadSink, VadSource), ProviderError> {
            Ok((Box::new(Ignore), Box::new(Events(self.0.iter().copied().collect()))))
        }
    }

    /// Recognizes each utterance as the next of its texts.
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    struct BatchStt(Mutex<VecDeque<&'static str>>);

    #[async_trait]
    impl Stt for BatchStt {
        fn label(&self) -> String {
            "batch".into()
        }
        fn capabilities(&self) -> SttCapabilities {
            SttCapabilities { keyterms: true, ..SttCapabilities::default() }
        }
        async fn recognize(
            &self,
            _: &[AudioFrame],
            _: Option<&LanguageCode>,
        ) -> Result<SpeechEvent, ProviderError> {
            let text = self.0.lock().unwrap().pop_front().unwrap_or_default();
            let mut ev = SpeechEvent::new(SpeechEventType::FinalTranscript);
            ev.alternatives.push(crate::SpeechData::new("en", text));
            Ok(ev)
        }
        async fn stream(
            &self,
            _: Option<&LanguageCode>,
            _: Option<f64>,
        ) -> Result<Option<(SttSink, SttSource)>, ProviderError> {
            Ok(None)
        }
    }

    // Catches: a transcript before its end of speech, an empty recognition reported, or the
    // VAD's inference ticks leaking out as events.
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
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
        let adapter = SttStreamAdapter::adapted(stt, Some(vad), Arc::new(crate::utils::clock::SimClock::new()));
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
