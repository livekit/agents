//! The fake models behind a real AMD run: a classifier LLM and a decision model whose
//! categories follow a callee persona, and AMD's own STT, which hears what the session's STT
//! finalized after a lag. The production executor, machine and legs run over them unchanged.
//!
//! Each fake draws its lags and failures from its own seeded stream, and logs every call it
//! answered in [`AmdModelLog`] for rules and stories to read.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use fastrand::Rng;
use lk_agents_abi::errors::{ApiError, ErrorKind, ProviderError};
use lk_agents_abi::utils::stream::{Closed, ProviderStream, StreamInput, StreamOutput};
use lk_agents_abi::{
    ApiConnectOptions, AudioFrame, BoxFuture, ChatRequest, ChoiceDelta, FunctionToolCall,
    LanguageCode, Llm, LlmChunk, LlmStream, SpeechData, SpeechEvent, SpeechEventType, Stt,
    SttCapabilities, SttSink, SttSource, Ts,
};
use lk_agents_runtime::voice::amd::classifier::RECORD_RESULT;
use lk_agents_runtime::voice::amd::prompts::MENU_PROMPT;
use lk_agents_runtime::voice::amd::{Choice, DecisionModel};
use lk_agents_voice::amd::events::Category;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::input::AmdPersona;

/// How often a classifier call fails at the provider: an `inference_error` prediction.
const REFUSED: u8 = 4;
/// How often a classifier answer comes after the default `inference_timeout` (2 s).
const SLOW: u8 = 8;

impl AmdPersona {
    /// The category the classifier answers for its `n`th classification, from 0: screener ->
    /// human -> human; voicemail -> message -> menu; hold -> `wait` until a person picks up.
    pub fn category(self, n: usize) -> Category {
        let script: &[Category] = match self {
            AmdPersona::Human => &[Category::Human],
            AmdPersona::Screener => &[Category::MachineScreening, Category::Human, Category::Human],
            AmdPersona::Voicemail => {
                &[Category::MachineVm, Category::MachineVm, Category::MachineIvr]
            }
            AmdPersona::Hold => &[Category::Wait, Category::Wait, Category::Human],
            AmdPersona::Unavailable => &[Category::MachineUnavailable],
        };
        script[n.min(script.len() - 1)]
    }
}

/// One call a fake AMD model answered.
#[derive(Clone, Debug, PartialEq)]
pub enum AmdModelCall {
    /// A classification: the category answered, `None` when the call was refused, and the
    /// request's last user message.
    Classify { category: Option<Category>, lag: Duration, last_user: String },
    /// A decision model's `category` choice.
    Choice { category: Option<Category>, options: usize, lag: Duration },
    /// A menu extraction.
    Menu,
}

/// Every call the run's fake models answered, in order.
pub type AmdModelLog = Arc<Mutex<Vec<AmdModelCall>>>;

/// The lags and failures one fake draws.
struct Draws {
    rng: Rng,
    classified: usize,
}

impl Draws {
    fn new(seed: u64) -> Mutex<Self> {
        Mutex::new(Draws { rng: Rng::with_seed(seed ^ 0x414d_445f_6d6f_6465), classified: 0 })
    }

    /// The next classification's category (or a refusal) and its lag.
    fn next(&mut self, persona: AmdPersona) -> (Option<Category>, Duration) {
        let category = persona.category(self.classified);
        self.classified += 1;
        let refused = self.rng.u8(0..100) < REFUSED;
        let lag = if self.rng.u8(0..100) < SLOW {
            self.rng.u64(2_000..3_000)
        } else {
            self.rng.u64(50..1_500)
        };
        ((!refused).then_some(category), Duration::from_millis(lag))
    }
}

/// The classifier LLM: one `record_result` call per request, the persona's next category.
pub struct AmdClassifier {
    persona: AmdPersona,
    draws: Mutex<Draws>,
    log: AmdModelLog,
}

impl AmdClassifier {
    pub fn new(persona: AmdPersona, seed: u64, log: AmdModelLog) -> Arc<Self> {
        Arc::new(AmdClassifier { persona, draws: Draws::new(seed), log })
    }
}

/// A one-call answer, after its lag.
struct Answer {
    lag: Option<Duration>,
    call: Option<FunctionToolCall>,
}

impl StreamOutput<LlmChunk> for Answer {
    fn next(&mut self) -> BoxFuture<'_, Result<Option<LlmChunk>, ProviderError>> {
        Box::pin(async move {
            if let Some(lag) = self.lag.take() {
                tokio::time::sleep(lag).await;
            }
            Ok(self.call.take().map(|call| LlmChunk {
                id: "amd".into(),
                delta: Some(ChoiceDelta { tool_calls: vec![call], ..Default::default() }),
                ..Default::default()
            }))
        })
    }

    fn close(self: Box<Self>, _: Ts) -> BoxFuture<'static, Closed> {
        Box::pin(async { Closed::Clean })
    }
}

impl ProviderStream<LlmChunk> for Answer {}

fn refused() -> ProviderError {
    ProviderError::Local(ApiError::new(ErrorKind::Server, "the fake AMD model refused"))
}

impl Llm for AmdClassifier {
    fn label(&self) -> String {
        "fake::amd_classifier".into()
    }

    fn chat<'a>(
        &'a self,
        _cx: &lk_agents_abi::telemetry::Cx,
        request: &'a ChatRequest,
    ) -> BoxFuture<'a, Result<LlmStream, ProviderError>> {
        let menu = request.chat_ctx.items.first().and_then(|item| item.as_message()).is_some_and(
            |m| matches!(m.content.first(), Some(lk_agents_abi::ChatContent::Text(t)) if t == MENU_PROMPT),
        );
        let arguments = if menu {
            self.log.lock().unwrap().push(AmdModelCall::Menu);
            Some((
                r#"{"menu": "main", "options": [{"label": "sales", "dtmf": "1"}]}"#.to_owned(),
                Duration::from_millis(200),
            ))
        } else {
            let (category, lag) = self.draws.lock().unwrap().next(self.persona);
            let last_user = (request.chat_ctx.items.iter().rev())
                .filter_map(|item| item.as_message())
                .find(|m| m.role == lk_agents_abi::Role::User)
                .and_then(|m| m.raw_text_content())
                .unwrap_or_default();
            self.log.lock().unwrap().push(AmdModelCall::Classify { category, lag, last_user });
            category.map(|c| (format!(r#"{{"category": "{}"}}"#, c.as_str()), lag))
        };
        Box::pin(async move {
            let Some((arguments, lag)) = arguments else { return Err(refused()) };
            let call = FunctionToolCall {
                name: RECORD_RESULT.into(),
                arguments,
                call_id: "amd_call".into(),
                extra: None,
            };
            Ok(Box::new(Answer { lag: Some(lag), call: Some(call) }) as LlmStream)
        })
    }
}

/// The decision model: the persona's next category as its `category` choice.
pub struct AmdDecision {
    persona: AmdPersona,
    draws: Mutex<Draws>,
    log: AmdModelLog,
}

impl AmdDecision {
    pub fn new(persona: AmdPersona, seed: u64, log: AmdModelLog) -> Arc<Self> {
        Arc::new(AmdDecision { persona, draws: Draws::new(seed), log })
    }
}

impl DecisionModel for AmdDecision {
    fn label(&self) -> String {
        "fake::amd_decision".into()
    }

    fn choose<'a>(
        &'a self,
        _cx: &lk_agents_abi::telemetry::Cx,
        choice: Choice,
    ) -> BoxFuture<'a, Result<Option<String>, ProviderError>> {
        let (category, lag) = self.draws.lock().unwrap().next(self.persona);
        let options = choice.options.len();
        self.log.lock().unwrap().push(AmdModelCall::Choice { category, options, lag });
        Box::pin(async move {
            tokio::time::sleep(lag).await;
            match category {
                Some(category) => Ok(Some(category.as_str().to_owned())),
                None => Err(refused()),
            }
        })
    }
}

/// What AMD's STT hears: each session final, to every open AMD stream, after its own lag.
pub struct AmdEar {
    streams: Mutex<Vec<mpsc::UnboundedSender<(Duration, String)>>>,
    rng: Mutex<Rng>,
}

impl AmdEar {
    pub fn new(seed: u64) -> Arc<Self> {
        Arc::new(AmdEar {
            streams: Mutex::default(),
            rng: Mutex::new(Rng::with_seed(seed ^ 0x0041_4d44_5f65_6172)),
        })
    }

    /// The session's STT finalized `text`: AMD's stream finalizes it too, a little later.
    pub fn hear(&self, text: &str) {
        let lag = Duration::from_millis(self.rng.lock().unwrap().u64(0..600));
        self.streams.lock().unwrap().retain(|stream| stream.send((lag, text.to_owned())).is_ok());
    }
}

/// AMD's own STT. With `fails`, its stream will not open.
pub struct AmdStt {
    ear: Arc<AmdEar>,
    fails: bool,
}

impl AmdStt {
    pub fn new(ear: Arc<AmdEar>, fails: bool) -> Arc<Self> {
        Arc::new(AmdStt { ear, fails })
    }
}

/// Ignores the audio: it hears through [`AmdEar`]. Its end ends the stream.
struct Ears {
    ended: CancellationToken,
}

impl StreamInput<AudioFrame> for Ears {
    fn push(&self, _: AudioFrame) -> Result<(), ProviderError> {
        Ok(())
    }

    fn flush(&self) -> Result<(), ProviderError> {
        Ok(())
    }

    fn end_input(&self) {
        self.ended.cancel();
    }
}

struct Finals {
    heard: mpsc::UnboundedReceiver<(Duration, String)>,
    ended: CancellationToken,
}

impl StreamOutput<SpeechEvent> for Finals {
    fn next(&mut self) -> BoxFuture<'_, Result<Option<SpeechEvent>, ProviderError>> {
        Box::pin(async move {
            let heard = tokio::select! {
                biased;
                heard = self.heard.recv() => heard,
                () = self.ended.cancelled() => None,
            };
            let Some((lag, text)) = heard else { return Ok(None) };
            tokio::time::sleep(lag).await;
            let mut event = SpeechEvent::new(SpeechEventType::FinalTranscript);
            event.alternatives = vec![SpeechData::new("en", text)];
            Ok(Some(event))
        })
    }

    fn close(self: Box<Self>, _: Ts) -> BoxFuture<'static, Closed> {
        Box::pin(async { Closed::Clean })
    }
}

impl ProviderStream<SpeechEvent> for Finals {}

impl Stt for AmdStt {
    fn label(&self) -> String {
        "fake::amd_stt".into()
    }

    fn capabilities(&self) -> SttCapabilities {
        SttCapabilities { streaming: true, ..Default::default() }
    }

    /// One attempt: a refused open is final.
    fn conn_options(&self) -> Option<ApiConnectOptions> {
        Some(ApiConnectOptions { max_retry: 0, ..Default::default() })
    }

    fn recognize(
        &self,
        _cx: &lk_agents_abi::telemetry::Cx,
        _: &[AudioFrame],
        _: Option<&LanguageCode>,
    ) -> BoxFuture<'_, Result<SpeechEvent, ProviderError>> {
        Box::pin(async move { unreachable!("AMD's STT streams") })
    }

    fn stream(
        &self,
        _cx: &lk_agents_abi::telemetry::Cx,
        _: Option<&LanguageCode>,
        _: Option<f64>,
    ) -> BoxFuture<'_, Result<Option<lk_agents_abi::DuplexStream<SttSink, SttSource>>, ProviderError>>
    {
        Box::pin(async move {
            if self.fails {
                let refused = ApiError::new(ErrorKind::Auth, "the fake AMD STT refused");
                return Err(ProviderError::Local(refused));
            }
            let (sender, heard) = mpsc::unbounded_channel();
            self.ear.streams.lock().unwrap().push(sender);
            let ended = CancellationToken::new();
            Ok(Some(lk_agents_abi::DuplexStream::new(
                Box::new(Ears { ended: ended.clone() }) as SttSink,
                Box::new(Finals { heard, ended }) as SttSource,
            )))
        })
    }
}
