//! A live stream's two halves: the sink the caller pushes audio into, with its mid-stream
//! control, and the source it drains speech events from.

use super::*;

/// A keyterm change that arrived mid-utterance, deferred to `END_OF_SPEECH`: "the gateway may
/// reconnect upstream when the keyterms change". One slot, latest wins.
#[derive(Debug, Default)]
pub(super) struct DeferredExtra {
    pub(super) speaking: std::sync::atomic::AtomicBool,
    pub(super) pending: Mutex<Option<serde_json::Map<String, serde_json::Value>>>,
}

pub struct RecognitionInput {
    pub(super) audio: AudioInput,
    pub(super) control: Arc<StreamControl>,
}

impl lk_agents_abi::stt::inference::InferenceSttAudio for RecognitionInput {
    fn update_options(
        &self,
        model: Option<String>,
        language: Option<String>,
        extra: Option<serde_json::Map<String, serde_json::Value>>,
    ) {
        self.control.update_options(model.as_deref(), language.as_deref(), extra.as_ref());
    }
}

/// What reaches a live stream from its client: `session.update` frames and deferred keyterms.
pub(super) struct StreamControl {
    pub(super) stop: tokio::sync::watch::Sender<bool>,
    pub(super) failure: tokio::sync::watch::Sender<Option<ApiError>>,
    /// Unbounded: one update per option change or turn, sent ahead of queued audio.
    pub(super) updates: mpsc::UnboundedSender<serde_json::Value>,
    pub(super) deferred: Arc<DeferredExtra>,
}

impl StreamControl {
    /// A mid-stream `session.update`; unsupported providers ignore it.
    pub fn update_options(
        &self,
        model: Option<&str>,
        language: Option<&str>,
        extra: Option<&serde_json::Map<String, serde_json::Value>>,
    ) {
        if extra.is_some() {
            // An explicit `extra` clears any deferred keyterm change (`_pending_extra = None`).
            *self.deferred.pending.lock() = None;
        }
        if let Some(frame) = wire::session_update(model, language, extra) {
            // A stream already gone has nothing to update.
            let _ = self.updates.send(frame);
        }
    }

    /// Now if the user is silent, at end of speech if not.
    pub fn apply_keyterms(&self, extra: serde_json::Map<String, serde_json::Value>) {
        if self.deferred.speaking.load(std::sync::atomic::Ordering::SeqCst) {
            *self.deferred.pending.lock() = Some(extra);
            return;
        }
        self.update_options(None, None, Some(&extra));
    }
}

impl StreamInput<AudioFrame> for RecognitionInput {
    fn push(&self, item: AudioFrame) -> Result<(), ProviderError> {
        if !*self.control.stop.borrow() && self.audio.push(item) {
            return Ok(());
        }
        Err(self.closed_error())
    }

    fn flush(&self) -> Result<(), ProviderError> {
        match self.audio.flush() {
            Some(flushed) if !*self.control.stop.borrow() => {
                flushed.map_err(|stuck| ApiError::new(stuck.to_string()).into())
            }
            _ => Err(self.closed_error()),
        }
    }

    fn end_input(&self) {
        self.audio.end();
    }
}

impl RecognitionInput {
    fn closed_error(&self) -> ProviderError {
        self.control
            .failure
            .borrow()
            .clone()
            .map(ProviderError::from)
            .unwrap_or_else(|| stream_closed(WHAT))
    }
}

/// The gateway stream's drain half, handed to the host as `InferenceSttStream` beside its sink.
pub struct RecognitionOutput {
    pub(super) events: mpsc::Receiver<Result<SpeechEvent, ApiError>>,
    pub(super) done: bool,
    pub(super) stop: tokio::sync::watch::Sender<bool>,
    pub(super) joined: tokio::sync::oneshot::Receiver<()>,
    pub(super) failure: tokio::sync::watch::Receiver<Option<ApiError>>,
    pub(super) clock: Arc<dyn Clock>,
}

impl StreamOutput<SpeechEvent> for RecognitionOutput {
    fn next(&mut self) -> lk_agents_abi::BoxFuture<'_, Result<Option<SpeechEvent>, ProviderError>> {
        Box::pin(async move {
            if self.done {
                return Ok(None);
            }
            match self.events.recv().await {
                Some(Ok(event)) => Ok(Some(event)),
                Some(Err(error)) => {
                    self.done = true;
                    Err(error.into())
                }
                None => {
                    self.done = true;
                    match self.failure.borrow().clone() {
                        Some(error) => Err(error.into()),
                        None => Ok(None),
                    }
                }
            }
        })
    }

    fn close(
        mut self: Box<Self>,
        deadline: lk_agents_abi::Ts,
    ) -> lk_agents_abi::BoxFuture<'static, lk_agents_abi::Closed> {
        Box::pin(async move {
            self.done = true;
            self.stop.send_replace(true);
            self.events.close();
            let mut settled = false;
            let mut closed = lk_agents_abi::Closed::bounded(self.clock.as_ref(), deadline, async {
                settled = (&mut self.joined).await.is_ok();
            })
            .await;
            if !settled {
                closed = lk_agents_abi::Closed::TimedOut;
            }
            if closed == lk_agents_abi::Closed::TimedOut {
                log::error!(
                    "STT executor cleanup incomplete; executor remains owned until settlement"
                );
            }
            closed
        })
    }
}

impl lk_agents_abi::utils::stream::ProviderStream<SpeechEvent> for RecognitionOutput {}
