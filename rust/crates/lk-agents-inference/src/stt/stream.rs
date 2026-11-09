//! A live stream's two halves: the sink the caller pushes audio into, with the control its
//! client's `update_options` reaches, and the source it drains speech events from.

use super::*;

/// A keyterm change that arrived mid-utterance, deferred to `END_OF_SPEECH`: "the gateway may
/// reconnect upstream when the keyterms change". One slot, latest wins.
#[derive(Debug, Default)]
pub(super) struct DeferredExtra {
    pub(super) speaking: std::sync::atomic::AtomicBool,
    pub(super) pending: Mutex<Option<serde_json::Map<String, serde_json::Value>>>,
    pub(super) options: Mutex<LiveOptions>,
}

#[derive(Debug, Default)]
pub(super) struct LiveOptions {
    pub(super) model: Option<String>,
    pub(super) language: Option<String>,
}

pub struct RecognitionInput {
    pub(super) audio: AudioInput,
    pub(super) control: Arc<StreamControl>,
}

/// What reaches a live stream from its client: `session.update` frames and deferred keyterms.
pub(super) struct StreamControl {
    /// Shared with the output: the stream stops once both halves are gone.
    pub(super) stop: Arc<tokio_util::sync::DropGuard>,
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
        let frame = {
            let mut options = self.deferred.options.lock();
            if let Some(model) = model {
                options.model = Some(model.to_owned());
            }
            if let Some(language) = language {
                options.language = Some(language.to_owned());
            }
            wire::session_update(
                options.model.as_deref().unwrap_or(""),
                options.language.as_deref(),
                model,
                language,
                extra,
            )
        };
        match frame {
            Ok(Some(frame)) => {
                let _ = self.updates.send(frame);
            }
            Err(error) => {
                self.failure.send_replace(Some(error));
                self.stop.token().cancel();
            }
            Ok(None) => {}
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
        if !self.control.stop.token().is_cancelled() && self.audio.push(item) {
            return Ok(());
        }
        Err(self.closed_error())
    }

    fn head(&self, item: AudioFrame) -> Result<(), ProviderError> {
        if !self.control.stop.token().is_cancelled() && self.audio.head(item) {
            return Ok(());
        }
        Err(self.closed_error())
    }

    fn filler(&self, item: AudioFrame) -> Result<(), ProviderError> {
        if !self.control.stop.token().is_cancelled() && self.audio.filler(item) {
            return Ok(());
        }
        Err(self.closed_error())
    }

    fn flush(&self) -> Result<(), ProviderError> {
        match self.audio.flush() {
            Some(flushed) if !self.control.stop.token().is_cancelled() => {
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
    pub(super) stop: Arc<tokio_util::sync::DropGuard>,
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
            self.stop.token().cancel();
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
                log::error!(target: lk_agents_abi::LOG_TARGET,
                    "STT executor cleanup incomplete; executor remains owned until settlement"
                );
            }
            closed
        })
    }
}

impl lk_agents_abi::utils::stream::ProviderStream<SpeechEvent> for RecognitionOutput {}
