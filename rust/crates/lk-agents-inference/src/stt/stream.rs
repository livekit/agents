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
    /// Every `extra` sent so far, merged: a reconnect opens with it.
    pub(super) extra: serde_json::Map<String, serde_json::Value>,
}

pub struct RecognitionInput {
    pub(super) audio: AudioInput,
    pub(super) control: Arc<StreamControl>,
}

/// What reaches a live stream from its client: `session.update` frames and deferred keyterms.
pub(super) struct StreamControl {
    /// Shared with the output: the stream stops once both halves are gone.
    pub(super) stop: Arc<tokio_util::sync::DropGuard>,
    /// The stream's failure, once it has one, or its end: what its input answers.
    pub(super) gate: lk_agents_net::source::InputGate,
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
            if let Some(extra) = extra {
                options.extra.extend(extra.clone());
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
                self.gate.fail(error.into());
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
        self.stopped()?;
        self.audio.push(item)
    }

    fn head(&self, item: AudioFrame) -> Result<(), ProviderError> {
        self.stopped()?;
        self.audio.head(item)
    }

    fn filler(&self, item: AudioFrame) -> Result<(), ProviderError> {
        self.stopped()?;
        self.audio.filler(item)
    }

    fn flush(&self) -> Result<(), ProviderError> {
        self.stopped()?;
        self.audio.flush()
    }

    fn end_input(&self) {
        self.audio.end();
    }
}

impl RecognitionInput {
    /// The stream stopped (both halves gone, or a failure): the gate's answer.
    fn stopped(&self) -> Result<(), ProviderError> {
        match self.control.stop.token().is_cancelled() {
            true => Err(self.control.gate.refused()),
            false => Ok(()),
        }
    }
}

/// The gateway stream's drain half, handed to the host as `InferenceSttStream` beside its sink.
pub struct RecognitionOutput {
    pub(super) events: mpsc::Receiver<Result<SpeechEvent, ApiError>>,
    pub(super) done: bool,
    pub(super) stop: Arc<tokio_util::sync::DropGuard>,
    pub(super) joined: tokio::sync::oneshot::Receiver<()>,
    pub(super) gate: lk_agents_net::source::InputGate,
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
                    let error = ProviderError::from(error);
                    self.gate.fail(error.clone());
                    Err(error)
                }
                None => {
                    self.done = true;
                    match self.gate.failure() {
                        Some(error) => Err(error),
                        None => {
                            self.gate.end();
                            Ok(None)
                        }
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
            self.gate.end();
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
