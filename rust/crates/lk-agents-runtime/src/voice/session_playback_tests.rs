//! A silent physical sink fails boundedly without completing an unread replacement stream.
use super::*;

#[tokio::test(start_paused = true)]
async fn missing_playback_progress_and_finish_ends_the_speech_without_inventing_heard_audio() {
    let api = running(unobserved(no_room(), None)).await;
    let output = Arc::new(MissingPlaybackReports::default());
    api.set_audio_output(Some(output.clone())).unwrap();
    let frames = Items([lk_agents_abi::AudioFrame::new(vec![0; 320], 16_000, 1, 160)].into());
    let speech =
        api.say_audio("Not confirmed heard.".into(), Box::new(frames), None, true, None).unwrap();
    settle().await;
    assert!(output.1.get().is_some(), "the output never accepted the segment");
    // Ten milliseconds delivered, the transport's two-second allowance, then one clock tick.
    api.time().advance(Duration::from_millis(2011));
    settle().await;
    assert!(speech.done(), "missing progress left the playback finish deadline unarmed");
    assert!(speech.exception().unwrap().is_some(), "missing reports fabricated clean playback");
    assert_eq!(output.0.position(), Duration::ZERO, "unreported audio was counted as heard");
    let key = *output.1.get().expect("the sink latched its external operation key");
    assert!(output.0.has_pending_segment(key), "timeout fabricated a leaf completion");
    // A replacement speech cannot turn an unavailable output into successful completion while
    // its input remains open. The refusal is separate from interruption or normal EOF.
    let (blocked, still_open) =
        api.say_audio_channel("Still open.".into(), None, true, None).unwrap();
    settle().await;
    assert!(blocked.done(), "unavailable output did not refuse the replacement speech");
    let failure = blocked.exception().unwrap().expect("output refusal was reported as success");
    assert!(failure.to_string().contains("output admission refused"), "{failure:?}");
    drop(still_open);
    // Explicit resource cleanup settles the retained uncertain binding.
    output.0.reset();
    api.close(CloseReason::UserInitiated, false, None).await;
}

struct PendingTextCleanup {
    captured: AtomicBool,
    cancelled: crate::monotone::Monotone<bool>,
    allow_cleanup: crate::monotone::Monotone<bool>,
    completed: crate::monotone::Monotone<bool>,
    retained: Mutex<Option<Box<dyn Send>>>,
}

impl Default for PendingTextCleanup {
    fn default() -> Self {
        Self {
            captured: AtomicBool::new(false),
            cancelled: crate::monotone::Monotone::new(false),
            allow_cleanup: crate::monotone::Monotone::new(false),
            completed: crate::monotone::Monotone::new(false),
            retained: Mutex::new(None),
        }
    }
}

struct ClosingTextNode(Arc<PendingTextCleanup>);

impl lk_agents_abi::voice::nodes::TextNode for ClosingTextNode {
    fn capture_text(&self, _: &str) {
        self.0.captured.store(true, Ordering::SeqCst);
    }
    fn capture_timed(&self, text: &lk_agents_abi::TranscriptChunk) {
        self.capture_text(&text.text);
    }
    fn flush(&self) {}
    fn retain_until_drained(&self, owner: Box<dyn Send>) {
        let mut retained = self.0.retained.lock();
        if !self.0.completed.get() {
            assert!(retained.is_none());
            *retained = Some(owner);
        } else {
            drop(retained);
            drop(owner);
        }
    }
    fn cancel(&self) {
        self.0.cancelled.set(true);
    }
    fn drain(&self) -> lk_agents_abi::BoxFuture<'_, lk_agents_abi::voice::io::Result<()>> {
        Box::pin(async move {
            self.0.completed.wait_for(|done| *done).await;
            Ok(())
        })
    }
}

impl Drop for ClosingTextNode {
    fn drop(&mut self) {
        self.0.cancelled.set(true);
    }
}

struct ClosingTextFactory(Arc<PendingTextCleanup>);

impl lk_agents_abi::voice::nodes::NodeFactory for ClosingTextFactory {
    fn text_node(
        &self,
        below: Arc<dyn lk_agents_abi::voice::io::TextOutput>,
    ) -> Result<Arc<dyn lk_agents_abi::voice::nodes::TextNode>, lk_agents_abi::errors::HostError>
    {
        let state = self.0.clone();
        // The host drain owns its output until its explicitly gated source close finishes.
        tokio::spawn(async move {
            state.cancelled.wait_for(|cancelled| *cancelled).await;
            state.allow_cleanup.wait_for(|allowed| *allowed).await;
            below.flush();
            let owner = {
                let mut retained = state.retained.lock();
                state.completed.set(true);
                retained.take()
            };
            drop(owner);
        });
        Ok(Arc::new(ClosingTextNode(self.0.clone())))
    }
    fn llm_node<'a>(
        &'a self,
        _: &'a lk_agents_abi::ChatRequest,
    ) -> lk_agents_abi::BoxFuture<'a, Result<lk_agents_abi::LlmStream, ProviderError>> {
        Box::pin(async { Err(lk_agents_abi::errors::ApiError::connection(None).into()) })
    }
    fn tts_node(
        &self,
        _: bool,
    ) -> lk_agents_abi::BoxFuture<
        '_,
        Result<
            lk_agents_abi::DuplexStream<lk_agents_abi::TtsSink, lk_agents_abi::TtsSource>,
            ProviderError,
        >,
    > {
        Box::pin(async { Err(lk_agents_abi::errors::ApiError::connection(None).into()) })
    }
    fn stt_node(
        &self,
    ) -> lk_agents_abi::BoxFuture<
        '_,
        Result<
            lk_agents_abi::DuplexStream<lk_agents_abi::SttSink, lk_agents_abi::SttSource>,
            ProviderError,
        >,
    > {
        Box::pin(async { Err(lk_agents_abi::errors::ApiError::connection(None).into()) })
    }
}

struct TextDestination(AtomicBool);

impl lk_agents_abi::voice::io::TextOutput for TextDestination {
    fn label(&self) -> &str {
        "text destination"
    }
    fn capture_text<'a>(
        &'a self,
        _: &'a str,
    ) -> lk_agents_abi::BoxFuture<'a, lk_agents_abi::voice::io::Result<()>> {
        Box::pin(async { Ok(()) })
    }
    fn flush(&self) {}
    fn on_detached(&self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

#[tokio::test(start_paused = true)]
async fn a_hard_aborted_generation_keeps_its_text_binding_until_source_cleanup_finishes() {
    let api = running(unobserved(no_room(), None)).await;
    api.set_audio_output(None).unwrap();
    let old = Arc::new(TextDestination(AtomicBool::new(false)));
    api.set_transcription_output(Some(old.clone())).unwrap();
    let state = Arc::new(PendingTextCleanup::default());
    let mut update = agent_update("buffering", Instructions::plain("Speak clearly."));
    update.output_nodes.transcription = Some(
        api.resources.register_transcription_node(Arc::new(ClosingTextFactory(state.clone()))),
    );
    let (call, reply) = api.call(Answer::handoff).unwrap();
    update.call = Some(call.id);
    api.send_awaited(VoiceCommandOrigin::Host, ActivityCommand::Update(Box::new(update)), call)
        .unwrap();
    assert_eq!(reply.await.unwrap(), Ok(()));
    let speech = api.say("Still processing.".into(), None, true, None).unwrap();
    settle().await;
    assert!(
        state.captured.load(Ordering::SeqCst),
        "the text segment never started: resources={:?}, done={}, exception={:?}",
        api.resources.live(),
        speech.done(),
        speech.exception()
    );
    api.set_transcription_output(Some(Arc::new(TextDestination(AtomicBool::new(false))))).unwrap();
    settle().await;
    assert!(!old.0.load(Ordering::SeqCst), "selection retired an admitted text segment");
    // The interruption watchdog cancels the speech's tasks while the generation's drain waits.
    lk_agents_abi::voice::speech_handle::SpeechHandle::interrupt(&*speech, true).unwrap();
    api.time().advance(lk_agents_abi::HOST_TEARDOWN_TIMEOUT);
    settle().await;
    assert!(state.cancelled.get(), "hard cancellation did not reach the source owner");
    assert!(
        !old.0.load(Ordering::SeqCst),
        "the old text binding retired before its source cleanup completed"
    );
    state.allow_cleanup.set(true);
    settle().await;
    assert!(state.completed.get(), "the cancelled source did not finish cleanup");
    assert!(old.0.load(Ordering::SeqCst), "the completed old binding was never retired");
    api.close(CloseReason::UserInitiated, false, None).await;
}
