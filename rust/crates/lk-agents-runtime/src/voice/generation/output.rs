use crate::telemetry;
use crate::voice::nodes::OutputNodes;
use crate::voice::session::SessionShared;
use crate::voice::speech_handle::SpeechHandle;
use lk_agents_abi::TranscriptChunk;
use lk_agents_abi::voice::io::Result as IoResult;
use lk_agents_abi::voice::io::{
    AudioOutput, AudioOutputExt, PlaybackEvent, PlaybackFinished, PlaybackObserver,
    PlaybackSegment, TranscriptionOutput,
};
use lk_agents_abi::voice::nodes::TextNode;
use lk_agents_voice::agent_session::machine::output::{OutputBindingEvent, SegmentBinding};
use lk_agents_voice::agent_session::{Answer, RequestId};
use lk_agents_voice::generation::GenEvent;
use lk_agents_voice::transcription::TranscriptSegment;
use std::sync::{Arc, Weak};

// A generation owns its admitted output binding even when cancelled while awaiting it.

pub(super) struct OutputBinding {
    shared: Arc<SessionShared>,
    speech: lk_agents_voice::agent_session::machine::activity::scheduling::SpeechId,
    request: RequestId,
    pub(super) binding: SegmentBinding,
    activated: bool,
    nodes: OutputNodes,
    /// The activated segment's transcript chain, the agent's `transcription_node` over the bound
    /// output, until its delivery settles.
    text: Option<Arc<dyn TranscriptionOutput>>,
}

/// Established before the grant is awaited, so cancellation still releases a late grant.
struct PendingBinding {
    shared: Arc<SessionShared>,
    speech: lk_agents_voice::agent_session::machine::activity::scheduling::SpeechId,
    request: RequestId,
    transferred: bool,
}

impl Drop for PendingBinding {
    fn drop(&mut self) {
        if !self.transferred {
            self.shared.report(GenEvent::OutputBinding(OutputBindingEvent::ReleaseRequested {
                binding: self.request,
                speech: self.speech,
            }));
        }
    }
}

impl OutputBinding {
    pub(super) async fn acquire(
        shared: &Arc<SessionShared>,
        handle: &SpeechHandle,
        nodes: &OutputNodes,
    ) -> Result<Option<Self>, lk_agents_voice::agent_session::DriverError> {
        let (call, answer) = shared.open_call(Answer::output_binding);
        let request = call.id;
        let mut pending = PendingBinding {
            shared: shared.clone(),
            speech: handle.id(),
            request,
            transferred: false,
        };
        shared.report_awaited(
            GenEvent::OutputBinding(OutputBindingEvent::BindRequested {
                call: request,
                speech: handle.id(),
            }),
            call,
        );
        let binding = answer.await.map_err(|_| {
            lk_agents_voice::agent_session::machine::output::OutputBindingError::ReplyLost
        })??;
        let Some(binding) = binding else { return Ok(None) };
        pending.transferred = true;
        Ok(Some(Self {
            shared: shared.clone(),
            speech: handle.id(),
            request,
            binding,
            activated: false,
            nodes: nodes.clone(),
            text: None,
        }))
    }

    pub(super) fn audio(&self) -> Option<Arc<dyn AudioOutput>> {
        self.binding
            .outputs
            .audio
            .as_ref()
            .filter(|_| self.binding.outputs.audio_enabled)
            .and_then(|r| self.shared.resources.audio_output(r.id))
    }

    /// The transcript chain; `None` before activation.
    pub(super) fn text(&self) -> Option<Arc<dyn TranscriptionOutput>> {
        self.text.clone()
    }

    pub(super) fn text_drained(&mut self) {
        self.text = None;
    }

    pub(super) async fn activate(
        &mut self,
    ) -> Result<bool, lk_agents_voice::agent_session::DriverError> {
        if self.activated {
            return Ok(true);
        }
        let (call, answer) = self.shared.open_call(Answer::output_activated);
        self.shared.report_awaited(
            GenEvent::OutputBinding(OutputBindingEvent::ActivationRequested {
                call: call.id,
                binding: self.request,
                speech: self.speech,
            }),
            call,
        );
        let Some(binding) = answer.await.map_err(|_| {
            lk_agents_voice::agent_session::machine::output::OutputBindingError::ReplyLost
        })??
        else {
            return Ok(false);
        };
        self.binding = binding;
        let sink = (self.binding.outputs.transcription.as_ref())
            .filter(|_| self.binding.outputs.transcription_enabled)
            .and_then(|r| self.shared.resources.transcription_output(r.id));
        self.text = transcription_output(&self.nodes, sink);
        self.activated =
            self.audio().is_none_or(|audio| audio.begin_segment(self.binding.segment).is_ok());
        if !self.activated {
            self.shared.trace(lk_agents_voice::trace::PlayoutStep::OutputRefused {
                speech: self.speech,
                reason: lk_agents_voice::agent_session::machine::output::OutputBindingError::ActivationUnavailable,
            });
            return Err(
                lk_agents_voice::agent_session::machine::output::OutputBindingError::ActivationUnavailable
                    .into(),
            );
        }
        Ok(true)
    }
}

impl Drop for OutputBinding {
    fn drop(&mut self) {
        if let Some(text) = self.text.take() {
            // An aborted generation cannot await. Its existing text receipt owns retirement.
            let retirement = Self {
                shared: self.shared.clone(),
                speech: self.speech,
                request: self.request,
                binding: self.binding.clone(),
                activated: self.activated,
                nodes: OutputNodes::default(),
                text: None,
            };
            text.retain_until_flushed(Box::new(retirement));
            text.cancel_flush();
            return;
        }
        if let Some(audio) = self.audio()
            && audio.tracker().has_pending_segment(self.binding.segment)
        {
            audio.tracker().observe(Arc::new(OutputSettlement {
                shared: Arc::downgrade(&self.shared),
                output: self.binding.outputs.audio.as_ref().expect("resolved audio").id,
                request: self.request,
                speech: self.speech,
                segment: self.binding.segment,
            }));
            audio.cut();
            if audio.tracker().has_pending_segment(self.binding.segment) {
                self.shared.report(GenEvent::OutputBinding(OutputBindingEvent::CleanupPending {
                    binding: self.request,
                    speech: self.speech,
                }));
                return;
            }
        }
        self.shared.report(GenEvent::OutputBinding(OutputBindingEvent::ReleaseRequested {
            binding: self.request,
            speech: self.speech,
        }));
    }
}

/// The resource table retains unreported media. A late matched report releases its core lease;
/// the weak reference prevents an output's observer from keeping its session alive.
struct OutputSettlement {
    shared: Weak<SessionShared>,
    output: lk_agents_abi::resources::ResourceId,
    request: RequestId,
    speech: lk_agents_voice::agent_session::machine::activity::scheduling::SpeechId,
    segment: PlaybackSegment,
}

impl OutputSettlement {
    fn release(&self) {
        if let Some(shared) = self.shared.upgrade() {
            shared.report(GenEvent::OutputBinding(OutputBindingEvent::ReleaseRequested {
                binding: self.request,
                speech: self.speech,
            }));
        }
    }
}

impl PlaybackObserver for OutputSettlement {
    fn on_playback_event(&self, _: &PlaybackEvent) {}
    fn on_segment_finished(&self, segment: PlaybackSegment, _: &PlaybackFinished) {
        if segment == self.segment {
            self.release();
        }
    }
    fn on_playback_reset(&self) {
        self.release();
    }
    fn is_closed(&self) -> bool {
        self.shared
            .upgrade()
            .and_then(|shared| shared.resources.audio_output(self.output))
            .is_none_or(|audio| !audio.tracker().has_pending_segment(self.segment))
    }
}

/// Bind the agent's transcription node to this segment's admitted transcription output.
fn transcription_output(
    nodes: &OutputNodes,
    below: Option<Arc<dyn TranscriptionOutput>>,
) -> Option<Arc<dyn TranscriptionOutput>> {
    let below: Arc<dyn TranscriptionOutput> = Arc::new(SegmentGate::new(below?));
    let Some(factory) = &nodes.transcription else { return Some(below) };
    match factory.text_node(Arc::clone(&below)) {
        Ok(node) => Some(Arc::new(NodeTranscriptionOutput::new(node, below))),
        Err(e) => {
            log::error!(target: lk_agents_abi::LOG_TARGET, error = e.to_string().as_str();
                    "failed to open the transcription_node");
            None
        }
    }
}

/// The session's transcription output as the generation (or its node) writes into it: a flush
/// reaches it only when text did since the last one, so it never closes an empty segment.
struct SegmentGate {
    next: Arc<dyn TranscriptionOutput>,
    segment: parking_lot::Mutex<TranscriptSegment>,
}

impl SegmentGate {
    fn new(next: Arc<dyn TranscriptionOutput>) -> Self {
        SegmentGate { next, segment: Default::default() }
    }
}

impl TranscriptionOutput for SegmentGate {
    fn label(&self) -> &str {
        self.next.label()
    }

    fn next_in_chain(&self) -> Option<Arc<dyn TranscriptionOutput>> {
        Some(Arc::clone(&self.next))
    }

    fn capture_timed<'a>(
        &'a self,
        cx: &telemetry::Cx,
        text: &'a TranscriptChunk,
    ) -> lk_agents_abi::BoxFuture<'a, IoResult<()>> {
        self.segment.lock().wrote(&text.text);
        self.next.capture_timed(cx, text)
    }

    fn capture_text<'a>(
        &'a self,
        cx: &telemetry::Cx,
        text: &'a str,
    ) -> lk_agents_abi::BoxFuture<'a, IoResult<()>> {
        self.segment.lock().wrote(text);
        self.next.capture_text(cx, text)
    }

    fn wait_for_flush(&self) -> lk_agents_abi::BoxFuture<'_, IoResult<()>> {
        self.next.wait_for_flush()
    }

    fn flush(&self) {
        let closes = self.segment.lock().flush();
        if closes {
            self.next.flush();
        }
    }
}

/// A [`TextNode`] as a link in the chain, with the core leg it emits into below it.
struct NodeTranscriptionOutput {
    node: Arc<dyn TextNode>,
    next: Arc<dyn TranscriptionOutput>,
}

impl NodeTranscriptionOutput {
    fn new(node: Arc<dyn TextNode>, next: Arc<dyn TranscriptionOutput>) -> Self {
        NodeTranscriptionOutput { node, next }
    }
}

impl TranscriptionOutput for NodeTranscriptionOutput {
    fn label(&self) -> &str {
        "Agent.transcription_node"
    }

    fn next_in_chain(&self) -> Option<Arc<dyn TranscriptionOutput>> {
        Some(Arc::clone(&self.next))
    }

    fn capture_timed<'a>(
        &'a self,
        _cx: &telemetry::Cx,
        text: &'a TranscriptChunk,
    ) -> lk_agents_abi::BoxFuture<'a, IoResult<()>> {
        Box::pin(async move {
            self.node.capture_timed(text);
            Ok(())
        })
    }

    fn capture_text<'a>(
        &'a self,
        _cx: &telemetry::Cx,
        text: &'a str,
    ) -> lk_agents_abi::BoxFuture<'a, IoResult<()>> {
        Box::pin(async move {
            self.node.capture_text(text);
            Ok(())
        })
    }

    fn retain_until_flushed(&self, owner: Box<dyn Send>) {
        self.node.retain_until_drained(Box::new(TranscriptionOutputOwner {
            next: self.next.clone(),
            owner: Some(owner),
        }));
    }

    fn cancel_flush(&self) {
        self.node.cancel();
        self.next.cancel_flush();
    }

    fn wait_for_flush(&self) -> lk_agents_abi::BoxFuture<'_, IoResult<()>> {
        Box::pin(async {
            self.node.drain().await?;
            self.next.wait_for_flush().await
        })
    }

    fn flush(&self) {
        self.node.flush();
    }
}

/// Pass a binding through every processing receipt in its text chain.
struct TranscriptionOutputOwner {
    next: Arc<dyn TranscriptionOutput>,
    owner: Option<Box<dyn Send>>,
}

impl Drop for TranscriptionOutputOwner {
    fn drop(&mut self) {
        if let Some(owner) = self.owner.take() {
            self.next.retain_until_flushed(owner);
        }
    }
}

#[cfg(test)]
mod text_node_tests {
    use super::*;
    use std::task::{Context, Poll, Waker};

    /// A text node whose drain finishes when the test completes it (or on its cancel).
    #[derive(Default)]
    struct BufferedTextNode(tokio::sync::watch::Sender<bool>);

    impl BufferedTextNode {
        fn complete(&self) {
            self.0.send_replace(true);
        }
    }

    impl TextNode for BufferedTextNode {
        fn capture_text(&self, _: &str) {}
        fn capture_timed(&self, _: &TranscriptChunk) {}
        fn flush(&self) {}
        fn cancel(&self) {
            self.complete();
        }
        fn retain_until_drained(&self, _: Box<dyn Send>) {}
        fn drain(&self) -> lk_agents_abi::BoxFuture<'_, IoResult<()>> {
            Box::pin(async move {
                let _ = self.0.subscribe().wait_for(|done| *done).await;
                Ok(())
            })
        }
    }

    struct Destination;
    impl TranscriptionOutput for Destination {
        fn label(&self) -> &str {
            "destination"
        }
        fn capture_text<'a>(
            &'a self,
            _cx: &telemetry::Cx,
            _: &'a str,
        ) -> lk_agents_abi::BoxFuture<'a, IoResult<()>> {
            Box::pin(async { Ok(()) })
        }
        fn flush(&self) {}
    }

    #[test]
    fn a_buffered_text_node_keeps_the_segment_pending_until_its_drain_finishes() {
        let delivered = Arc::new(BufferedTextNode::default());
        let output = NodeTranscriptionOutput::new(delivered.clone(), Arc::new(Destination));
        output.flush();
        let mut completion = output.wait_for_flush();
        let mut cx = Context::from_waker(Waker::noop());
        assert!(matches!(completion.as_mut().poll(&mut cx), Poll::Pending));
        delivered.complete();
        assert!(matches!(completion.as_mut().poll(&mut cx), Poll::Ready(Ok(()))));
    }
    #[test]
    fn cancelling_a_buffered_text_node_settles_its_own_segment() {
        let delivered = Arc::new(BufferedTextNode::default());
        let output = NodeTranscriptionOutput::new(delivered, Arc::new(Destination));
        output.flush();
        let mut completion = output.wait_for_flush();
        let mut cx = Context::from_waker(Waker::noop());
        assert!(matches!(completion.as_mut().poll(&mut cx), Poll::Pending));
        output.cancel_flush();
        assert!(matches!(completion.as_mut().poll(&mut cx), Poll::Ready(Ok(()))));
    }

    /// What reached the session's transcription output.
    #[derive(Default)]
    struct Recorded(parking_lot::Mutex<Vec<String>>);
    impl TranscriptionOutput for Recorded {
        fn label(&self) -> &str {
            "recorded"
        }
        fn capture_text<'a>(
            &'a self,
            _cx: &telemetry::Cx,
            text: &'a str,
        ) -> lk_agents_abi::BoxFuture<'a, IoResult<()>> {
            self.0.lock().push(text.to_owned());
            Box::pin(async { Ok(()) })
        }
        fn flush(&self) {
            self.0.lock().push("<flush>".into());
        }
    }

    // Catches: a generation that wrote no text (a realtime message cancelled before its first
    // delta) closing an empty segment on the session's transcription output.
    #[tokio::test]
    async fn a_flush_closes_only_a_segment_that_had_text() {
        let recorded = Arc::new(Recorded::default());
        let output = SegmentGate::new(recorded.clone());
        output.flush();
        output.capture_text(&telemetry::Cx::NONE, "").await.unwrap();
        output.flush();
        output.capture_text(&telemetry::Cx::NONE, "Hello.").await.unwrap();
        output.flush();
        output.flush();
        assert_eq!(*recorded.0.lock(), ["", "Hello.", "<flush>"]);
    }
}
