//! A generation owns its admitted output binding even when it is cancelled while awaiting it.

use lk_agents_abi::TimedString;
use lk_agents_abi::voice::io::Result as IoResult;
use lk_agents_abi::voice::io::{
    AudioOutput, AudioOutputExt, PlaybackEvent, PlaybackFinished, PlaybackObserver,
    PlaybackSegment, TextOutput,
};
use lk_agents_abi::voice::nodes::TextNode;
use lk_agents_voice::agent_session::output::SegmentBinding;
use lk_agents_voice::agent_session::{Answer, RequestId, SessionShared};
use lk_agents_voice::generation::GenEvent;
use lk_agents_voice::speech_handle::SpeechHandle;
use std::sync::{Arc, Weak};

pub(super) struct OutputBinding {
    shared: Arc<SessionShared>,
    speech: lk_agents_voice::agent_activity::scheduling::SpeechId,
    request: RequestId,
    pub(super) binding: SegmentBinding,
    activated: bool,
    text_processing: Option<Arc<dyn TextOutput>>,
}

/// Established before the grant is awaited, so cancellation still releases a late grant.
struct PendingBinding {
    shared: Arc<SessionShared>,
    speech: lk_agents_voice::agent_activity::scheduling::SpeechId,
    request: RequestId,
    transferred: bool,
}

impl Drop for PendingBinding {
    fn drop(&mut self) {
        if !self.transferred {
            self.shared
                .report(GenEvent::ReleaseOutput { binding: self.request, speech: self.speech });
        }
    }
}

impl OutputBinding {
    pub(super) async fn acquire(
        shared: &Arc<SessionShared>,
        handle: &SpeechHandle,
    ) -> Result<Option<Self>, lk_agents_voice::agent_session::DriverError> {
        let (call, answer) = shared.open_call(Answer::output_binding);
        let request = call.id;
        let mut pending = PendingBinding {
            shared: shared.clone(),
            speech: handle.id(),
            request,
            transferred: false,
        };
        shared.report_awaited(GenEvent::BindOutput { call: request, speech: handle.id() }, call);
        let binding = answer
            .await
            .map_err(|_| lk_agents_voice::agent_session::output::OutputBindingError::ReplyLost)??;
        let Some(binding) = binding else { return Ok(None) };
        pending.transferred = true;
        Ok(Some(Self {
            shared: shared.clone(),
            speech: handle.id(),
            request,
            binding,
            activated: false,
            text_processing: None,
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

    pub(super) fn text(&self) -> Option<Arc<dyn TextOutput>> {
        self.binding
            .outputs
            .transcription
            .as_ref()
            .filter(|_| self.binding.outputs.transcription_enabled)
            .and_then(|r| self.shared.resources.text_output(r.id))
    }

    pub(super) fn retain_text_processing(&mut self, text: Option<Arc<dyn TextOutput>>) {
        self.text_processing = text;
    }

    pub(super) fn text_drained(&mut self) {
        self.text_processing = None;
    }

    pub(super) async fn activate(
        &mut self,
    ) -> Result<bool, lk_agents_voice::agent_session::DriverError> {
        if self.activated {
            return Ok(true);
        }
        let (call, answer) = self.shared.open_call(Answer::output_activated);
        self.shared.report_awaited(
            GenEvent::ActivateOutput { call: call.id, binding: self.request, speech: self.speech },
            call,
        );
        if !answer
            .await
            .map_err(|_| lk_agents_voice::agent_session::output::OutputBindingError::ReplyLost)??
        {
            return Ok(false);
        }
        self.activated =
            self.audio().is_none_or(|audio| audio.begin_segment(self.binding.segment).is_ok());
        if !self.activated {
            self.shared.trace(lk_agents_voice::trace::PlayoutStep::OutputRefused {
                speech: self.speech,
                reason: lk_agents_voice::agent_session::output::OutputBindingError::ActivationUnavailable,
            });
            return Err(
                lk_agents_voice::agent_session::output::OutputBindingError::ActivationUnavailable
                    .into(),
            );
        }
        Ok(true)
    }
}

impl Drop for OutputBinding {
    fn drop(&mut self) {
        if let Some(text) = self.text_processing.take() {
            // An aborted generation cannot await. Its existing text receipt owns retirement.
            let retirement = Self {
                shared: self.shared.clone(),
                speech: self.speech,
                request: self.request,
                binding: self.binding.clone(),
                activated: self.activated,
                text_processing: None,
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
                self.shared.report(GenEvent::OutputUnsettled {
                    binding: self.request,
                    speech: self.speech,
                });
                return;
            }
        }
        self.shared.report(GenEvent::ReleaseOutput { binding: self.request, speech: self.speech });
    }
}

/// The resource table retains unreported media. A late matched report releases its core lease;
/// the weak reference prevents an output's observer from keeping its session alive.
struct OutputSettlement {
    shared: Weak<SessionShared>,
    output: lk_agents_abi::resources::ResourceId,
    request: RequestId,
    speech: lk_agents_voice::agent_activity::scheduling::SpeechId,
    segment: PlaybackSegment,
}

impl OutputSettlement {
    fn release(&self) {
        if let Some(shared) = self.shared.upgrade() {
            shared.report(GenEvent::ReleaseOutput { binding: self.request, speech: self.speech });
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

/// Bind the agent's transcription node to this segment's admitted text output.
pub(super) fn text_output(
    nodes: &lk_agents_voice::agent_session::nodes::OutputNodes,
    below: Option<Arc<dyn TextOutput>>,
) -> Option<Arc<dyn TextOutput>> {
    let below = below?;
    let Some(factory) = &nodes.transcription else { return Some(below) };
    match factory.text_node(Arc::clone(&below)) {
        Ok(node) => Some(Arc::new(NodeTextOutput::new(node, below))),
        Err(e) => {
            log::error!(target: "livekit.agents", error = e.to_string().as_str();
                    "failed to open the transcription_node");
            None
        }
    }
}

/// A [`TextNode`] as a link in the chain, with the core leg it emits into below it.
struct NodeTextOutput {
    node: Arc<dyn TextNode>,
    next: Arc<dyn TextOutput>,
}

impl NodeTextOutput {
    fn new(node: Arc<dyn TextNode>, next: Arc<dyn TextOutput>) -> Self {
        NodeTextOutput { node, next }
    }
}

impl TextOutput for NodeTextOutput {
    fn label(&self) -> &str {
        "Agent.transcription_node"
    }

    fn next_in_chain(&self) -> Option<Arc<dyn TextOutput>> {
        Some(Arc::clone(&self.next))
    }

    fn capture_timed<'a>(
        &'a self,
        text: &'a TimedString,
    ) -> lk_agents_abi::BoxFuture<'a, IoResult<()>> {
        Box::pin(async move {
            self.node.capture_timed(text);
            Ok(())
        })
    }

    fn capture_text<'a>(&'a self, text: &'a str) -> lk_agents_abi::BoxFuture<'a, IoResult<()>> {
        Box::pin(async move {
            self.node.capture_text(text);
            Ok(())
        })
    }

    fn retain_until_flushed(&self, owner: Box<dyn Send>) {
        self.node.retain_until_drained(Box::new(TextOutputOwner {
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
struct TextOutputOwner {
    next: Arc<dyn TextOutput>,
    owner: Option<Box<dyn Send>>,
}

impl Drop for TextOutputOwner {
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

    use lk_agents_test_support::text::BufferedTextNode;

    struct Destination;
    impl TextOutput for Destination {
        fn label(&self) -> &str {
            "destination"
        }
        fn capture_text<'a>(&'a self, _: &'a str) -> lk_agents_abi::BoxFuture<'a, IoResult<()>> {
            Box::pin(async { Ok(()) })
        }
        fn flush(&self) {}
    }

    #[test]
    fn a_buffered_text_node_keeps_the_segment_pending_until_its_drain_finishes() {
        let delivered = Arc::new(BufferedTextNode::default());
        let output = NodeTextOutput::new(delivered.clone(), Arc::new(Destination));
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
        let output = NodeTextOutput::new(delivered, Arc::new(Destination));
        output.flush();
        let mut completion = output.wait_for_flush();
        let mut cx = Context::from_waker(Waker::noop());
        assert!(matches!(completion.as_mut().poll(&mut cx), Poll::Pending));
        output.cancel_flush();
        assert!(matches!(completion.as_mut().poll(&mut cx), Poll::Ready(Ok(()))));
    }
}
