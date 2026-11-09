use crate::telemetry;
use crate::voice::session::Timebase;
use crate::voice::session::Tx;
use lk_agents_abi::resources::ResourceId;
use lk_agents_abi::{Closed, Spawner, StreamOutput};
use lk_agents_voice::audio::{AudioChannel, ChannelEnd, Tagged};
use lk_agents_voice::trace::ChannelEndCause;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

struct TaggedSource<T> {
    stream: ResourceId,
    inner: Box<dyn StreamOutput<T>>,
}

impl<T: Send + 'static> StreamOutput<Tagged<T>> for TaggedSource<T> {
    fn next(
        &mut self,
    ) -> lk_agents_abi::BoxFuture<'_, Result<Option<Tagged<T>>, lk_agents_abi::errors::ProviderError>>
    {
        Box::pin(async move {
            Ok(self.inner.next().await?.map(|item| Tagged { stream: self.stream, item }))
        })
    }
    fn cancel_safe(&self) -> bool {
        self.inner.cancel_safe()
    }
    fn close(
        self: Box<Self>,
        deadline: lk_agents_abi::Ts,
    ) -> lk_agents_abi::BoxFuture<'static, Closed> {
        self.inner.close(deadline)
    }
}

impl<T: Send + 'static> lk_agents_abi::utils::stream::ProviderStream<Tagged<T>>
    for TaggedSource<T>
{
}

pub fn tagged<T: Send + 'static>(
    stream: ResourceId,
    source: Box<dyn StreamOutput<T>>,
) -> Box<dyn StreamOutput<Tagged<T>>> {
    Box::new(TaggedSource { stream, inner: source })
}

#[derive(Clone)]
pub struct VideoForwarding {
    pub closing: CancellationToken,
    /// The loop's queue: the frames (the sampler runs in the loop, on the session clock) and the
    /// source's end.
    pub queue: Tx,
}

impl VideoForwarding {
    pub async fn pump(self, input: Arc<dyn lk_agents_abi::voice::io::VideoInput>) {
        let p = self;
        let cause = loop {
            let frame = tokio::select! {
                biased;
                frame = input.next_frame(&telemetry::Cx::NONE) => match frame {
                    Ok(Some(frame)) => frame,
                    Ok(None) => break ChannelEndCause::SourceEnded,
                    Err(e) => {
                        log::warn!(target: lk_agents_abi::LOG_TARGET, error = e.to_string().as_str(); "video input failed");
                        break ChannelEndCause::Error;
                    }
                },
                () = p.closing.cancelled() => break ChannelEndCause::Closing,
            };
            // A queue that is gone is a loop that has returned: nothing to report to.
            if p.queue.send(frame).is_err() {
                return;
            }
        };
        let leg = lk_agents_voice::trace::Leg::VideoIn;
        let end = ChannelEnd { leg, cause, sink: None, closed: None, error: None };
        let _ = p.queue.send(end);
    }
}

/// A drain the session may drop before its stream ends: on `token`, the stream is closed, and
/// only then is its sink released from `resources`, so its input never ends while the provider
/// still listens and nothing it heard is finalized.
pub struct Abort {
    pub token: CancellationToken,
    pub resources: crate::voice::resources::Resources,
}

/// Drain a provider source into the loop, close it once, then report the end with the stream's
/// sink, which the loop removes if it is still installed: a sink of a closed stream would raise on
/// every frame for the rest of the call.
pub fn spawn_drain<T: Into<crate::voice::VoiceRuntimeEvent> + Send + 'static>(
    (tasks, spawner): (&Arc<lk_agents_net::task::TaskSet>, &dyn Spawner),
    time: Timebase,
    (closing, abort): (CancellationToken, Abort),
    mut source: Box<dyn StreamOutput<T>>,
    queue: Tx,
    (channel, sink): (AudioChannel, ResourceId),
) {
    tasks.spawn(spawner, "leg drain", async move {
        let (cause, error) = forward_items(&mut *source, &closing, &abort.token, &queue).await;
        // Always, bounded on the session clock (dropping doesn't cancel a host stream).
        let deadline = time.now().saturating_add(lk_agents_abi::HOST_TEARDOWN_TIMEOUT);
        let closed = source.close(deadline).await;
        if cause == ChannelEndCause::Aborted {
            abort.resources.release(sink);
        }
        // Done: the session prunes the abort of a drain that finished.
        abort.token.cancel();
        let leg = channel.leg();
        let _ =
            queue.send(ChannelEnd { leg, cause, sink: Some(sink), closed: Some(closed), error });
    });
}

/// Forwards the source's items to the loop until it ends, fails, the loop is gone or the session
/// closes.
async fn forward_items<T: Into<crate::voice::VoiceRuntimeEvent> + Send + 'static>(
    source: &mut dyn StreamOutput<T>,
    closing: &CancellationToken,
    abort: &CancellationToken,
    queue: &Tx,
) -> (ChannelEndCause, Option<lk_agents_abi::errors::ProviderError>) {
    loop {
        // `biased` so an arrived item is delivered before a simultaneous close.
        let next = tokio::select! {
            biased;
            item = source.next() => item,
            () = closing.cancelled() => return (ChannelEndCause::Closing, None),
            () = abort.cancelled() => return (ChannelEndCause::Aborted, None),
        };
        match next {
            Ok(Some(item)) => {
                if queue.send(item).is_err() {
                    return (ChannelEndCause::Closing, None); // the loop is gone
                }
            }
            Ok(None) => return (ChannelEndCause::SourceEnded, None),
            Err(e) => return (ChannelEndCause::Error, Some(e)),
        }
    }
}

/// A stream's retried failures, sent to the session loop.
pub fn report_to(queue: Tx) -> lk_agents_abi::RetryReport {
    Arc::new(move |error, _, _| {
        let _ = queue.send(lk_agents_voice::agent_session::VoiceEvent::SttRetried(error));
    })
}
