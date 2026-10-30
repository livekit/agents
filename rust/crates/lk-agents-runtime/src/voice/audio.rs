//! Provider drains and camera forwarding owned by the session runtime.

use crate::voice::session::Timebase;
use crate::voice::session::Tx;
use lk_agents_abi::resources::ResourceId;
use lk_agents_abi::{Closed, Spawner, StreamSource};
use lk_agents_voice::audio::{AudioChannel, ChannelEnd, Tagged};
use lk_agents_voice::trace::ChannelEndCause;
use std::sync::Arc;
const VIDEO_IN_LEG: &str = "video_in";

struct TaggedSource<T> {
    stream: ResourceId,
    inner: Box<dyn StreamSource<T>>,
}

impl<T: Send + 'static> StreamSource<Tagged<T>> for TaggedSource<T> {
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

pub fn tagged<T: Send + 'static>(
    stream: ResourceId,
    source: Box<dyn StreamSource<T>>,
) -> Box<dyn StreamSource<Tagged<T>>> {
    Box::new(TaggedSource { stream, inner: source })
}

#[derive(Clone)]
pub struct VideoForwarding {
    pub closing: tokio::sync::watch::Receiver<bool>,
    /// The loop's queue: the frames (the sampler runs in the loop, on the session clock) and the
    /// source's end.
    pub queue: Tx,
}

impl VideoForwarding {
    pub async fn pump(self, input: Arc<dyn lk_agents_abi::voice::io::VideoInput>) {
        let mut p = self;
        let cause = loop {
            let frame = tokio::select! {
                biased;
                frame = input.next_frame() => match frame {
                    Ok(Some(frame)) => frame,
                    Ok(None) => break ChannelEndCause::SourceEnded,
                    Err(e) => {
                        log::warn!(target: "livekit.agents", error = e.to_string().as_str(); "video input failed");
                        break ChannelEndCause::Error;
                    }
                },
                _ = p.closing.wait_for(|c| *c) => break ChannelEndCause::Closing,
            };
            // A queue that is gone is a loop that has returned: nothing to report to.
            if p.queue.send(frame).is_err() {
                return;
            }
        };
        let end = ChannelEnd { leg: VIDEO_IN_LEG, cause, sink: None, closed: None, error: None };
        let _ = p.queue.send(end);
    }
}

/// Drain a provider source into the loop, close it once, then report the end with the stream's
/// sink, which the loop removes if it is still installed: a sink of a closed stream would raise on
/// every frame for the rest of the call.
pub fn spawn_drain<T: Into<lk_agents_voice::agent_session::Event> + Send + 'static>(
    (tasks, spawner): (&Arc<lk_agents_net::task::TaskSet>, &dyn Spawner),
    time: Timebase,
    mut closing: lk_agents_abi::BoxFuture<'static, ()>,
    mut source: Box<dyn StreamSource<T>>,
    queue: Tx,
    (channel, sink): (AudioChannel, ResourceId),
) {
    tasks.spawn(spawner, "leg drain", async move {
        let mut error = None;
        let cause = loop {
            // `biased` so an arrived item is delivered before a simultaneous close.
            let next = tokio::select! {
                biased;
                item = source.next() => Some(item),
                _ = &mut closing => None,
            };
            match next {
                Some(Ok(Some(item))) => {
                    if queue.send(item).is_err() {
                        break ChannelEndCause::Closing; // the loop is gone
                    }
                }
                Some(Ok(None)) => break ChannelEndCause::SourceEnded,
                Some(Err(e)) => {
                    error = Some(e);
                    break ChannelEndCause::Error;
                }
                None => break ChannelEndCause::Closing,
            }
        };
        // Always, bounded on the session clock (dropping doesn't cancel a host stream).
        let deadline = time.now().saturating_add(lk_agents_abi::HOST_TEARDOWN_TIMEOUT);
        let closed = source.close(deadline).await;
        let leg = channel.label();
        let _ =
            queue.send(ChannelEnd { leg, cause, sink: Some(sink), closed: Some(closed), error });
    });
}

/// A stream's retried failures, sent to the session loop.
pub fn report_to(queue: Tx) -> lk_agents_abi::RetryReport {
    Arc::new(move |error, _, _| {
        let _ = queue.send(lk_agents_voice::agent_session::Input::SttRetried(error));
    })
}

/// A running session's closing signal, owned by one provider drain.
pub fn closing(
    mut closing: tokio::sync::watch::Receiver<bool>,
) -> lk_agents_abi::BoxFuture<'static, ()> {
    Box::pin(async move {
        let _ = closing.wait_for(|closing| *closing).await;
    })
}
