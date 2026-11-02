//! Async reader wakeups around voice's synchronous queue and playback state machine.

use lk_agents_abi::AudioFrame;
use lk_agents_abi::voice::avatar::AudioReceiverObservation;
use lk_agents_abi::voice::avatar_queue::{QueueClosed, QueueEffects};
use lk_agents_abi::voice::io::{AvatarPlaybackSegment, PlaybackSegment};
use lk_agents_voice::avatar::machine::queue::Queue;
use parking_lot::Mutex;
use tokio::sync::Notify;

pub struct QueueAudio {
    state: Mutex<Queue>,
    changed: Notify,
}

impl lk_agents_abi::voice::avatar_queue::QueueAudio for QueueAudio {
    fn new(wait_playback_start: bool) -> Self {
        Self { state: Mutex::new(Queue::new(wait_playback_start)), changed: Notify::new() }
    }
    fn check_open(&self) -> Result<(), QueueClosed> {
        self.state.lock().check_open()
    }
    fn capture(
        &self,
        frame: AudioFrame,
        segment: PlaybackSegment,
        epoch: u64,
    ) -> Result<QueueEffects, QueueClosed> {
        let result = self.state.lock().capture(frame, segment, epoch);
        self.changed.notify_waiters();
        result
    }
    fn flush(&self) -> Result<(), QueueClosed> {
        let result = self.state.lock().flush();
        self.changed.notify_waiters();
        result
    }
    fn clear(&self) -> QueueEffects {
        self.state.lock().clear()
    }
    fn playback_finished(
        &self,
        playback_position: f64,
        interrupted: bool,
        segment: Option<AvatarPlaybackSegment>,
    ) -> QueueEffects {
        self.state.lock().finished(playback_position, interrupted, segment)
    }
    fn playback_started(&self, segment: Option<AvatarPlaybackSegment>) -> QueueEffects {
        self.state.lock().started(segment)
    }
    async fn next_audio(&self) -> AudioReceiverObservation {
        loop {
            let changed = self.changed.notified();
            tokio::pin!(changed);
            changed.as_mut().enable();
            if let Some(observation) = self.state.lock().next_audio() {
                return observation;
            }
            changed.await;
        }
    }
    fn close(&self) -> QueueEffects {
        let effects = self.state.lock().close();
        self.changed.notify_waiters();
        effects
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lk_agents_abi::voice::avatar_queue::QueueAudio as _;

    #[tokio::test]
    async fn close_releases_pending_read_and_remains_terminal() {
        let queue = QueueAudio::new(false);
        let mut pending = Box::pin(queue.next_audio());
        assert!(
            std::future::poll_fn(|cx| std::task::Poll::Ready(
                pending.as_mut().poll(cx).is_pending()
            ))
            .await
        );
        assert!(queue.close().reset);
        assert_eq!(pending.await, (0, false, None, None));
        assert_eq!(queue.next_audio().await, (0, false, None, None));
        assert!(!queue.close().reset);
    }
}
