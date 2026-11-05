use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Duration;

use lk_agents_abi::errors::{HostError, ProviderError};
use lk_agents_abi::tts::stream_pacer::{HostSentenceStream, PushedAudio};
use parking_lot::Mutex;
use tokio::sync::Notify;

use super::*;

#[test]
fn first_sentence_is_immediate_then_wait_for_audio_and_batch() {
    let mut p = SentencePacer::new(5.0, 4);
    p.push("hello".into());
    p.push("你好".into());
    p.push("ab".into());
    p.end();
    assert_eq!(p.poll(100.0, 0.0).text.as_deref(), Some("hello"));
    assert_eq!(p.poll(101.0, 10.0).text, None);
    assert_eq!(p.poll(102.0, 10.0).text, None);
    assert_eq!(p.poll(106.0, 10.0).text.as_deref(), Some("你好 ab"));
    assert_eq!(p.poll(107.0, 10.0).delay, None);
}

/// A host sentence stream: sentences queued by the test, the end once `ended`.
#[derive(Default)]
struct Host {
    queued: Mutex<VecDeque<String>>,
    ended: Mutex<bool>,
    closed: Mutex<bool>,
    changed: Notify,
}

impl Host {
    fn send(&self, sentence: &str) {
        self.queued.lock().push_back(sentence.into());
        self.changed.notify_waiters();
    }
    fn end(&self) {
        *self.ended.lock() = true;
        self.changed.notify_waiters();
    }
}

impl HostSentenceStream for Host {
    fn push_text(&self, _text: String) {}
    fn flush(&self) {}
    fn end_input(&self) {}
    fn next(
        &self,
        _cx: &telemetry::Cx,
    ) -> lk_agents_abi::BoxFuture<'_, Result<Option<String>, ProviderError>> {
        Box::pin(async move {
            loop {
                let changed = self.changed.notified();
                if let Some(sentence) = self.queued.lock().pop_front() {
                    return Ok(Some(sentence));
                }
                if *self.ended.lock() {
                    return Ok(None);
                }
                changed.await;
            }
        })
    }
    fn aclose(&self, _cx: &telemetry::Cx) -> lk_agents_abi::BoxFuture<'_, Result<(), HostError>> {
        *self.closed.lock() = true;
        Box::pin(async { Ok(()) })
    }
}

/// The emitter's pushed audio, set by the test.
#[derive(Default)]
struct Audio(Mutex<f64>);

impl PushedAudio for Audio {
    fn pushed_duration(&self) -> f64 {
        *self.0.lock()
    }
}

fn paced(host: &Arc<Host>, audio: &Arc<Audio>) -> PacedSentences {
    let env = ProcessEnv::current_runtime();
    PacedSentences::new(&env, host.clone(), audio.clone(), 5.0, 300)
}

#[tokio::test(start_paused = true)]
async fn the_first_sentence_goes_at_once_and_the_rest_wait_for_the_audio_to_play_out() {
    let (host, audio) = (Arc::new(Host::default()), Arc::new(Audio::default()));
    let paced = Arc::new(paced(&host, &audio));
    host.send("One.");
    host.send("Two.");
    host.send("Three.");
    let start = tokio::time::Instant::now();
    assert_eq!(paced.next(telemetry::Cx::NONE).await.unwrap().as_deref(), Some("One."));
    assert_eq!(start.elapsed(), Duration::ZERO);

    // Eight seconds of audio arrive at once: the rest waits until five remain.
    *audio.0.lock() = 8.0;
    let next = tokio::spawn({
        let paced = paced.clone();
        async move { paced.next(telemetry::Cx::NONE).await }
    });
    host.end();
    let batch = next.await.unwrap().unwrap();
    assert_eq!(batch.as_deref(), Some("Two. Three."));
    let waited = start.elapsed().as_secs_f64();
    assert!((3.0..3.7).contains(&waited), "released after {waited}s");
    assert_eq!(paced.next(telemetry::Cx::NONE).await.unwrap(), None);
}

#[tokio::test(start_paused = true)]
async fn aclose_ends_a_pending_next_and_closes_the_host_stream() {
    let (host, audio) = (Arc::new(Host::default()), Arc::new(Audio::default()));
    let paced = Arc::new(paced(&host, &audio));
    let next = tokio::spawn({
        let paced = paced.clone();
        async move { paced.next(telemetry::Cx::NONE).await }
    });
    tokio::time::sleep(Duration::from_secs(1)).await;
    paced.aclose(telemetry::Cx::NONE).await;
    assert_eq!(next.await.unwrap().unwrap(), None);
    assert!(*host.closed.lock());
}
