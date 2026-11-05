use super::*;
use lk_agents_abi::utils::clock::VirtualClock;
use parking_lot::Mutex;
use tokio::sync::{Notify, Semaphore};

/// What the plugin took: audio as its sample count, `None` for a flush.
type Taken = Arc<Mutex<Vec<Option<u32>>>>;

/// A plugin whose `_core_push` waits for a permit per entry.
struct Slow {
    taken: Taken,
    permits: Arc<Semaphore>,
    ended: Arc<Notify>,
}

impl StreamInput<AudioFrame> for Slow {
    fn push(&self, _: AudioFrame) -> Result<(), ProviderError> {
        unreachable!("the pump awaits each entry")
    }
    fn flush(&self) -> Result<(), ProviderError> {
        unreachable!("the pump awaits each entry")
    }
    fn end_input(&self) {
        unreachable!("the pump awaits each entry")
    }
    fn send(&self, entry: Entry<AudioFrame>) -> BoxFuture<'static, Result<(), ProviderError>> {
        let (taken, permits, ended) =
            (self.taken.clone(), self.permits.clone(), self.ended.clone());
        Box::pin(async move {
            permits.acquire().await.expect("never closed").forget();
            match entry {
                Entry::Item(frame) => taken.lock().push(Some(frame.samples_per_channel)),
                Entry::Flush => taken.lock().push(None),
                Entry::EndInput => ended.notify_one(),
            }
            Ok(())
        })
    }
}

/// No events; ends once the plugin's input ended.
struct Events(Arc<Notify>);

impl StreamOutput<VadEvent> for Events {
    fn next(&mut self) -> BoxFuture<'_, Result<Option<VadEvent>, ProviderError>> {
        Box::pin(async move {
            self.0.notified().await;
            Ok(None)
        })
    }
    fn close(self: Box<Self>, _: Ts) -> BoxFuture<'static, Closed> {
        Box::pin(async { Closed::Clean })
    }
}

impl ProviderStream<VadEvent> for Events {}

fn frame() -> AudioFrame {
    AudioFrame::new(vec![0u8; 320], 16_000, 1, 160)
}

#[tokio::test]
async fn a_stalled_plugin_never_fails_the_stream_and_resumes_on_current_audio() {
    let (taken, permits, ended) = (Taken::default(), Arc::new(Semaphore::new(0)), Arc::default());
    let plugin = Slow { taken: taken.clone(), permits: permits.clone(), ended: Arc::clone(&ended) };
    let (input, mut events) = PluginInput::wrap(
        DuplexStream::new(Box::new(plugin) as VadSink, Box::new(Events(ended)) as VadSource),
        &telemetry::Cx::NONE,
        Arc::new(VirtualClock::new()),
    )
    .split();
    // Two seconds of pre-connect audio, then three of live audio, while the plugin takes nothing.
    for _ in 0..200 {
        input.head(frame()).expect("head is kept");
    }
    for _ in 0..300 {
        input.push(frame()).expect("a stalled plugin never refuses audio");
    }
    input.end_input();
    permits.add_permits(1_000);
    assert!(events.next().await.expect("the stream does not fail").is_none());

    let taken = taken.lock();
    let gap = taken.iter().position(Option::is_none).expect("the dropped audio is a flush");
    let head: u32 = taken[..gap].iter().flatten().sum();
    let live: u32 = taken[gap + 1..].iter().flatten().sum();
    assert_eq!(head, 200 * 160, "the pre-connect audio arrives in full");
    assert!(live > 0 && live <= 8_000 + 160, "only the newest audio follows: {live} samples");
}
