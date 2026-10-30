//! The binding boundary under contention: a Python thread holding the GIL calls a generated
//! synchronous entry whose Rust side waits for a lock, while the Rust task holding that lock waits
//! for the GIL (review13's deadlock class, decision 129).

use std::sync::{Arc, mpsc};
use std::time::Duration;

use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::{BoxFuture, StreamSink, StreamSource, SynthesizedAudio, Tts, TtsCapabilities};
use lk_agents_abi::{Closed, Ts, TtsSink, TtsSource};
use lk_agents_runtime::inference::fallback::{ChainPolicy, FallbackTts, IgnoreEvents};
use lk_agents_runtime::inference::tts::stream::TtsStream;
use pyo3::prelude::*;

/// A host TTS as the fallback chain sees it: its stream opens once `opened` fires, and its sink
/// takes the GIL on every push, as a host sink scheduling its drain does.
struct HostLikeTts {
    opened: tokio::sync::watch::Receiver<bool>,
    pushing: mpsc::SyncSender<()>,
}

struct HostLikeSink(mpsc::SyncSender<()>);

impl StreamSink<String> for HostLikeSink {
    fn push(&self, _item: String) -> Result<(), ProviderError> {
        let _ = self.0.try_send(());
        Python::attach(|_| ());
        Ok(())
    }
    fn flush(&self) -> Result<(), ProviderError> {
        Ok(())
    }
    fn end_input(&self) {}
}

struct Silent;

impl StreamSource<SynthesizedAudio> for Silent {
    fn next(&mut self) -> BoxFuture<'_, Result<Option<SynthesizedAudio>, ProviderError>> {
        Box::pin(std::future::pending())
    }
    fn close(self: Box<Self>, _deadline: Ts) -> BoxFuture<'static, Closed> {
        Box::pin(async { Closed::Clean })
    }
}

impl Tts for HostLikeTts {
    fn label(&self) -> String {
        "host-like".into()
    }
    fn capabilities(&self) -> TtsCapabilities {
        TtsCapabilities { streaming: true, aligned_transcript: false }
    }
    fn sample_rate(&self) -> u32 {
        24_000
    }
    fn num_channels(&self) -> u16 {
        1
    }
    fn synthesize<'a>(&'a self, _text: &'a str) -> BoxFuture<'a, Result<TtsSource, ProviderError>> {
        unreachable!("the test only streams")
    }
    fn stream(
        &self,
        _expressive: bool,
    ) -> BoxFuture<'_, Result<(TtsSink, TtsSource), ProviderError>> {
        let mut opened = self.opened.clone();
        let pushing = self.pushing.clone();
        Box::pin(async move {
            opened.wait_for(|opened| *opened).await.unwrap();
            Ok((Box::new(HostLikeSink(pushing)) as TtsSink, Box::new(Silent) as TtsSource))
        })
    }
}

// Catches: a generated sync entry (`TtsStream.push_text`) keeping the GIL while it waits for the
// fallback's token log, whose holder (the chain pump replaying text into a host TTS) waits for
// the GIL.
#[test]
fn pushing_text_while_the_fallback_replays_into_a_host_tts_does_not_deadlock() {
    Python::initialize();
    let (open, opened) = tokio::sync::watch::channel(false);
    let (pushing, pushed) = mpsc::sync_channel(1);
    let child: Arc<dyn Tts> = Arc::new(HostLikeTts { opened, pushing });
    let runtime = lk_agents_runtime::process::handle().unwrap();
    let fallback = FallbackTts::new(
        vec![child],
        None,
        ChainPolicy::tts_defaults(),
        lk_agents_runtime::process::host_clock(),
        lk_agents_runtime::process::host_spawner().unwrap(),
        Arc::new(IgnoreEvents),
    )
    .unwrap();
    let (sink, source) = runtime.block_on(fallback.stream(false)).unwrap();
    let stream = Arc::new(TtsStream::new(Some(sink), source));
    lk_agents_abi::tts::streams::TtsStream::push_text(&*stream, "replayed".into()).unwrap();

    let (attached, attaching) = mpsc::channel();
    let (done, finished) = mpsc::channel();
    std::thread::spawn(move || {
        Python::attach(|py| {
            let stream = Py::new(py, crate::generated::tts::TtsStream(stream)).unwrap();
            attached.send(()).unwrap();
            // The pump now holds the log, replaying into the host TTS, which waits for the GIL.
            pushed.recv_timeout(Duration::from_secs(10)).expect("the pump replays");
            stream.bind(py).call_method1("push_text", ("live",)).unwrap();
        });
        done.send(()).unwrap();
    });
    attaching.recv().unwrap();
    open.send_replace(true);
    assert!(finished.recv_timeout(Duration::from_secs(10)).is_ok(), "deadlocked");
}
