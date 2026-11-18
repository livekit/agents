use super::super::tests::{close_session, no_room, running, unobserved};
use super::*;
use lk_agents_abi::voice::io::{
    AudioOutput, AudioOutputCapabilities, PlaybackFinished, PlaybackSegment, PlaybackTracker,
};
use lk_agents_abi::voice::session::VoiceSession as _;
use lk_agents_net::signal;
use std::time::Duration;

struct HeldPlayback {
    tracker: Arc<dyn PlaybackTracker>,
    /// The operation key belongs to capture; flush clears the tracker's current selection.
    captured: signal::Producer<Option<PlaybackSegment>>,
    flushed: signal::Producer<Option<PlaybackSegment>>,
}

impl HeldPlayback {
    fn new() -> Self {
        Self {
            tracker: Arc::new(crate::voice::playback::Tracker::new()),
            captured: signal::signal(None).0,
            flushed: signal::signal(None).0,
        }
    }

    fn finish(&self, interrupted: bool) {
        let segment = self.captured.get().expect("the real output captured this operation key");
        assert!(self.tracker.playback_finished(
            segment,
            PlaybackFinished {
                playback_position: if interrupted {
                    Duration::ZERO
                } else {
                    Duration::from_millis(10)
                },
                interrupted,
                synchronized_transcript: Some(
                    if interrupted { "" } else { "Reference ABC." }.into()
                ),
            }
        ));
    }
}

impl AudioOutput for HeldPlayback {
    fn label(&self) -> &str {
        "held exact-output playback"
    }
    fn tracker(&self) -> &Arc<dyn PlaybackTracker> {
        &self.tracker
    }
    fn capabilities(&self) -> AudioOutputCapabilities {
        Default::default()
    }
    fn capture<'a>(
        &'a self,
        _: &lk_agents_abi::telemetry::Cx,
        _: &'a lk_agents_abi::AudioFrame,
    ) -> lk_agents_abi::BoxFuture<'a, lk_agents_abi::voice::io::Result<()>> {
        Box::pin(async move {
            let segment: PlaybackSegment =
                self.tracker.current_segment().expect("admitted capture");
            self.captured.set(Some(segment));
            assert!(self.tracker.playback_started(segment, lk_agents_abi::Ts::ZERO));
            Ok(())
        })
    }
    fn flush_segment(&self) {
        if let Some(segment) = self.captured.get()
            && self.tracker.has_pending_segment(segment)
        {
            self.flushed.set(Some(segment));
        }
    }
    fn clear_buffer(&self) {
        if let Some(segment) = self.captured.get()
            && self.tracker.has_pending_segment(segment)
        {
            self.finish(true);
        }
    }
}

async fn boundary<T>(what: &str, work: impl std::future::Future<Output = T>) -> T {
    tokio::time::timeout(Duration::from_secs(1), work).await.unwrap_or_else(|_| panic!("{what}"))
}

#[tokio::test(start_paused = true)]
async fn exact_delivery_waits_for_the_real_output_playback_receipt() {
    for (interrupted, lost_reader) in [(false, false), (true, false), (false, true)] {
        let api = running(unobserved(no_room(), None)).await;
        let sink = Arc::new(HeldPlayback::new());
        api.set_audio_output(Some(sink.clone())).unwrap();
        let (speech, frames) =
            api.say_audio_channel("Reference ABC.".into(), None, true, None).unwrap();
        assert!(frames.push(lk_agents_abi::AudioFrame::new(vec![0; 320], 16_000, 1, 160)).await);
        frames.end();
        boundary(
            "the real say captured and flushed its latched playback key",
            sink.flushed.cell().wait_for(Option::is_some),
        )
        .await
        .expect("the held output still owns its flush gate");
        let mut receipt = Box::pin(speech.exact_delivery("Reference ABC."));
        assert!(
            futures_util::poll!(receipt.as_mut()).is_pending(),
            "audio accepted and flushed is not yet audio played"
        );
        let result = if lost_reader {
            drop(receipt);
            assert!(!speech.interrupted(), "losing a reader cannot interrupt the native speech");
            sink.finish(false);
            boundary(
                "native exact delivery survived the dropped reader and consumed its keyed finish",
                speech.exact_delivery("Reference ABC."),
            )
            .await
        } else {
            if interrupted {
                lk_agents_abi::voice::speech_handle::SpeechHandle::interrupt(&*speech, true)
                    .unwrap();
            } else {
                sink.finish(false);
            }
            boundary(
                "native exact delivery consumed the keyed playback or interruption receipt",
                receipt,
            )
            .await
        };
        if interrupted {
            assert!(
                !result.unwrap_err().cancelled(),
                "presentation cannot cancel the business task"
            );
        } else {
            result.unwrap();
            let items = speech.chat_items();
            assert_eq!(items.len(), 1, "exact output keeps one canonical history item");
            assert_eq!(
                items[0].as_message().unwrap().raw_text_content().as_deref(),
                Some("Reference ABC.")
            );
        }
        close_session(&api).await;
    }
}
