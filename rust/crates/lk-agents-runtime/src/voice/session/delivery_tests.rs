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
    /// The speaker breaks on its first frame.
    broken: bool,
    /// The operation key belongs to capture; flush clears the tracker's current selection.
    captured: signal::Producer<Option<PlaybackSegment>>,
    flushed: signal::Producer<Option<PlaybackSegment>>,
}

impl HeldPlayback {
    fn new() -> Self {
        Self {
            tracker: Arc::new(crate::voice::playback::Tracker::new()),
            broken: false,
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
            if self.broken {
                return Err(lk_agents_abi::voice::io::IoError::Rtc("the speaker broke".into()));
            }
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

/// A session with `sink` as its speaker, saying "Reference ABC." as exact output, its audio
/// captured and flushed: what remains is the speaker's playout.
async fn saying(
    sink: Arc<HeldPlayback>,
) -> (VoiceSession, Arc<crate::voice::speech_handle::SpeechHandle>) {
    let api = running(unobserved(no_room(), None)).await;
    api.set_audio_output(Some(sink.clone())).unwrap();
    let (speech, frames) =
        api.say_audio_channel("Reference ABC.".into(), None, true, None).unwrap();
    assert!(frames.push(lk_agents_abi::AudioFrame::new(vec![0; 320], 16_000, 1, 160)).await);
    frames.end();
    if !sink.broken {
        boundary(
            "the real say captured and flushed its latched playback key",
            sink.flushed.cell().wait_for(Option::is_some),
        )
        .await
        .expect("the held output still owns its flush gate");
    }
    (api, speech)
}

#[tokio::test(start_paused = true)]
async fn exact_delivery_waits_for_the_real_output_playback_receipt() {
    for (interrupted, lost_reader) in [(false, false), (true, false), (false, true)] {
        let sink = Arc::new(HeldPlayback::new());
        let (api, speech) = saying(sink.clone()).await;
        let mut receipt = Box::pin(speech.exact_delivery("Reference ABC."));
        assert!(
            futures_util::poll!(receipt.as_mut()).is_pending(),
            "audio accepted and flushed is not yet audio played"
        );
        let outcome = if lost_reader {
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
            assert_eq!(outcome, SayOutcome::Interrupted);
        } else {
            assert_eq!(outcome, SayOutcome::Delivered);
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

// Catches: a session closed mid-playout leaving the exact output's waiter parked, or reading the
// close as a failed output.
#[tokio::test(start_paused = true)]
async fn closing_the_session_mid_playout_settles_exact_output_as_interrupted() {
    let sink = Arc::new(HeldPlayback::new());
    let (api, speech) = saying(sink).await;
    let receipt = speech.exact_delivery("Reference ABC.");
    close_session(&api).await;
    assert_eq!(
        boundary("the close settled the exact waiter", receipt).await,
        SayOutcome::Interrupted
    );
    assert!(
        speech
            .chat_items()
            .iter()
            .all(|item| { item.as_message().is_none_or(|message| message.interrupted) }),
        "unheard words are never recorded as heard"
    );
}

// Catches: a broken speaker reported as delivered or as an interruption, or its waiter parked.
#[tokio::test(start_paused = true)]
async fn a_failed_speaker_settles_exact_output_as_failed() {
    let sink = Arc::new(HeldPlayback { broken: true, ..HeldPlayback::new() });
    let (api, speech) = saying(sink).await;
    let outcome = boundary(
        "the output failure settled the exact waiter",
        speech.exact_delivery("Reference ABC."),
    )
    .await;
    assert_eq!(outcome, SayOutcome::Failed);
    close_session(&api).await;
}
