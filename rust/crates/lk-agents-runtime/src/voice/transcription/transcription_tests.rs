#![allow(
    clippy::disallowed_types,
    clippy::disallowed_methods,
    reason = "tests drive tokio's sync primitives directly"
)]
use super::*;
use lk_agents_abi::Ts;
use std::time::Duration;
use tokio::sync::watch;

/// A physical output that latches each frame's segment key; the test reports its playback.
struct KeyedPlayback(Arc<dyn PlaybackTracker>, parking_lot::Mutex<Option<PlaybackSegment>>);

impl Default for KeyedPlayback {
    fn default() -> Self {
        Self(Arc::new(crate::voice::playback::Tracker::new()), Default::default())
    }
}

impl AudioOutput for KeyedPlayback {
    fn label(&self) -> &str {
        "keyed playback"
    }
    fn tracker(&self) -> &Arc<dyn PlaybackTracker> {
        &self.0
    }
    fn capabilities(&self) -> lk_agents_abi::voice::io::AudioOutputCapabilities {
        lk_agents_abi::voice::io::AudioOutputCapabilities { pause: true }
    }
    fn capture<'a>(
        &'a self,
        _cx: &lk_agents_abi::telemetry::Cx,
        _: &'a AudioFrame,
    ) -> lk_agents_abi::BoxFuture<'a, lk_agents_abi::voice::io::Result<()>> {
        Box::pin(async move {
            *self.1.lock() = self.0.current_segment();
            Ok(())
        })
    }
    fn flush_segment(&self) {}
    fn clear_buffer(&self) {}
}

/// The words the synchroniser released, with where each one's audio starts.
struct Shown(watch::Sender<Vec<(String, f64)>>);

impl PlaybackObserver for Shown {
    fn on_playback_event(&self, _ev: &PlaybackEvent) {}

    fn on_transcript_shown(
        &self,
        _: Option<lk_agents_abi::voice::io::PlaybackSegment>,
        word: &TranscriptChunk,
    ) {
        let start = word.start_time.unwrap_or_default();
        self.0.send_modify(|shown| shown.push((word.text.clone(), start)));
    }
}

struct Chain {
    sync: Arc<TranscriptSynchronizer>,
    leaf: Arc<dyn PlaybackTracker>,
    shown: watch::Sender<Vec<(String, f64)>>,
    output: Arc<KeyedPlayback>,
}

/// A transcript sink below the synchroniser that refuses every word, counting the refusals.
struct RefusingText(watch::Sender<usize>);

impl TranscriptionOutput for RefusingText {
    fn label(&self) -> &str {
        "refusing text"
    }
    fn capture_text<'a>(
        &'a self,
        _cx: &lk_agents_abi::telemetry::Cx,
        _: &'a str,
    ) -> lk_agents_abi::BoxFuture<'a, Result<()>> {
        self.0.send_modify(|refused| *refused += 1);
        Box::pin(async { Err(IoError::TranscriptionCapacity("text".into())) })
    }
    fn flush(&self) {}
}

impl Chain {
    fn new() -> Chain {
        Chain::with_text(None)
    }

    fn with_text(text_next: Option<Arc<dyn TranscriptionOutput>>) -> Chain {
        let output = Arc::new(KeyedPlayback::default());
        let leaf = output.tracker().clone();
        let sync = TranscriptSynchronizer::new(TranscriptSyncConfig {
            audio_next: output.clone() as Arc<dyn AudioOutput>,
            text_next,
            speed: 1.0,
            clock: Arc::new(lk_agents_net::clock::RealClock::new()),
            spawner: Arc::new(lk_agents_net::task::TokioSpawner::current()),
        });
        // As the session's first activation does: both legs attached open the gate.
        sync.audio_output().on_attached();
        sync.transcription_output().on_attached();
        let shown = watch::Sender::new(Vec::new());
        sync.audio_output().tracker().observe(Arc::new(Shown(shown.clone())));
        Chain { sync, leaf, shown, output }
    }

    async fn text(&self, text: &str) {
        let out = self.sync.transcription_output();
        out.capture_text(&lk_agents_abi::telemetry::Cx::NONE, text).await.unwrap();
        out.flush();
    }

    /// Words the voice aligned: each with where its audio starts and ends.
    async fn aligned(&self, words: &[(&str, f64, f64)]) {
        let out = self.sync.transcription_output();
        for (word, start, end) in words {
            let text = TranscriptChunk {
                start_time: Some(*start),
                end_time: Some(*end),
                ..TranscriptChunk::new(*word)
            };
            out.capture_timed(&lk_agents_abi::telemetry::Cx::NONE, &text).await.unwrap();
        }
        out.flush();
    }

    /// `secs` of audio in 100 ms frames, then the segment's flush.
    async fn audio(&self, secs: u32) {
        let out = self.sync.audio_output();
        for _ in 0..secs * 10 {
            out.capture_frame(&AudioFrame::new(vec![0u8; 4800], 24_000, 1, 2_400)).await.unwrap();
        }
        out.flush();
    }

    fn key(&self) -> PlaybackSegment {
        self.output.1.lock().expect("the physical output accepted a frame")
    }

    fn played(&self, offset_ms: u64, ms: u64) {
        let at = Duration::from_millis;
        self.leaf.playback_progressed(self.key(), Ts::ZERO, at(offset_ms), at(ms));
    }

    fn finished(&self, at_ms: u64) {
        let playback_position = Duration::from_millis(at_ms);
        self.leaf.playback_finished(
            self.key(),
            PlaybackFinished {
                playback_position,
                interrupted: false,
                synchronized_transcript: None,
            },
        );
    }

    /// What was shown since the last call.
    fn take_shown(&self) -> Vec<(String, f64)> {
        self.shown.send_replace(Vec::new())
    }
}

// Catches: switching synchronisation off and on while a segment played out closed that segment's
// pacer and opened a fresh one that the old segment's playback then moved; the next segment's
// words joined it and were all shown at its first frame, seconds before their audio.
#[tokio::test]
async fn a_segment_closed_while_playing_paces_nothing_after_it() {
    let c = Chain::new();
    c.text("one two three four five six").await;
    c.audio(2).await;
    c.leaf.playback_started(c.key(), Ts::ZERO);
    c.played(0, 500);
    let audio = c.sync.audio_output();
    audio.on_detached();
    audio.on_attached();
    c.played(500, 1_500);
    c.finished(2_000);
    c.take_shown();

    c.aligned(&[("alpha ", 0.0, 0.5), ("bravo ", 0.5, 1.0), ("charlie ", 1.0, 1.5)]).await;
    c.audio(2).await;
    c.leaf.playback_started(c.key(), Ts::ZERO);
    let early: Vec<_> = c.take_shown().into_iter().filter(|(_, start)| *start > 1e-9).collect();
    assert!(early.is_empty(), "shown before their audio: {early:?}");
}

// Catches: a segment whose text ended with no audio left open (the next speech's audio paired
// with it) or cut with nothing shown, though its text is the speech's history: it is shown whole
// at once, and the next segment paces on its own.
#[tokio::test]
async fn a_segment_with_no_audio_shows_its_text_and_pairs_with_nothing() {
    let c = Chain::new();
    c.text("Ref one.").await;
    c.sync.audio_output().flush();
    let shown: String = c.take_shown().into_iter().map(|(w, _)| w).collect();
    assert_eq!(shown, "Ref one.");

    // The words of a segment ended with no audio may arrive after the next one's audio began
    // (through a `transcription_node`): they are still that segment's.
    c.sync.audio_output().flush();
    let out = c.sync.audio_output();
    out.capture_frame(&AudioFrame::new(vec![0u8; 4800], 24_000, 1, 2_400)).await.unwrap();
    c.text("Ref two.").await;
    let shown: String = c.take_shown().into_iter().map(|(w, _)| w).collect();
    assert_eq!(shown, "Ref two.");
    c.text("after words").await;
    c.audio(1).await;
    c.leaf.playback_started(c.key(), Ts::ZERO);
    let mut shown = c.take_shown();
    assert!(shown.iter().all(|(_, start)| *start <= 1e-9), "{shown:?}");
    c.played(0, 1_100);
    c.finished(1_100);
    shown.extend(c.take_shown());
    let shown: String = shown.into_iter().map(|(w, _)| w).collect();
    assert_eq!(shown.split_whitespace().collect::<Vec<_>>(), ["after", "words"]);
}

// Catches: a word the transcript sink below refused dropped silently, so the speech reported its
// transcript delivered; the refusal is what the segment's flush wait answers.
#[tokio::test]
async fn a_word_the_sink_below_refused_fails_the_segments_flush() {
    let (refusals, mut refused) = watch::channel(0);
    let c = Chain::with_text(Some(Arc::new(RefusingText(refusals))));
    c.text("Refused.").await;
    c.sync.audio_output().flush();
    refused.wait_for(|refused| *refused > 0).await.unwrap();
    let flushed = c.sync.transcription_output().wait_for_flush().await;
    assert!(matches!(flushed, Err(IoError::TranscriptionCapacity(_))), "{flushed:?}");
    assert!(c.sync.transcription_output().wait_for_flush().await.is_ok(), "reported once");
}

// Catches: a speech cut before any audio leaves its segment cleared with no
// audio end owed in practice; the next unvoiced speech's audio end was absorbed by it, so that
// speech's words stayed open and paired with the following speech's audio.
#[tokio::test]
async fn a_cut_wordy_segment_does_not_take_the_next_ones_audio_end() {
    let c = Chain::new();
    c.text("cut before its voice").await;
    c.sync.audio_output().clear_buffer();
    let _ = c.take_shown();
    c.text("Ref two.").await;
    c.sync.audio_output().flush();
    let shown: String = c.take_shown().into_iter().map(|(w, _)| w).collect();
    assert_eq!(shown, "Ref two.");
}

// Catches: the synchroniser still capturing after the output below reported its open segment cut
// and reset. The speech's later frames opened no segment of their own, so its wait returned at
// once with the cut and its words were shown under the next speech.
#[tokio::test]
async fn frames_after_a_sink_reset_open_their_own_segment() {
    let c = Chain::new();
    let out = c.sync.audio_output();
    let frame = AudioFrame::new(vec![0u8; 4800], 24_000, 1, 2_400);
    out.capture_frame(&frame).await.unwrap();
    c.leaf.playback_finished(c.key(), PlaybackFinished { interrupted: true, ..Default::default() });
    c.leaf.reset();
    out.capture_frame(&frame).await.unwrap();
    out.flush();
    let mut wait = std::pin::pin!(out.wait_for_playout());
    let mut cx = std::task::Context::from_waker(std::task::Waker::noop());
    assert!(
        wait.as_mut().poll(&mut cx).is_pending(),
        "the wait returned before the segment after the reset played"
    );
    c.finished(100);
    assert!(!out.wait_for_playout().await.interrupted);
}

// Catches: a sink that cut and reset its playout while the segment's audio was still coming threw
// the segment's words away. The audio after the reset paced no text, so the speech's transcript
// fell back to every word forwarded, and aligned words after it were shown off their stamps.
#[tokio::test]
async fn a_playout_reset_mid_segment_plays_on_with_its_words() {
    let c = Chain::new();
    let out = c.sync.audio_output();
    let frame = AudioFrame::new(vec![0u8; 4800], 24_000, 1, 2_400);
    let words = [("one ", 0.0, 0.5), ("two ", 0.5, 1.0), ("three ", 1.0, 1.5), ("four", 1.5, 2.0)];
    c.aligned(&words).await;
    for _ in 0..10 {
        out.capture_frame(&frame).await.unwrap();
    }
    c.leaf.playback_started(c.key(), Ts::ZERO);
    c.played(0, 600);
    let cut = PlaybackFinished {
        playback_position: Duration::from_millis(600),
        interrupted: true,
        synchronized_transcript: None,
    };
    c.leaf.playback_finished(c.key(), cut);
    c.leaf.reset();
    let shown: Vec<_> = c.take_shown().into_iter().map(|(w, _)| w).collect();
    assert_eq!(shown.concat().trim(), "one two");

    // The rest of the audio plays on from the cut.
    for _ in 0..14 {
        out.capture_frame(&frame).await.unwrap();
    }
    out.flush();
    c.leaf.playback_started(c.key(), Ts::ZERO);
    c.played(0, 500);
    let starts: Vec<_> = c.take_shown().into_iter().filter(|(w, _)| !w.trim().is_empty()).collect();
    assert_eq!(starts, [("three".to_owned(), 1.0)]);
    c.finished(1_400);
    let heard = out.wait_for_playout().await.synchronized_transcript;
    assert_eq!(heard.as_deref().map(str::trim), Some("one two three four"));
}

// A cut before any audio (`clear_buffer` with nothing captured) shows none of its text.
#[tokio::test]
async fn a_segment_cut_before_its_audio_shows_nothing() {
    let c = Chain::new();
    c.text("never said").await;
    c.sync.audio_output().clear_buffer();
    c.text("said").await;
    c.audio(1).await;
    c.leaf.playback_started(c.key(), Ts::ZERO);
    c.played(0, 1_000);
    c.finished(1_000);
    let shown: String = c.take_shown().into_iter().map(|(w, _)| w).collect();
    assert_eq!(shown.trim(), "said");
}

#[tokio::test]
async fn a_delayed_old_observation_does_not_release_the_next_segments_words() {
    let c = Chain::new();
    c.aligned(&[("old", 0.0, 1.0)]).await;
    c.audio(1).await;
    let old = c.key();
    c.leaf.playback_started(old, Ts::ZERO);
    c.played(0, 1_000);
    c.finished(1_000);
    c.take_shown();

    c.aligned(&[("next ", 0.0, 0.5), ("words", 0.5, 1.0)]).await;
    c.audio(1).await;
    let new = c.key();
    c.sync.audio.handle_playback(&PlaybackEvent::Finished {
        segment: old,
        event: PlaybackFinished { playback_position: Duration::from_secs(1), ..Default::default() },
    });
    assert!(c.take_shown().is_empty(), "a late observation released words before their audio");
    c.leaf.playback_started(new, Ts::ZERO);
    c.played(0, 1_000);
    c.finished(1_000);
    let shown: String = c.take_shown().into_iter().map(|(word, _)| word).collect();
    assert_eq!(shown, "next words");
}

/// The playback each released word was reported against.
struct Attributed(parking_lot::Mutex<Vec<(String, Option<PlaybackSegment>)>>);

impl PlaybackObserver for Attributed {
    fn on_playback_event(&self, _ev: &PlaybackEvent) {}

    fn on_transcript_shown(&self, segment: Option<PlaybackSegment>, word: &TranscriptChunk) {
        self.0.lock().push((word.text.clone(), segment));
    }
}

// Catches: a pause released the playing segment's started word with no playback named, so the
// speech playing next took it for its own: after a speech whose finish was lost, the next one's
// launch pause showed the old speech's " refund" as the new speech's, ahead of its audio.
#[tokio::test]
async fn a_pause_names_the_playback_whose_word_it_shows() {
    let c = Chain::new();
    let attributed = Arc::new(Attributed(Default::default()));
    c.sync.audio_output().tracker().observe(attributed.clone());
    c.aligned(&[("one ", 0.0, 0.5), ("two ", 0.5, 0.75), ("three", 0.75, 2.0)]).await;
    c.audio(2).await;
    c.leaf.playback_started(c.key(), Ts::ZERO);
    c.played(0, 700);
    attributed.0.lock().clear();
    // the playhead moves on to "three" before the pause reads it
    tokio::time::sleep(Duration::from_millis(100)).await;
    c.sync.audio_output().pause();
    let shown = attributed.0.lock().clone();
    assert!(!shown.is_empty(), "the pause shows the word whose audio started");
    assert!(shown.iter().all(|(_, segment)| *segment == Some(c.key())), "{shown:?}");
}
