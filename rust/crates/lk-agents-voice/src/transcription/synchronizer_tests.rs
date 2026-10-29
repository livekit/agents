use super::*;
use lk_agents_abi::Ts;
use lk_agents_test_support::playback::KeyedPlayback;
use tokio::sync::watch;

// ----------------------------------------------------------------------------- the chain

/// The words the synchroniser released, with where each one's audio starts.
struct Shown(watch::Sender<Vec<(String, f64)>>);

impl PlaybackObserver for Shown {
    fn on_playback_event(&self, _ev: &PlaybackEvent) {}

    fn on_transcript_shown(&self, word: &TimedString) {
        let start = word.start_time.unwrap_or_default();
        self.0.send_modify(|shown| shown.push((word.text.clone(), start)));
    }
}

struct Chain {
    sync: Arc<TranscriptSynchronizer>,
    leaf: Arc<PlaybackTracker>,
    shown: watch::Sender<Vec<(String, f64)>>,
    output: Arc<KeyedPlayback>,
}

impl Chain {
    fn new() -> Chain {
        let output = Arc::new(KeyedPlayback::default());
        let leaf = output.tracker().clone();
        let sync = TranscriptSynchronizer::new(SyncConfig {
            audio_next: output.clone() as Arc<dyn AudioOutput>,
            text_next: None,
            speed: 1.0,
            hyphenator: Arc::new(Liang),
            clock: Arc::new(lk_agents_abi::RealClock::new()),
            spawner: Arc::new(lk_agents_test_support::TokioSpawner::current()),
        });
        let shown = watch::Sender::new(Vec::new());
        sync.audio_output().tracker().observe(Arc::new(Shown(shown.clone())));
        Chain { sync, leaf, shown, output }
    }

    async fn text(&self, text: &str) {
        let out = self.sync.text_output();
        out.capture_text(text).await.unwrap();
        out.flush();
    }

    /// Words the voice aligned: each with where its audio starts and ends.
    async fn aligned(&self, words: &[(&str, f64, f64)]) {
        let out = self.sync.text_output();
        for (word, start, end) in words {
            let text = TimedString {
                start_time: Some(*start),
                end_time: Some(*end),
                ..TimedString::new(*word)
            };
            out.capture_timed(&text).await.unwrap();
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

    fn key(&self) -> lk_agents_abi::voice::io::PlaybackSegment {
        self.output.segment()
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
// words joined it and were all shown at its first frame, seconds before their audio (survey
// seed 2815).
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
// with it) or cut with nothing shown, though its text is the speech's history (survey seed
// 21650): it is shown whole at once, and the next segment paces on its own.
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
    assert!(c.take_shown().iter().all(|(_, start)| *start <= 1e-9));
    c.played(0, 1_100);
    c.finished(1_100);
    let shown: String = c.take_shown().into_iter().map(|(w, _)| w).collect();
    assert_eq!(shown.split_whitespace().collect::<Vec<_>>(), ["after", "words"]);
}

// Catches: the synchroniser still capturing after the output below reported its open segment cut
// and reset. The speech's later frames opened no segment of their own, so its wait returned at
// once with the cut and its words were shown under the next speech (survey seeds 29564, 24667,
// 6113).
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
// fell back to every word forwarded, and aligned words after it were shown off their stamps
// (campaign seeds 25043, 1906, 22545).
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
