use super::*;
use lk_agents_abi::Ts;
use tokio::sync::watch;

// ----------------------------------------------------------------------------- the chain

/// The sink below the synchroniser: it reports playback as the case says.
struct Leaf(Arc<PlaybackTracker>);

#[async_trait]
impl AudioOutput for Leaf {
    fn label(&self) -> &str {
        "leaf"
    }

    fn tracker(&self) -> &Arc<PlaybackTracker> {
        &self.0
    }

    fn capabilities(&self) -> AudioOutputCapabilities {
        AudioOutputCapabilities { pause: true }
    }

    async fn capture(&self, _frame: &AudioFrame) -> Result<()> {
        Ok(())
    }

    fn flush_segment(&self) {}

    fn clear_buffer(&self) {}
}

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
}

impl Chain {
    fn new() -> Chain {
        let leaf = Arc::new(PlaybackTracker::new());
        let sync = TranscriptSynchronizer::new(SyncConfig {
            audio_next: Arc::new(Leaf(Arc::clone(&leaf))),
            text_next: None,
            speed: 1.0,
            hyphenator: Arc::new(Liang),
            clock: Arc::new(lk_agents_abi::RealClock::new()),
            spawner: Arc::new(lk_agents_abi::TokioSpawner::current()),
        });
        let shown = watch::Sender::new(Vec::new());
        sync.audio_output().tracker().observe(Arc::new(Shown(shown.clone())));
        Chain { sync, leaf, shown }
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

    fn played(&self, offset_ms: u64, ms: u64) {
        let at = Duration::from_millis;
        self.leaf.playback_progressed(Ts::ZERO, at(offset_ms), at(ms));
    }

    fn finished(&self, at_ms: u64) {
        let playback_position = Duration::from_millis(at_ms);
        self.leaf.playback_finished(PlaybackFinished {
            playback_position,
            interrupted: false,
            synchronized_transcript: None,
        });
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
    c.leaf.playback_started(Ts::ZERO);
    c.played(0, 500);
    let audio = c.sync.audio_output();
    audio.on_detached();
    audio.on_attached();
    c.played(500, 1_500);
    c.finished(2_000);
    c.take_shown();

    c.aligned(&[("alpha ", 0.0, 0.5), ("bravo ", 0.5, 1.0), ("charlie ", 1.0, 1.5)]).await;
    c.audio(2).await;
    c.leaf.playback_started(Ts::ZERO);
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
    c.leaf.playback_started(Ts::ZERO);
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
    c.leaf.playback_finished(PlaybackFinished { interrupted: true, ..Default::default() });
    c.leaf.reset();
    out.capture_frame(&frame).await.unwrap();
    out.flush();
    let wait = tokio::time::timeout(Duration::ZERO, out.wait_for_playout()).await;
    assert!(wait.is_err(), "the wait returned before the segment after the reset played");
    c.finished(100);
    assert!(!out.wait_for_playout().await.interrupted);
}

// A cut before any audio (`clear_buffer` with nothing captured) shows none of its text.
#[tokio::test]
async fn a_segment_cut_before_its_audio_shows_nothing() {
    let c = Chain::new();
    c.text("never said").await;
    c.sync.audio_output().clear_buffer();
    c.text("said").await;
    c.audio(1).await;
    c.leaf.playback_started(Ts::ZERO);
    c.played(0, 1_000);
    c.finished(1_000);
    let shown: String = c.take_shown().into_iter().map(|(w, _)| w).collect();
    assert_eq!(shown.trim(), "said");
}

// ----------------------------------------------------------------------------- the pacer

fn paced(text: &str, audio_secs: f64) -> Pacer {
    let mut p = Pacer::new(PacerOptions::default());
    for ev in [
        PaceEvent::Text { raw: text.into(), start_time: None, end_time: None },
        PaceEvent::TextEnd,
        PaceEvent::Audio { secs: audio_secs },
        PaceEvent::AudioEnd,
    ] {
        p = p.step(ev).0;
    }
    p
}

fn warned(effects: &[PaceEffect]) -> Vec<PaceWarning> {
    effects
        .iter()
        .filter_map(|e| match e {
            PaceEffect::Warn(w) => Some(*w),
            _ => None,
        })
        .collect()
}

// Catches: an event out of order (a missing rotation upstream) silently changing a segment
// instead of being warned about and ignored, or a segment that finished early never reporting
// what was heard.
#[test]
fn events_out_of_order_are_warned_and_change_nothing() {
    let text = || PaceEvent::Text { raw: "late".into(), start_time: None, end_time: None };
    let mut p = Pacer::new(PacerOptions::default());
    for ev in [text(), PaceEvent::TextEnd] {
        p = p.step(ev).0;
    }
    let (p, effects) = p.step(text());
    assert_eq!(warned(&effects), [PaceWarning::TextAfterEnd]);
    let (p, _) = p.step(PaceEvent::Audio { secs: 0.2 });
    let (p, _) = p.step(PaceEvent::AudioEnd);
    let (p, effects) = p.step(PaceEvent::AudioEnd);
    assert_eq!(warned(&effects), [PaceWarning::AudioAfterEnd]);

    // Audio first: the text's end is what ends the inputs.
    let mut q = Pacer::new(PacerOptions::default());
    for ev in [text(), PaceEvent::Audio { secs: 0.2 }, PaceEvent::AudioEnd, PaceEvent::TextEnd] {
        let (next, effects) = q.step(ev);
        assert!(warned(&effects).is_empty());
        q = next;
    }
    assert!(q.text_ended() && q.audio_ended());

    // Finished before the inputs ended: warned, and still heard once.
    let early = Pacer::new(PacerOptions::default()).step(text()).0;
    let finish =
        PaceEvent::PlaybackFinished { position: Duration::from_millis(50), interrupted: false };
    let (early, effects) = early.step(finish.clone());
    assert_eq!(warned(&effects), [PaceWarning::PlaybackFinishedTooEarly]);
    assert_eq!(effects.iter().filter(|e| matches!(e, PaceEffect::Heard(_))).count(), 1);
    let (early, effects) = early.step(PaceEvent::Position(Duration::from_millis(60)));
    assert_eq!(warned(&effects), [PaceWarning::AfterFinished]);
    let (early, effects) = early.step(PaceEvent::Close);
    assert!(effects.is_empty() && early.is_over());
    let (_, effects) = early.step(finish);
    assert_eq!(warned(&effects), [PaceWarning::AfterClose]);
    let _ = p;
}

// Catches: an aligned delta (a TTS that reports where each word starts) planned on the
// estimated pace instead of its own onset.
#[test]
fn an_aligned_delta_starts_where_the_voice_says() {
    let mut p = Pacer::new(PacerOptions::default());
    for ev in [
        PaceEvent::Text { raw: "one ".into(), start_time: Some(0.0), end_time: None },
        PaceEvent::Text { raw: "two".into(), start_time: Some(1.0), end_time: None },
        PaceEvent::TextEnd,
        PaceEvent::Audio { secs: 2.0 },
        PaceEvent::AudioEnd,
    ] {
        p = p.step(ev).0;
    }
    let (p, effects) = p.step(PaceEvent::Position(Duration::from_millis(900)));
    assert_eq!(shown(&effects), "one", "the second word starts at its aligned onset, 1 s");
    let (_, effects) = p.step(PaceEvent::Position(Duration::from_millis(1900)));
    assert_eq!(shown(&effects), " two");
}

fn shown(effects: &[PaceEffect]) -> String {
    effects
        .iter()
        .filter_map(|e| match e {
            PaceEffect::Forward { word, .. } => Some(word.as_str()),
            _ => None,
        })
        .collect()
}

// The reference's pacing loop shows a word halfway through its delay and counts it heard only
// once the whole delay passed (`_SegmentSynchronizerImpl._main_task`): a cut keeps the words
// whose audio ended, not the one playing. Found by `scripts/parity.py say_cut`.
#[test]
fn a_cut_keeps_the_words_whose_audio_ended() {
    // Four one-hyphen words over 0.4 s: 0.1 s each, shown at 0.05, 0.15, ...
    let p = paced("one two three four", 0.4);
    let (p, effects) = p.step(PaceEvent::Position(Duration::from_millis(160)));
    assert_eq!(shown(&effects), "one two");
    let (_, effects) = p.step(PaceEvent::PlaybackFinished {
        position: Duration::from_millis(160),
        interrupted: true,
    });
    let heard = effects.iter().find_map(|e| match e {
        PaceEffect::Heard(h) => Some(h.text.clone()),
        _ => None,
    });
    assert_eq!(heard.as_deref(), Some("one"), "\"two\" was shown but still playing");
}

// Catches: a sink that reports no progress (`DataStreamAudioOutput`, an avatar) cut after
// seconds of speech leaving nothing heard, so the interrupted reply had no item and the caller
// no transcript (found by `scripts/e2e.py avatar` on LiveKit Cloud).
#[test]
fn a_cut_without_progress_hears_up_to_the_reported_playhead() {
    let p = paced("one two three four", 0.4);
    let (_, effects) = p.step(PaceEvent::PlaybackFinished {
        position: Duration::from_millis(260),
        interrupted: true,
    });
    assert_eq!(shown(&effects), "one two three");
    let heard = effects.iter().find_map(|e| match e {
        PaceEffect::Heard(h) => Some(h.text.clone()),
        _ => None,
    });
    assert_eq!(heard.as_deref(), Some("one two"));
}

// The reference's pacing loop sleeps through a pause: the word whose delay began before it is
// still shown (not heard), and only the next word waits. Found by `parity.py
// realtime_interruption`.
#[test]
fn a_pause_shows_the_word_in_its_delay() {
    let p = paced("one two three four", 0.4);
    // "one" is out; "two" started at 0.1 and shows at 0.15.
    let (p, effects) = p.step(PaceEvent::Position(Duration::from_millis(120)));
    assert_eq!(shown(&effects), "one");
    let (p, effects) = p.step(PaceEvent::Paused);
    assert_eq!(shown(&effects), " two");
    let (_, effects) = p.step(PaceEvent::PlaybackFinished {
        position: Duration::from_millis(120),
        interrupted: true,
    });
    let heard = effects.iter().find_map(|e| match e {
        PaceEffect::Heard(h) => Some(h.text.clone()),
        _ => None,
    });
    assert_eq!(heard.as_deref(), Some("one"), "\"two\" was shown but still playing");
}

// An aligned word starts at its stamp: a pause before it shows nothing (the pause shows only
// a word already playing). Found by `transcript_follows_audio` reading it.
#[test]
fn a_pause_shows_no_aligned_word_ahead_of_its_stamp() {
    let mut p = Pacer::new(PacerOptions::default());
    for (raw, at) in [("one ", 0.0), ("two ", 0.2)] {
        p = p.step(PaceEvent::Text { raw: raw.into(), start_time: Some(at), end_time: None }).0;
    }
    p = p.step(PaceEvent::Audio { secs: 0.4 }).0;
    let (p, effects) = p.step(PaceEvent::Position(Duration::from_millis(100)));
    assert_eq!(shown(&effects), "one");
    let (_, effects) = p.step(PaceEvent::Paused);
    assert_eq!(shown(&effects), "", "nothing after \"one\" has started by the pause at 0.1 s");
}

// An aligned word's audio is over at the TTS's `end_time`, however long its hyphens pace it: a
// cut after it keeps it. Found by `cut_keeps_heard_words` (seed 7680): "Okay." stamped over the
// first 10 ms was paced to 0.26 s, so a cut at 0.24 s remembered nothing.
#[test]
fn an_aligned_word_ends_at_its_stamp() {
    let mut p = Pacer::new(PacerOptions::default());
    for (raw, at) in [("Okay. ", 0.0), ("I've ", 0.01)] {
        let ev =
            PaceEvent::Text { raw: raw.into(), start_time: Some(at), end_time: Some(at + 0.01) };
        p = p.step(ev).0;
    }
    p = p.step(PaceEvent::Audio { secs: 0.24 }).0;
    let (p, _) = p.step(PaceEvent::Position(Duration::from_millis(240)));
    let (_, effects) = p.step(PaceEvent::PlaybackFinished {
        position: Duration::from_millis(240),
        interrupted: true,
    });
    let heard = effects.iter().find_map(|e| match e {
        PaceEffect::Heard(h) => Some(h.text.clone()),
        _ => None,
    });
    assert_eq!(heard.as_deref(), Some("Okay. I've"));
}

// An aligned delta's trailing space, paced a hyphen past the word, holds back no stamped word
// after it. Found by `cut_keeps_heard_words` (seed 27432): the space after "That's" was due at
// 0.26 s, so nothing after it showed while 120 ms played, and the cut remembered it all.
#[test]
fn a_paced_space_holds_back_no_stamped_word() {
    let mut p = Pacer::new(PacerOptions::default());
    for (raw, at, end) in
        [("That's ", 0.0, None), ("outside ", 0.008, None), ("what", 0.016, Some(0.03))]
    {
        let ev = PaceEvent::Text { raw: raw.into(), start_time: Some(at), end_time: end };
        p = p.step(ev).0;
    }
    p = p.step(PaceEvent::TextEnd).0;
    let (_, effects) = p.step(PaceEvent::Position(Duration::from_millis(120)));
    assert_eq!(shown(&effects).trim_end(), "That's outside what");
}

// A word's audio is over once the next one's starts: an adapted voice stamps a whole sentence,
// its first word paced at the standard rate until the audio ended, the rest at the segment's.
// Found by `cut_keeps_heard_words` (seed 7680): the first word outlasted the cut and the
// reply remembered nothing of the words after it.
#[test]
fn a_word_ends_where_the_next_starts() {
    let mut p = Pacer::new(PacerOptions::default());
    let ev =
        PaceEvent::Text { raw: "one two three ".into(), start_time: Some(0.0), end_time: None };
    p = p.step(ev).0;
    p = p.step(PaceEvent::Audio { secs: 0.03 }).0;
    // "one" shows at the start, paced at the standard rate (to 0.26 s).
    p = p.step(PaceEvent::Position(Duration::ZERO)).0;
    p = p.step(PaceEvent::TextEnd).0;
    p = p.step(PaceEvent::AudioEnd).0;
    let (p, _) = p.step(PaceEvent::Position(Duration::from_millis(25)));
    let (_, effects) = p.step(PaceEvent::PlaybackFinished {
        position: Duration::from_millis(25),
        interrupted: true,
    });
    let heard = effects.iter().find_map(|e| match e {
        PaceEffect::Heard(h) => Some(h.text.clone()),
        _ => None,
    });
    assert_eq!(heard.as_deref(), Some("one two"));
}

// Each word lasts its hyphens at the speed known when it starts, as the reference computes
// each delay when it reaches the word: a re-estimate moves the words after it, not before.
#[test]
fn a_word_keeps_the_speed_it_started_at() {
    let mut p = Pacer::new(PacerOptions::default());
    p = p.step(PaceEvent::Text { raw: "one two three".into(), start_time: None, end_time: None }).0;
    p = p.step(PaceEvent::TextEnd).0;
    p = p.step(PaceEvent::Audio { secs: 0.3 }).0;
    // The first word starts with the playback, at the standard rate (1 / 3.83 s).
    p = p.step(PaceEvent::Position(Duration::ZERO)).0;
    p = p.step(PaceEvent::AudioEnd).0;
    let (_, effects) = p.step(PaceEvent::Position(Duration::from_millis(200)));
    // "one" ends at 0.261 s; "two" (0.1 s at the re-estimated 10 hyphens/s) is shown at 0.311.
    assert_eq!(shown(&effects), "one");
}
