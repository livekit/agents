use super::*;

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
