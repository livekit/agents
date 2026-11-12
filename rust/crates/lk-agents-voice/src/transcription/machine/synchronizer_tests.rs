use super::{TranscriptPacerOptions, TranscriptSync, TranscriptSyncEvent};
use crate::transcription::effects::TranscriptSyncEffect;
use lk_agents_abi::{TranscriptChunk, Ts};
use std::time::Duration;

fn synchronized() -> TranscriptSync {
    let mut sync = TranscriptSync::new(TranscriptPacerOptions::default());
    sync.handle(TranscriptSyncEvent::AttachText);
    sync.handle(TranscriptSyncEvent::AttachmentChanged { audio: Some(true), text: Some(true) });
    sync
}

fn capture(raw: &str, start_time: Option<f64>, end_time: Option<f64>) -> TranscriptSyncEvent {
    TranscriptSyncEvent::Capture {
        text: TranscriptChunk { start_time, end_time, ..TranscriptChunk::new(raw) },
        aligned: start_time.is_some() || end_time.is_some(),
    }
}

fn words(effects: &[TranscriptSyncEffect]) -> String {
    effects
        .iter()
        .filter_map(|effect| match effect {
            TranscriptSyncEffect::PublishWord { chunk, .. } => Some(chunk.text.as_str()),
            _ => None,
        })
        .collect()
}

/// Repeated audio flushes preserve the current and following transcripts. https://github.com/livekit/agents/pull/8316
#[test]
#[ignore = "https://github.com/livekit/agents/pull/8316 reproduces: a repeated audio end strands the next transcript"]
fn a_second_audio_end_cannot_consume_the_next_segments_audio_slot() {
    for repeat in [false, true] {
        let mut sync = synchronized();
        let mut heard = Vec::new();
        let mut shown = String::new();
        for text in ["hello world", "next turn"] {
            let mut events = vec![
                capture(text, Some(0.0), Some(1.0)),
                TranscriptSyncEvent::FlushText,
                TranscriptSyncEvent::AudioCaptureStarted { secs: 1.0, segment: None },
                TranscriptSyncEvent::AudioEnded,
            ];
            if repeat {
                events.push(TranscriptSyncEvent::AudioEnded);
            }
            events.push(TranscriptSyncEvent::PlaybackFinished {
                position: Duration::from_secs(1),
                interrupted: false,
            });
            for event in events {
                let effects = sync.handle(event);
                shown.push_str(&words(&effects));
                for effect in effects {
                    if let TranscriptSyncEffect::ReportHeard(report) = effect {
                        assert!(report.complete);
                        heard.push(report.text);
                    }
                }
            }
        }
        assert_eq!(heard, ["hello world", "next turn"], "repeated flush: {repeat}");
        assert_eq!(shown, "hello worldnext turn");
    }
}

#[test]
fn expressive_markup_adds_no_spoken_pacing_delay() {
    let text = concat!(
        "<expr type=\"expression\" label=\"speak with warm surprise and bright energy\"/> ",
        "Hello there my friend! ",
        "<expr type=\"sound\" label=\"laugh\"/> ",
        "<expr type=\"expression\" label=\"speak calmly and evenly, unhurried\"/> ",
        "How are you today?"
    );
    let mut sync = synchronized();
    for event in [
        capture(text, None, None),
        TranscriptSyncEvent::FlushText,
        TranscriptSyncEvent::AudioCaptureStarted { secs: 3.0, segment: None },
        TranscriptSyncEvent::AudioEnded,
    ] {
        sync.handle(event);
    }
    let effects = sync.handle(TranscriptSyncEvent::PlaybackProgressed {
        position: Duration::from_secs(3),
        at: Ts::ZERO,
    });
    assert_eq!(words(&effects), text);
}

#[test]
fn a_timed_transcripts_last_word_is_shown_before_text_input_ends() {
    let mut sync = synchronized();
    for event in [
        TranscriptSyncEvent::AudioCaptureStarted { secs: 0.4, segment: None },
        TranscriptSyncEvent::AudioEnded,
        capture("It is", Some(0.0), Some(0.2)),
        capture(" sunny", Some(0.2), Some(0.4)),
    ] {
        sync.handle(event);
    }
    let effects = sync.handle(TranscriptSyncEvent::PlaybackProgressed {
        position: Duration::from_millis(400),
        at: Ts::ZERO,
    });
    assert_eq!(words(&effects), "It is sunny");
}

// The owner must publish the final paced words and flush before reporting heard text, with no
// second terminal output if a sink repeats its finish notification.
#[test]
fn playback_finish_publishes_words_before_flush_and_heard_once() {
    let mut sync = synchronized();
    for event in [
        capture("hello world", Some(0.0), Some(1.0)),
        TranscriptSyncEvent::FlushText,
        TranscriptSyncEvent::AudioCaptureStarted { secs: 1.0, segment: None },
        TranscriptSyncEvent::AudioEnded,
    ] {
        sync.handle(event);
    }
    let finish = || TranscriptSyncEvent::PlaybackFinished {
        position: Duration::from_secs(1),
        interrupted: false,
    };
    let effects = sync.handle(finish());
    assert_eq!(words(&effects), "hello world");
    let flush = effects
        .iter()
        .position(|e| matches!(e, TranscriptSyncEffect::FlushTranscript))
        .expect("completed transcript flushes");
    let report = effects
        .iter()
        .position(|e| matches!(e, TranscriptSyncEffect::ReportHeard(_)))
        .expect("completed playback reports what was heard");
    assert!(effects[..flush].iter().any(|e| matches!(e, TranscriptSyncEffect::PublishWord { .. })));
    assert!(flush < report);
    let TranscriptSyncEffect::ReportHeard(heard) = &effects[report] else { unreachable!() };
    assert_eq!(heard.text, "hello world");
    assert!(heard.complete);
    assert!(sync.handle(finish()).is_empty());
}

// Closing must deliver pending transcript output before stopping its physical forwarder.
#[test]
fn closing_flushes_transcript_before_sender_closes_and_waits_once() {
    let mut sync = synchronized();
    sync.handle(capture("hello world", None, None));
    sync.handle(TranscriptSyncEvent::AudioCaptureStarted { secs: 1.0, segment: None });
    let shown = sync.handle(TranscriptSyncEvent::PlaybackProgressed {
        position: Duration::from_millis(400),
        at: Ts::ZERO,
    });
    assert!(words(&shown).starts_with("hello"), "a word was shown before the close");
    let effects = sync.handle(TranscriptSyncEvent::Close);
    let flush = effects
        .iter()
        .position(|e| matches!(e, TranscriptSyncEffect::FlushTranscript))
        .expect("closing flushes the live transcript");
    let closed = effects
        .iter()
        .position(|e| matches!(e, TranscriptSyncEffect::CloseSender))
        .expect("closing ends forwarding admission");
    let wait = effects
        .iter()
        .position(|e| matches!(e, TranscriptSyncEffect::BeginWait))
        .expect("closing waits for forwarding");
    assert!(flush < closed && closed < wait);
    assert!(sync.handle(TranscriptSyncEvent::Close).is_empty());
}

/// A segment's text, whether aligned CJK fragments or untimed aligned text, is shown, heard and
/// finalized once, whole and with its punctuation.
/// https://github.com/livekit/agents/issues/3872 https://github.com/livekit/agents/pull/6704
/// https://github.com/livekit/agents/pull/648 https://github.com/livekit/agents/issues/2288
#[test]
fn fragmented_transcripts_finalize_without_lost_text() {
    let cjk = vec![
        capture("你", Some(0.0), Some(0.2)),
        capture("好", Some(0.2), Some(0.4)),
        capture("，世界。", Some(0.4), Some(1.0)),
    ];
    let untimed = vec![TranscriptSyncEvent::Capture {
        text: TranscriptChunk::new("Well, hello there!"),
        aligned: true,
    }];
    for (captures, text) in [(cjk, "你好，世界。"), (untimed, "Well, hello there!")] {
        let mut sync = synchronized();
        let mut events = captures;
        events.extend([
            TranscriptSyncEvent::FlushText,
            TranscriptSyncEvent::AudioCaptureStarted { secs: 1.0, segment: None },
            TranscriptSyncEvent::AudioEnded,
            TranscriptSyncEvent::PlaybackStarted { at: Ts::ZERO },
            TranscriptSyncEvent::PlaybackFinished {
                position: Duration::from_secs(1),
                interrupted: false,
            },
            TranscriptSyncEvent::PlaybackFinished {
                position: Duration::from_secs(1),
                interrupted: false,
            },
        ]);

        let effects: Vec<_> = events.into_iter().flat_map(|event| sync.handle(event)).collect();

        let heard: Vec<_> = effects
            .iter()
            .filter_map(|effect| match effect {
                TranscriptSyncEffect::ReportHeard(report) => {
                    Some((report.text.as_str(), report.complete))
                }
                _ => None,
            })
            .collect();
        assert_eq!(words(&effects), text);
        assert_eq!(heard, [(text, true)]);
        let flushes = effects.iter().filter(|e| matches!(e, TranscriptSyncEffect::FlushTranscript));
        assert_eq!(flushes.count(), 1);
        let warnings: Vec<_> = effects
            .iter()
            .filter_map(|effect| match effect {
                TranscriptSyncEffect::Warn(warning) => Some(warning),
                _ => None,
            })
            .collect();
        assert!(warnings.is_empty(), "{warnings:?}");
    }
}

/// Audio whose text input ended with no text (a realtime reply with no transcript) finalizes
/// once without a synchronizer warning. https://github.com/livekit/agents/pull/2661
#[test]
#[ignore = "https://github.com/livekit/agents/pull/2661 reproduces: a text flush with nothing captured is dropped, so playback_finished warns PlaybackFinishedTooEarly"]
fn textless_audio_finalizes_without_a_sync_warning() {
    let mut sync = synchronized();
    let events = [
        TranscriptSyncEvent::FlushText,
        TranscriptSyncEvent::AudioCaptureStarted { secs: 1.0, segment: None },
        TranscriptSyncEvent::AudioEnded,
        TranscriptSyncEvent::PlaybackStarted { at: Ts::ZERO },
        TranscriptSyncEvent::PlaybackFinished {
            position: Duration::from_secs(1),
            interrupted: false,
        },
    ];

    let effects: Vec<_> = events.into_iter().flat_map(|event| sync.handle(event)).collect();

    // No words, so no transcript segment to close.
    let flushes = effects.iter().filter(|e| matches!(e, TranscriptSyncEffect::FlushTranscript));
    assert_eq!(flushes.count(), 0);
    let warnings: Vec<_> = effects
        .iter()
        .filter_map(|effect| match effect {
            TranscriptSyncEffect::Warn(warning) => Some(warning),
            _ => None,
        })
        .collect();
    assert!(warnings.is_empty(), "{warnings:?}");
}

const TWELVE: &str = "one two three four five six seven eight nine ten eleven twelve";

fn at(ms: u64) -> Ts {
    Ts::from_nanos(ms * 1_000_000)
}

/// Feeds every tick the synchronizer asks for up to `until`, as a clock would, after `effects`.
/// Returns each word published with the instant it was, and the tick still pending past `until`.
fn tick_until(
    sync: &mut TranscriptSync,
    effects: Vec<TranscriptSyncEffect>,
    until: Ts,
) -> (Vec<(Ts, String)>, Option<Ts>) {
    let mut shown = Vec::new();
    let mut pending: Option<Ts> = None;
    let mut collect = |effects: Vec<TranscriptSyncEffect>, now: Ts, pending: &mut Option<Ts>| {
        for effect in effects {
            match effect {
                TranscriptSyncEffect::PublishWord { chunk, .. } => shown.push((now, chunk.text)),
                TranscriptSyncEffect::TickAt(tick) => *pending = Some(tick),
                _ => {}
            }
        }
    };
    collect(effects, Ts::ZERO, &mut pending);
    while let Some(tick) = pending.filter(|tick| *tick <= until) {
        pending = None;
        collect(sync.handle(TranscriptSyncEvent::Tick(tick)), tick, &mut pending);
    }
    (shown, pending)
}

/// Twelve words, three seconds of audio, all of it sent and ended.
fn twelve_words_sent(sync: &mut TranscriptSync) {
    for event in [
        capture(TWELVE, None, None),
        TranscriptSyncEvent::FlushText,
        TranscriptSyncEvent::AudioCaptureStarted { secs: 3.0, segment: None },
        TranscriptSyncEvent::AudioEnded,
    ] {
        sync.handle(event);
    }
}

/// A sink that reports only its start and its finish still gets its words paced through the
/// playout, not in one burst at the finish.
#[test]
fn words_move_with_elapsed_time_when_the_sink_reports_no_progress() {
    let mut sync = synchronized();
    twelve_words_sent(&mut sync);
    let started = sync.handle(TranscriptSyncEvent::PlaybackStarted { at: Ts::ZERO });
    let (shown, _) = tick_until(&mut sync, started, at(2_990));
    let text: String = shown.iter().map(|(_, w)| w.as_str()).collect();
    assert!(text.starts_with("one two three"), "{text:?}");
    let first = shown.first().expect("words before the finish").0;
    let last = shown.last().expect("words before the finish").0;
    assert!(first <= at(500), "the first word at {first:?}");
    assert!(last >= at(2_000), "the words came in a burst by {last:?}");
    // The finish shows only what is left, at once.
    let finish = sync.handle(TranscriptSyncEvent::PlaybackFinished {
        position: Duration::from_secs(3),
        interrupted: false,
    });
    assert_eq!(format!("{text}{}", words(&finish)), TWELVE);
}

/// No word is shown ahead of the audio sent to the sink: elapsed time past it waits for more.
#[test]
fn elapsed_time_never_passes_the_audio_sent() {
    let mut sync = synchronized();
    for event in [capture(TWELVE, None, None), TranscriptSyncEvent::FlushText] {
        sync.handle(event);
    }
    sync.handle(TranscriptSyncEvent::AudioCaptureStarted { secs: 0.5, segment: None });
    let started = sync.handle(TranscriptSyncEvent::PlaybackStarted { at: Ts::ZERO });
    let (shown, pending) = tick_until(&mut sync, started, at(2_000));
    assert!(pending.is_none(), "a tick past the audio sent: {pending:?}");
    let held = shown.len();
    assert!(held < 4, "{shown:?}");
    // More audio: the words move on from the time already elapsed.
    let more = sync.handle(TranscriptSyncEvent::AudioCaptureStarted { secs: 2.5, segment: None });
    let tick = more.iter().find_map(|e| match e {
        TranscriptSyncEffect::TickAt(tick) => Some(*tick),
        _ => None,
    });
    assert!(tick.is_some_and(|tick| tick <= at(2_000)), "{tick:?}");
    let (shown, _) = tick_until(&mut sync, more, at(2_000));
    assert!(shown.len() > held, "{shown:?}");
}

/// Paused, the playhead stands still; resumed, it moves on from where it stopped.
#[test]
fn a_pause_stops_the_words_until_the_resume() {
    let mut sync = synchronized();
    twelve_words_sent(&mut sync);
    let started = sync.handle(TranscriptSyncEvent::PlaybackStarted { at: Ts::ZERO });
    let (before, _) = tick_until(&mut sync, started, at(1_000));
    let paused = sync.handle(TranscriptSyncEvent::PlaybackPaused { at: at(1_000) });
    let shown = before.len() + words(&paused).split_whitespace().count();
    assert!(shown < 12, "{before:?}");
    // A tick asked for before the pause finds the playhead where the pause left it.
    let late = sync.handle(TranscriptSyncEvent::Tick(at(4_000)));
    assert_eq!(words(&late), "", "a word shown while paused");
    assert!(!late.iter().any(|e| matches!(e, TranscriptSyncEffect::TickAt(_))));
    let resumed = sync.handle(TranscriptSyncEvent::PlaybackResumed { at: at(5_000) });
    let (after, _) = tick_until(&mut sync, resumed, at(5_500));
    assert!(!after.is_empty(), "the words move on after the resume");
    assert!(after.iter().all(|(when, _)| *when > at(5_000)), "{after:?}");
    assert!(shown + after.len() < 12, "the paused time counted as played: {after:?}");
}

/// A sink's progress report corrects the estimate: a sink behind the clock holds the words back.
#[test]
fn a_progress_report_corrects_the_elapsed_time_estimate() {
    let mut sync = synchronized();
    twelve_words_sent(&mut sync);
    sync.handle(TranscriptSyncEvent::PlaybackStarted { at: Ts::ZERO });
    // At 2 s the sink says it has played only 0.2 s.
    let behind = sync.handle(TranscriptSyncEvent::PlaybackProgressed {
        position: Duration::from_millis(200),
        at: at(2_000),
    });
    let (shown, _) = tick_until(&mut sync, behind, at(2_300));
    assert!(shown.len() < 4, "words past the playhead the sink reported: {shown:?}");
}

/// A flush closes a transcript segment that had text: none when synchronisation starts or the
/// session closes with nothing shown, and none for a segment whose audio had no words.
#[test]
fn a_flush_only_closes_text_that_was_shown() {
    let mut sync = TranscriptSync::new(TranscriptPacerOptions::default());
    let mut effects = sync.handle(TranscriptSyncEvent::AttachText);
    effects.extend(
        sync.handle(TranscriptSyncEvent::AttachmentChanged { audio: Some(true), text: Some(true) }),
    );
    for event in [
        TranscriptSyncEvent::AudioCaptureStarted { secs: 1.0, segment: None },
        TranscriptSyncEvent::AudioEnded,
        TranscriptSyncEvent::PlaybackStarted { at: Ts::ZERO },
        TranscriptSyncEvent::PlaybackFinished {
            position: Duration::from_secs(1),
            interrupted: false,
        },
        TranscriptSyncEvent::Close,
    ] {
        effects.extend(sync.handle(event));
    }
    let flushes = effects.iter().filter(|e| matches!(e, TranscriptSyncEffect::FlushTranscript));
    assert_eq!(flushes.count(), 0);
}

/// A three-word aligned segment playing on `KEY`, its playout reported 600 ms in.
fn cut_mid_speech() -> (TranscriptSync, String) {
    let mut sync = synchronized();
    let mut shown = String::new();
    for event in [
        capture("one", Some(0.0), Some(0.5)),
        capture(" two", Some(0.5), Some(1.0)),
        capture(" three", Some(1.0), Some(1.5)),
        TranscriptSyncEvent::FlushText,
        TranscriptSyncEvent::AudioCaptureStarted { secs: 1.5, segment: Some(KEY) },
        TranscriptSyncEvent::AudioEnded,
        TranscriptSyncEvent::PlaybackStarted { at: Ts::ZERO },
        TranscriptSyncEvent::PlaybackProgressed {
            position: Duration::from_millis(600),
            at: Ts::ZERO.saturating_add(Duration::from_millis(600)),
        },
    ] {
        shown.push_str(&words(&sync.handle(event)));
    }
    (sync, shown)
}

const KEY: lk_agents_abi::voice::io::PlaybackSegment =
    lk_agents_abi::voice::io::PlaybackSegment { attachment: 1, segment: 1 };

fn at_ms(ms: u64) -> Ts {
    Ts::ZERO.saturating_add(Duration::from_millis(ms))
}

// Catches: a clear the sink never reported left the segment's words to be paced on, by elapsed
// time, up to all of its audio, and nothing told what had been heard at the clear.
#[test]
fn a_clear_reports_what_was_heard_and_shows_no_more_words() {
    let (mut sync, shown) = cut_mid_speech();
    assert_eq!(shown, "one two");
    let effects = sync.handle(TranscriptSyncEvent::AudioCleared { at: at_ms(600) });
    let reported: Vec<_> = effects
        .iter()
        .filter_map(|effect| match effect {
            TranscriptSyncEffect::ReportCleared { segment, heard } => Some((*segment, heard)),
            _ => None,
        })
        .collect();
    let [(segment, heard)] = reported.as_slice() else { panic!("one report: {}", reported.len()) };
    assert_eq!((*segment, heard.text.as_str(), heard.complete), (KEY, "one", false));
    assert!(effects.iter().any(|e| matches!(e, TranscriptSyncEffect::FlushTranscript)));
    assert!(!effects.iter().any(|e| matches!(e, TranscriptSyncEffect::TickAt(_))));
    for event in [
        TranscriptSyncEvent::Tick(at_ms(1_400)),
        TranscriptSyncEvent::PlaybackProgressed {
            position: Duration::from_millis(1_500),
            at: at_ms(1_500),
        },
    ] {
        let effects = sync.handle(event);
        assert_eq!(words(&effects), "", "a word shown after the clear");
        assert!(!effects.iter().any(|e| matches!(e, TranscriptSyncEffect::TickAt(_))));
    }
}

// Catches: a cleared segment's late finish showed the words its reported position reached, after
// the cut's text was committed; or, reported played whole, all of them.
#[test]
fn a_cleared_segments_late_finish_shows_nothing_more() {
    for interrupted in [true, false] {
        let (mut sync, shown) = cut_mid_speech();
        sync.handle(TranscriptSyncEvent::AudioCleared { at: at_ms(600) });
        let effects = sync.handle(TranscriptSyncEvent::PlaybackFinished {
            position: Duration::from_millis(1_500),
            interrupted,
        });
        assert_eq!(words(&effects), "", "a word shown at the late finish");
        let heard: Vec<_> = effects
            .iter()
            .filter_map(|effect| match effect {
                TranscriptSyncEffect::ReportHeard(heard) => Some(heard),
                _ => None,
            })
            .collect();
        let [heard] = heard.as_slice() else { panic!("one heard report: {}", heard.len()) };
        assert!(!heard.complete, "a cleared segment is cut whatever its finish says");
        assert!(shown.starts_with(&heard.text), "heard {:?} beyond shown {shown:?}", heard.text);
    }
}

// Pins: a sink that reports its clear before the synchroniser takes it finishes the segment as
// before; the clear finds nothing to report.
#[test]
fn a_clear_after_the_finish_reports_nothing() {
    let (mut sync, _) = cut_mid_speech();
    let effects = sync.handle(TranscriptSyncEvent::PlaybackFinished {
        position: Duration::from_millis(600),
        interrupted: true,
    });
    assert!(
        effects
            .iter()
            .any(|e| matches!(e, TranscriptSyncEffect::ReportHeard(h) if h.text == "one"))
    );
    let effects = sync.handle(TranscriptSyncEvent::AudioCleared { at: at_ms(600) });
    assert!(!effects.iter().any(|e| matches!(e, TranscriptSyncEffect::ReportCleared { .. })));
}
