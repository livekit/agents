use super::*;
use lk_agents_abi::voice::io::{PlaybackFinished, PlaybackProgressed, PlaybackStarted};

const SEGMENT: PlaybackSegment = PlaybackSegment { attachment: 1, segment: 1 };

fn ms(n: u64) -> Duration {
    Duration::from_millis(n)
}

/// `n` ms of 1 kHz mono audio.
fn frame(n: u32) -> AudioFrame {
    AudioFrame::new(vec![1u8; n as usize * 2], 1_000, 1, n)
}

fn at(n: u64) -> Ts {
    Ts::ZERO.saturating_add(ms(n))
}

fn progressed(started: u64, offset: u64, duration: u64) -> PlaybackEvent {
    PlaybackEvent::Progressed(PlaybackProgressed {
        segment: SEGMENT,
        started_at: at(started),
        offset: ms(offset),
        duration: ms(duration),
    })
}

fn playing(heard: Option<u64>) -> UnplayedAudio {
    let mut audio = UnplayedAudio::default();
    audio.captured(SEGMENT, &frame(300));
    audio.captured(SEGMENT, &frame(200));
    audio.observe(&PlaybackEvent::Started(PlaybackStarted {
        segment: SEGMENT,
        created_at: at(1_000),
    }));
    if let Some(heard) = heard {
        audio.observe(&progressed(1_000, 0, heard));
    }
    audio
}

// Catches: a recorder attached mid-segment given nothing of it, or all of it from the start.
#[test]
fn a_joined_segment_replays_from_its_reported_position() {
    let mut audio = playing(Some(350));
    let replay = audio.join(at(1_400));
    let [Replay::Started(started), Replay::Capture(rest)] = replay.as_slice() else {
        panic!("{replay:?}")
    };
    assert_eq!(*started, at(1_400));
    assert_eq!(rest.samples_per_channel, 150);
    // The next report is told from where the recorder joined.
    let told = audio.told(&progressed(1_400, 350, 100));
    assert_eq!(told, vec![progressed(1_400, 0, 100)]);
}

fn finished(position: u64) -> PlaybackEvent {
    PlaybackEvent::Finished {
        segment: SEGMENT,
        event: PlaybackFinished {
            playback_position: ms(position),
            interrupted: false,
            synchronized_transcript: None,
        },
    }
}

// Catches: a sink that never reports progress placed from its start, so the audio it played
// before the recorder attached lands before the recording, or none of it lands at all.
#[test]
fn a_sink_that_never_reports_is_placed_from_the_join() {
    let mut audio = playing(None);
    let replay = audio.join(at(1_100));
    let [Replay::Started(started), Replay::Capture(first), Replay::Capture(second)] =
        replay.as_slice()
    else {
        panic!("{replay:?}")
    };
    assert_eq!(
        (*started, first.samples_per_channel, second.samples_per_channel),
        (at(1_100), 300, 200)
    );
    // Its finish says it played from its start: what played after the join is placed there.
    assert_eq!(audio.told(&finished(500)), vec![progressed(1_100, 100, 400), finished(500)]);
}

// Catches: a sink that played everything before the join left to the recorder's fallback, which
// places it before the recording began.
#[test]
fn a_sink_that_never_reports_and_played_before_the_join_places_nothing() {
    let mut audio = playing(None);
    audio.join(at(3_000));
    assert_eq!(audio.told(&finished(500)), vec![progressed(1_500, 500, 0), finished(500)]);
}

// Catches: seed 591's hole. A segment started on an output that had not reported yet, then held
// while the output was replaced; a recorder attached seconds later was told it had played since
// it started and got nothing, so the progress the sink reported from offset 0 placed silence where
// the person heard the audio.
#[test]
fn a_held_segment_with_no_report_is_joined_whole() {
    let mut audio = playing(None);
    let replay = audio.join(at(5_000));
    let [Replay::Started(_), Replay::Capture(first), Replay::Capture(second)] = replay.as_slice()
    else {
        panic!("{replay:?}")
    };
    assert_eq!((first.samples_per_channel, second.samples_per_channel), (300, 200));
    let report = progressed(9_000, 0, 500);
    assert_eq!(audio.told(&report), vec![report.clone()]);
    audio.observe(&report);
    // Reported, so its finish adds nothing the sink did not say.
    assert_eq!(audio.told(&finished(500)), vec![finished(500)]);
}

// Catches: a finished segment's audio kept, and replayed to the next recorder.
#[test]
fn a_finished_segment_is_forgotten() {
    let mut audio = playing(Some(100));
    audio.observe(&PlaybackEvent::Finished {
        segment: SEGMENT,
        event: PlaybackFinished {
            playback_position: ms(500),
            interrupted: false,
            synchronized_transcript: None,
        },
    });
    assert_eq!(audio.join(at(2_000)), Vec::new());
}

// Catches: seed 107's hole. A sink that reports progress, its segment started long ago but held
// (nothing to play yet, or paused), taken as having played it: the recorder attached then got
// nothing, and the progress reported later placed silence where the person heard the audio.
#[test]
fn a_reporting_sink_with_no_report_yet_has_played_nothing() {
    let mut audio = playing(None);
    let earlier = PlaybackSegment { attachment: 1, segment: 0 };
    audio.observe(&PlaybackEvent::Progressed(PlaybackProgressed {
        segment: earlier,
        started_at: at(0),
        offset: ms(0),
        duration: ms(100),
    }));
    let replay = audio.join(at(30_000));
    let [Replay::Started(started), Replay::Capture(first), Replay::Capture(second)] =
        replay.as_slice()
    else {
        panic!("{replay:?}")
    };
    assert_eq!(
        (*started, first.samples_per_channel, second.samples_per_channel),
        (at(30_000), 300, 200)
    );
    // its first report is told as it came: the recorder places the audio where it played
    assert_eq!(audio.told(&progressed(30_100, 0, 100)), vec![progressed(30_100, 0, 100)]);
}

#[test]
fn a_delayed_progress_report_discards_only_audio_confirmed_before_the_join() {
    let mut audio = playing(Some(100));
    audio.join(at(1_175));
    // The recorder kept the unconfirmed audio rather than guessing it played. This report
    // confirms 75 ms before the join and 25 ms after it, at the segment's original offsets.
    assert_eq!(audio.told(&progressed(1_100, 100, 100)), vec![progressed(1_175, 75, 25)]);
    // A report entirely before attachment must not enter the recording.
    assert!(audio.told(&progressed(1_000, 0, 150)).is_empty());
}
