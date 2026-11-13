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
    assert_eq!(*started, at(1_350));
    assert_eq!(rest.samples_per_channel, 150);
    // The next report is told from where the recorder joined.
    let told = audio.told(&progressed(1_400, 350, 100));
    assert_eq!(told, Some(progressed(1_400, 0, 100)));
}

// Catches: a sink that reports no progress taken as having played nothing.
#[test]
fn a_silent_sink_has_played_since_it_started() {
    let mut audio = playing(None);
    let replay = audio.join(at(1_100));
    let [Replay::Started(started), Replay::Capture(first), Replay::Capture(second)] =
        replay.as_slice()
    else {
        panic!("{replay:?}")
    };
    assert_eq!(
        (*started, first.samples_per_channel, second.samples_per_channel),
        (at(1_100), 200, 200)
    );
    let finished = PlaybackEvent::Finished {
        segment: SEGMENT,
        event: PlaybackFinished {
            playback_position: ms(500),
            interrupted: false,
            synchronized_transcript: None,
        },
    };
    let Some(PlaybackEvent::Finished { event, .. }) = audio.told(&finished) else { panic!() };
    assert_eq!(event.playback_position, ms(400));
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
        (at(1_000), 300, 200)
    );
    // its first report is told as it came: the recorder places the audio where it played
    assert_eq!(audio.told(&progressed(30_100, 0, 100)), Some(progressed(30_100, 0, 100)));
}
