use super::*;
use lk_agents_abi::voice::rtc::{AudioSource as _, RtcRoom as _, TrackPublishOptions};
use lk_agents_runtime::rtc::events::RoomEventBus;
use std::time::Duration;

fn frame(tag: i16, samples: u32) -> AudioFrame {
    AudioFrame::new(tag.to_le_bytes().repeat(samples as usize), RATE, 1, samples)
}

/// The oracle must detect corrupted recordings even when their own callbacks are consistent.
#[tokio::test]
async fn the_speaker_oracle_rejects_wrong_missing_shifted_and_unplayed_pcm() {
    let (rtc, clock) = super::super::room::FakeRtc::with_virtual_clock(RoomEventBus::port);
    let room = rtc.room("recording-oracle");
    room.local()
        .publish_audio_track(
            &lk_agents_abi::telemetry::Cx::NONE,
            TrackPublishOptions { sample_rate: RATE, ..Default::default() },
        )
        .await
        .unwrap();
    let source = room.local_participant().sources()[0].clone();
    source.capture(&lk_agents_abi::telemetry::Cx::NONE, &frame(1000, 960)).await.unwrap();
    clock.advance(Duration::from_millis(20));
    let speaker = RoomSpeaker { room, anchor: (0.0, Ts::ZERO) };

    for (tag, at, duration, matches) in [
        (1000, 0.0, 0.02, true),
        (2000, 0.0, 0.02, false), // wrong audio
        (1000, 0.1, 0.02, false), // right audio at the wrong time
        (1000, 0.0, 0.01, false), // missing played audio
        (1000, 0.0, 0.04, false), // audio that never played
    ] {
        let mut recording = Recording::default();
        for command in [
            RecordingCommand::Capture(0.0, frame(tag, 1920)),
            RecordingCommand::PlaybackStarted(at),
            RecordingCommand::Progress(at, 0.0, duration),
            RecordingCommand::PlaybackFinished(at + duration, duration),
            RecordingCommand::Flush(2.0),
        ] {
            recording.file.apply(command, &mut || Ok(())).unwrap();
        }
        assert_eq!(speaker.check(&recording).is_none(), matches, "{tag} at {at} for {duration}");
    }
}
