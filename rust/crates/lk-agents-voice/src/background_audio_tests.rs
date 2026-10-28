use super::*;
use crate::agent_session::FixedRoom;

fn file(path: &str, probability: f64) -> Sound {
    Sound::for_host(Some(path.to_owned()), None, 1.0, probability, 0.0, 0.0)
}

/// A host stream of `frames` short frames.
#[allow(clippy::disallowed_types, reason = "S0: existing test harness live resources")]
struct Frames(Mutex<u32>);

impl FrameStream for Frames {
    fn next(&self) -> lk_agents_abi::BoxFuture<'_, Result<Option<AudioFrame>, ProviderError>> {
        let mut left = self.0.lock().unwrap();
        let frame = (*left > 0).then(|| {
            *left -= 1;
            AudioFrame::new(1000i16.to_le_bytes().repeat(480), 48_000, 1, 480)
        });
        Box::pin(std::future::ready(Ok(frame)))
    }
    fn cancel_pending(&self) {}
    fn close(&self) -> lk_agents_abi::BoxFuture<'_, ()> {
        Box::pin(std::future::ready(()))
    }
}

async fn started() -> (Arc<BackgroundAudioPlayer>, Arc<lk_agents_rtc::FakeRoom>) {
    let (rtc, clock) = lk_agents_rtc::FakeRtc::with_virtual_clock();
    let room = rtc.room("background");
    std::mem::forget(rtc);
    let fixed = FixedRoom {
        rtc: room.clone(),
        time: crate::agent_session::time::Timebase::from_virtual(clock),
    };
    let player = Arc::new(BackgroundAudioPlayer::for_host(48_000, 1, 4_800, 200));
    player.start_for_host(Arc::new(fixed), 0.5, Vec::new(), true, None).await.unwrap();
    (player, room)
}

// Catches: a sound played before `start()` or after `aclose()` queued where nothing mixes it, and
// a looping iterator accepted (it can only play once).
#[allow(clippy::disallowed_types, reason = "S0: existing test harness live resources")]
#[tokio::test(start_paused = true)]
async fn play_refuses_what_cannot_play() {
    let idle = BackgroundAudioPlayer::for_host(48_000, 1, 4_800, 200);
    assert!(matches!(
        idle.play_for_host(vec![file("a.ogg", 1.0)], 0.5, false, false, true),
        Err(PlaybackError::NotStarted)
    ));

    let (player, _room) = started().await;
    let stream = Sound::for_host(None, Some(Arc::new(Frames(Mutex::new(1)))), 1.0, 1.0, 0.0, 0.0);
    let looping = player.play_for_host(vec![stream], 0.5, true, false, true);
    assert!(matches!(looping, Err(PlaybackError::LoopingStream(_))));
    assert!(LoopingStreamError.to_string().contains("loop=False"));
    // The draw picked silence: the handle is born done.
    let silent = player.play_for_host(vec![file("a.ogg", 0.0)], 0.5, false, false, true).unwrap();
    assert!(silent.done());

    player.close().await;
    assert!(player.closed());
    assert!(matches!(
        player.play(vec![file("a.ogg", 1.0)], false, false, false, 0.5),
        Err(PlaybackError::Closed)
    ));
}

// Catches: a stop with a fade-out reporting the sound done at once (the generator would be closed
// mid-fade), or a stop on a finished sound reopening it.
#[tokio::test]
async fn a_stop_waits_out_the_fade() {
    let fading = PlaybackHandle::for_host(0.5);
    assert!(!fading.stop());
    assert!(fading.stop_requested() && !fading.done());
    assert!(fading.finish());
    assert!(!fading.finish());
    fading.wait().await;
    assert!(!fading.stop());

    let abrupt = PlaybackHandle::new(0.0);
    assert!(abrupt.stop());
    abrupt.wait().await;
}

// Catches: a builtin clip naming a file the package does not ship, which fails only when a call
// first plays it.
#[test]
fn every_builtin_clip_is_shipped() {
    let resources = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../livekit-agents-next/livekit/agents/resources");
    use BuiltinAudioClip as C;
    for clip in [
        C::CityAmbience,
        C::ForestAmbience,
        C::OfficeAmbience,
        C::CrowdedRoom,
        C::KeyboardTyping,
        C::KeyboardTyping2,
        C::HoldMusic,
    ] {
        assert!(
            resources.join(clip.file_name()).is_file(),
            "{clip:?}: {} is not shipped",
            clip.file_name()
        );
    }
}
