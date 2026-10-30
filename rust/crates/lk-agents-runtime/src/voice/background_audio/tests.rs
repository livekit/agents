use super::*;

// Catches: a stop with a fade-out reporting the sound done at once (the generator would be closed
// mid-fade), or a stop on a finished sound reopening it.
#[tokio::test]
async fn a_stop_waits_out_the_fade() {
    let fading = PlaybackDone::new(0.5);
    assert!(!fading.stop());
    assert!(fading.stop_requested() && !fading.done());
    assert!(fading.finish());
    assert!(!fading.finish());
    fading.wait().await;
    assert!(!fading.stop());

    let abrupt = PlaybackDone::new(0.0);
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
