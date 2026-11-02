use super::*;
use std::time::{Duration, UNIX_EPOCH};

fn begin(preemptive_tts: bool, authorization_available: bool) -> GenerationPreparationEvent {
    GenerationPreparationEvent::Begin {
        has_tts: true,
        audio_enabled: true,
        has_audio_output: true,
        llm_reply: true,
        preemptive_tts,
        authorization_available,
    }
}

// Speculative synthesis may produce audio before the gate is open, but that does not grant
// permission to deliver either its audio or its deferred transcript.
#[test]
fn speculative_synthesis_starts_without_granting_playback() {
    let mut preparation = GenerationPreparation::default();
    assert!(matches!(preparation.handle(begin(true, false)), GenerationPreparationEffect::Start));
    assert_eq!(preparation.authorization(false), PlaybackAuthorization::Pending);
    assert!(preparation.authorized_at().is_none());
}

// Non-speculative synthesis waits for scheduling readiness; scheduling and playback permission
// are distinct facts. A scheduled generation may synthesize while playback remains pending.
#[test]
fn ordinary_synthesis_waits_for_readiness_not_necessarily_playback() {
    let mut preparation = GenerationPreparation::default();
    assert!(matches!(
        preparation.handle(begin(false, false)),
        GenerationPreparationEffect::WaitForReadiness
    ));
    assert!(matches!(
        preparation.handle(GenerationPreparationEvent::ReadinessResolved {
            ready: true,
            authorization_available: false,
        }),
        GenerationPreparationEffect::Start
    ));
    assert_eq!(preparation.authorization(false), PlaybackAuthorization::Pending);
}

// An already-published authorization must be observed before starting, so its timestamp is
// captured once and reused by each synthesis segment's metrics and playback checks.
#[test]
fn a_published_authorization_is_consumed_before_synthesis_starts() {
    for preemptive in [false, true] {
        let mut preparation = GenerationPreparation::default();
        let effect = preparation.handle(begin(preemptive, true));
        if preemptive {
            assert!(matches!(effect, GenerationPreparationEffect::WaitForPlaybackAuthorization));
        } else {
            assert!(matches!(effect, GenerationPreparationEffect::WaitForReadiness));
            assert!(matches!(
                preparation.handle(GenerationPreparationEvent::ReadinessResolved {
                    ready: true,
                    authorization_available: true,
                }),
                GenerationPreparationEffect::WaitForPlaybackAuthorization
            ));
        }
        assert!(preparation.pending_authorization(true));
        let at = UNIX_EPOCH + Duration::from_secs(10);
        let authorization = preparation.authorize(at);
        assert!(!preparation.pending_authorization(true));
        assert!(matches!(
            preparation.handle(GenerationPreparationEvent::PlaybackResolved {
                authorization,
                authorization_available: true,
            }),
            GenerationPreparationEffect::Start
        ));
        assert_eq!(preparation.authorized_at(), Some(at));
    }
}

// Text-only/caller-audio/no-output paths cannot use speculative TTS permission to bypass the
// generation's playback gate. Their user-visible words must wait too.
#[test]
fn non_synthesis_output_waits_for_playback_authorization() {
    for (has_tts, audio_enabled, has_audio_output) in
        [(false, true, true), (true, false, true), (true, true, false), (false, false, false)]
    {
        let mut preparation = GenerationPreparation::default();
        assert!(matches!(
            preparation.handle(GenerationPreparationEvent::Begin {
                has_tts,
                audio_enabled,
                has_audio_output,
                llm_reply: true,
                preemptive_tts: true,
                authorization_available: false,
            }),
            GenerationPreparationEffect::WaitForPlaybackAuthorization
        ));
        let authorization = preparation.authorize(UNIX_EPOCH);
        assert!(matches!(
            preparation.handle(GenerationPreparationEvent::PlaybackResolved {
                authorization,
                authorization_available: true,
            }),
            GenerationPreparationEffect::Start
        ));
    }
}

// The preemptive-TTS option applies only to model replies, not explicit say/caller requests.
#[test]
fn a_non_model_request_does_not_use_speculative_model_permission() {
    let mut preparation = GenerationPreparation::default();
    assert!(matches!(
        preparation.handle(GenerationPreparationEvent::Begin {
            has_tts: true,
            audio_enabled: true,
            has_audio_output: true,
            llm_reply: false,
            preemptive_tts: true,
            authorization_available: false,
        }),
        GenerationPreparationEffect::WaitForReadiness
    ));
}

// Interruption wins during either awaited phase. A later observation cannot restart preparation
// or manufacture a reply-start timestamp for work that never obtained authorization.
#[test]
fn interruption_during_preparation_does_not_start_the_generation() {
    let mut readiness = GenerationPreparation::default();
    let _ = readiness.handle(begin(false, false));
    assert!(matches!(
        readiness.handle(GenerationPreparationEvent::ReadinessResolved {
            ready: false,
            authorization_available: true,
        }),
        GenerationPreparationEffect::Interrupted
    ));
    assert!(readiness.authorized_at().is_none());
    assert!(matches!(
        readiness.handle(GenerationPreparationEvent::ReadinessResolved {
            ready: true,
            authorization_available: true,
        }),
        GenerationPreparationEffect::Interrupted
    ));

    let mut playback = GenerationPreparation::default();
    let _ = playback.handle(begin(true, true));
    assert!(matches!(
        playback.handle(GenerationPreparationEvent::PlaybackResolved {
            authorization: PlaybackAuthorization::Interrupted,
            authorization_available: true,
        }),
        GenerationPreparationEffect::Interrupted
    ));
    assert!(playback.authorized_at().is_none());
}

// A gate is step-scoped, not segment-scoped: later segments reuse the first observation. An
// interrupted wait can still leave a real earlier authorization timestamp for final metrics.
#[test]
fn segments_reuse_authorization_and_interruption_does_not_erase_it() {
    let mut preparation = GenerationPreparation::default();
    let _ = preparation.handle(begin(true, false));
    let first = UNIX_EPOCH + Duration::from_secs(10);
    assert_eq!(preparation.authorize(first), PlaybackAuthorization::Authorized);
    for _ in 0..3 {
        assert_eq!(preparation.authorization(false), PlaybackAuthorization::Authorized);
        assert_eq!(preparation.authorized_at(), Some(first));
        assert!(!preparation.pending_authorization(true));
    }
    preparation.authorize(first + Duration::from_secs(30));
    assert_eq!(preparation.authorized_at(), Some(first));
    assert_eq!(preparation.authorization(true), PlaybackAuthorization::Interrupted);
    assert_eq!(preparation.authorized_at(), Some(first));
    assert_eq!(
        GenerationPreparation::default().authorization(false),
        PlaybackAuthorization::Pending
    );
}
