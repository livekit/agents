use super::*;
use lk_agents_abi::voice::session_options::InterruptionGateFunctions;

// Catches: a mode the legs cannot honour kept instead of falling as `Auto` does.
#[test]
fn an_unhonourable_mode_falls_as_auto() {
    let none = LegSet::from(SessionLegs::none());
    let vad = LegSet::from(SessionLegs { vad: true, ..SessionLegs::none() });
    assert_eq!(resolve_turn_detection_mode(TurnDetection::Vad, &vad), TurnDetectionMode::Vad);
    assert_eq!(
        resolve_turn_detection_mode(TurnDetection::Vad, &none),
        resolve_turn_detection_mode(TurnDetection::Auto, &none)
    );
}

fn gate(
    allow_interruptions: bool,
    asked: Option<InterruptionMode>,
    hosted: bool,
    dev_mode: bool,
) -> InterruptionGateVerdict {
    <InterruptionGateRules as InterruptionGateFunctions>::resolve_interruption_detector(
        Some(true),
        Some(true),
        false,
        false,
        true,
        None,
        allow_interruptions,
        asked,
        None,
        hosted,
        dev_mode,
    )
}

// Catches: the host's gate answering a different reason than `_resolve_interruption_detection`,
// or losing the sentence the reference logs with it.
#[test]
fn the_interruption_gate_names_its_reason() {
    assert!(gate(true, None, false, true).build);
    assert!(gate(true, Some(InterruptionMode::Adaptive), true, false).build);
    assert_eq!(gate(false, None, false, true).reason.as_deref(), Some("interruptions_disabled"));
    assert_eq!(
        gate(true, Some(InterruptionMode::Vad), false, true).reason.as_deref(),
        Some("asked_vad")
    );
    let production = gate(true, None, false, false);
    assert_eq!(production.reason.as_deref(), Some("production_default"));
    assert_eq!(production.level.as_deref(), Some("info"));
    assert_eq!(production.message.as_deref(), Some(ADAPTIVE_OFF_IN_PRODUCTION));

    // No aligned streaming STT: adaptive asked for is warned about, unasked is silent.
    let cannot = |asked| {
        <InterruptionGateRules as InterruptionGateFunctions>::resolve_interruption_detector(
            None,
            None,
            false,
            false,
            true,
            Some("vad".to_owned()),
            true,
            asked,
            None,
            false,
            true,
        )
    };
    let warned = cannot(Some(InterruptionMode::Adaptive));
    assert_eq!(warned.reason.as_deref(), Some("cannot_gatekeep"));
    assert_eq!(warned.level.as_deref(), Some("warning"));
    assert_eq!(cannot(None).level, None);
    // A realtime model detecting turns itself leaves nothing to gatekeep with.
    let realtime =
        <InterruptionGateRules as InterruptionGateFunctions>::resolve_interruption_detector(
            None, None, true, true, true, None, true, None, None, false, true,
        );
    assert_eq!(realtime.reason.as_deref(), Some("cannot_gatekeep"));
}

// Catches: the host's read of `_rt_turn_detection_enabled` parsing its mode string differently
// from the driver's, so the server keeps detecting turns a local VAD was asked to own.
#[test]
fn the_server_keeps_turn_detection_unless_a_local_vad_takes_it() {
    let rt = |caps, vad, mode: Option<&str>, interruption| {
        <InterruptionGateRules as InterruptionGateFunctions>::rt_server_turn_detection(
            caps,
            true,
            vad,
            mode.map(str::to_owned),
            false,
            interruption,
        )
    };
    assert!(rt(true, true, None, false));
    assert!(!rt(true, true, Some("vad"), false));
    assert!(!rt(true, true, None, true));
    assert!(rt(true, false, Some("vad"), false));
    assert!(!rt(true, true, Some("manual"), false));
    assert!(rt(true, true, Some("realtime_llm"), true));
    assert!(!rt(false, true, None, false));
}
