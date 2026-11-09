use super::*;
use lk_agents_abi::voice::events::DiagnosticLevel;
use lk_agents_abi::voice::session_options::InterruptionGateFunctions;

#[test]
fn an_unhonourable_mode_falls_as_automatic_selection() {
    let none = SessionLegs::none();
    let vad = SessionLegs { vad: true, ..SessionLegs::none() };
    assert_eq!(resolve_turn_detection_mode(Some(TurnDetection::Vad), &vad), TurnDetectionMode::Vad);
    assert_eq!(
        resolve_turn_detection_mode(Some(TurnDetection::Vad), &none),
        resolve_turn_detection_mode(None, &none)
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
    assert_eq!(production.level, Some(DiagnosticLevel::Info));
    assert_eq!(production.message.as_deref(), Some(ADAPTIVE_OFF_IN_PRODUCTION));

    // No aligned streaming STT: adaptive asked for is warned about, unasked is silent.
    let cannot = |asked| {
        <InterruptionGateRules as InterruptionGateFunctions>::resolve_interruption_detector(
            None,
            None,
            false,
            false,
            true,
            Some(TurnDetection::Vad),
            true,
            asked,
            None,
            false,
            true,
        )
    };
    let warned = cannot(Some(InterruptionMode::Adaptive));
    assert_eq!(warned.reason.as_deref(), Some("cannot_gatekeep"));
    assert_eq!(warned.level, Some(DiagnosticLevel::Warning));
    assert_eq!(cannot(None).level, None);
    // A realtime model detecting turns itself leaves nothing to gatekeep with.
    let realtime =
        <InterruptionGateRules as InterruptionGateFunctions>::resolve_interruption_detector(
            None, None, true, true, true, None, true, None, None, false, true,
        );
    assert_eq!(realtime.reason.as_deref(), Some("cannot_gatekeep"));
}

// The host's read must match the driver's, or the server keeps turns a local VAD should own.
#[test]
fn the_server_keeps_turn_detection_unless_a_local_vad_takes_it() {
    let rt = |caps, vad, mode: Option<TurnDetection>, interruption| {
        <InterruptionGateRules as InterruptionGateFunctions>::rt_server_turn_detection(
            caps,
            true,
            vad,
            mode,
            false,
            interruption,
        )
    };
    assert!(rt(true, true, None, false));
    assert!(!rt(true, true, Some(TurnDetection::Vad), false));
    assert!(!rt(true, true, None, true));
    assert!(rt(true, false, Some(TurnDetection::Vad), false));
    assert!(!rt(true, true, Some(TurnDetection::Manual), false));
    assert!(rt(true, true, Some(TurnDetection::RealtimeLlm), true));
    assert!(!rt(false, true, None, false));
}

#[test]
fn an_override_wins_leaf_by_leaf_until_its_own_release() {
    use crate::agent_session::machine::activity::DriverConfig;
    use std::time::Duration;
    let ms = Duration::from_millis;
    let endpointing = |min: Option<u64>, max: Option<u64>| {
        let mut options = AgentSessionOptions::default();
        options.turn.endpointing.min_delay = min.map(ms);
        options.turn.endpointing.max_delay = max.map(ms);
        options
    };
    let stack = ConfigStack {
        session: endpointing(Some(300), Some(2000)),
        agent: Some(endpointing(Some(500), None)),
        overrides: Vec::new(),
    };
    let mut config = DriverConfig::new(stack, SessionLegs::none());
    let delays = |config: &DriverConfig| {
        let e = config.effective().turn.endpointing;
        (e.min_delay, e.max_delay)
    };
    config.push_override(OverrideLayer { id: 1, options: endpointing(Some(900), None) });
    config.push_override(OverrideLayer { id: 2, options: endpointing(Some(700), None) });
    assert_eq!(delays(&config), (Some(ms(700)), Some(ms(2000))), "the topmost layer wins");
    config.configure(&endpointing(None, Some(4000)));
    assert_eq!(delays(&config), (Some(ms(700)), Some(ms(4000))), "the base shows through");
    config.release_override(1);
    assert_eq!(delays(&config), (Some(ms(700)), Some(ms(4000))), "the inner layer stays");
    config.release_override(2);
    assert_eq!(delays(&config), (Some(ms(500)), Some(ms(4000))), "the agent layer is back");
}

// The agent's given `None` wins over the session's mode rather than deferring to it.
#[test]
fn an_agents_turn_detection_none_is_automatic_selection() {
    let kwargs = TurnHandlingKwargs {
        turn_detection: Maybe::Given(TurnDetectionKwarg::Null),
        ..TurnHandlingKwargs::default()
    };
    let mut session = AgentSessionOptions::default();
    session.turn.detection = Maybe::Given(Some(TurnDetection::Stt));
    let stack = ConfigStack {
        session,
        agent: Some(crate::turn::agent_layer(Some(&kwargs), None, None)),
        overrides: Vec::new(),
    };
    let legs = SessionLegs { stt: true, vad: true, ..SessionLegs::none() };
    assert_eq!(stack.resolve(&legs).turn.detection, Maybe::Given(None));
}

#[test]
fn updates_cannot_reselect_interruption_mode_even_before_start() {
    use lk_agents_abi::voice::session_options::InterruptionKwargs;
    let with_mode = |mode| UpdateKwargs {
        interruption: Some(InterruptionKwargs { mode, ..Default::default() }),
        ..Default::default()
    };
    assert!(validate_session_update(&UpdateKwargs::default()).is_ok());
    assert!(validate_session_update(&with_mode(Maybe::NotGiven)).is_ok());
    // A host's explicit null still counts as supplying the construction-only key.
    assert!(validate_session_update(&with_mode(Maybe::Given(None))).is_err());
    let update = with_mode(Maybe::Given(Some(InterruptionMode::Vad)));
    assert!(validate_session_update(&update).is_err());
    assert!(running_layer(&update).is_err());
}
