use super::*;

pub(crate) fn kwargs(turn_detection: Option<TurnDetectionKwarg>) -> SessionKwargs {
    SessionKwargs {
        turn_handling: Some(TurnHandlingKwargs { turn_detection, ..Default::default() }),
        stt_context_options: HostValue::NotGiven,
        max_tool_steps: 3,
        use_tts_aligned_transcript: Maybe::NotGiven,
        tts_text_transforms: Maybe::NotGiven,
        min_consecutive_speech_delay: 0.0,
        expressive: Default::default(),
        aec_warmup_duration: Maybe::NotGiven,
        ivr_detection: false,
        user_away_timeout: Some(15.0),
        transcription_timeout: None,
        session_close_transcript_timeout: 2.0,
        conn_options: Default::default(),
    }
}

fn legs() -> SessionLegs {
    SessionLegs { stt: true, vad: true, llm: true, tts: true, ..SessionLegs::none() }
}

// Catches: `session.options` before `start()` missing an `update_options` or the outbound-SIP
// warmup switch, or a detector object passed to `update_options` changing the resolved mode.
#[test]
fn options_read_before_start_follow_every_update() {
    let options = HostOptions::new(kwargs(None));
    assert!(!options.turn_detection_explicit());
    assert_eq!(options.resolved(legs()).aec_warmup_duration, Some(3.0));
    options.set_aec_warmup(None);
    assert_eq!(options.resolved(legs()).aec_warmup_duration, None);

    let mode = |options: &HostOptions| options.resolved(legs()).turn_detection_mode;
    options.update(UpdateKwargs {
        turn_detection: Maybe::Given(TurnDetectionKwarg::Mode(TurnDetection::Manual)),
        expressive: Maybe::Given(Expressive::default()),
        ..Default::default()
    });
    assert_eq!(mode(&options).as_deref(), Some("manual"));
    options.update(UpdateKwargs {
        turn_detection: Maybe::Given(TurnDetectionKwarg::Detector { streaming: true }),
        ..Default::default()
    });
    assert_eq!(mode(&options).as_deref(), Some("manual"));
    options.update(UpdateKwargs {
        turn_detection: Maybe::Given(TurnDetectionKwarg::Null),
        ..Default::default()
    });
    assert_ne!(mode(&options).as_deref(), Some("manual"));

    let named = HostOptions::new(kwargs(Some(TurnDetectionKwarg::Mode(TurnDetection::Vad))));
    assert!(named.turn_detection_explicit());
}

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
    resolve_interruption_detector_for_host(
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
        resolve_interruption_detector_for_host(
            None,
            None,
            false,
            false,
            true,
            Some("vad"),
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
    let realtime = resolve_interruption_detector_for_host(
        None, None, true, true, true, None, true, None, None, false, true,
    );
    assert_eq!(realtime.reason.as_deref(), Some("cannot_gatekeep"));
}

// Catches: the host's read of `_rt_turn_detection_enabled` parsing its mode string differently
// from the driver's, so the server keeps detecting turns a local VAD was asked to own.
#[test]
fn the_server_keeps_turn_detection_unless_a_local_vad_takes_it() {
    let rt = |caps, vad, mode, interruption| {
        rt_server_turn_detection_for_host(caps, true, vad, mode, false, interruption)
    };
    assert!(rt(true, true, None, false));
    assert!(!rt(true, true, Some("vad"), false));
    assert!(!rt(true, true, None, true));
    assert!(rt(true, false, Some("vad"), false));
    assert!(!rt(true, true, Some("manual"), false));
    assert!(rt(true, true, Some("realtime_llm"), true));
    assert!(!rt(false, true, None, false));
}
