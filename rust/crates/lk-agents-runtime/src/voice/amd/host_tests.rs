use super::*;
use lk_agents_voice::amd::events::IvrMenuOption;

fn host_options() -> abi::AmdOptions {
    let machine = AmdOptions::default();
    abi::AmdOptions {
        wait_until_answered: machine.wait_until_answered,
        screening_instructions: machine.screening_instructions,
        voicemail_instructions: machine.voicemail_instructions,
        ivr_instructions: machine.ivr_instructions,
        human_instructions: machine.human_instructions,
        opening_text: None,
        opening_instructions: None,
        idle_timeout: 10.0,
        voicemail_idle_timeout: 10.0,
        timeout: 60.0,
        inference_timeout: 2.0,
        machine_silence_threshold: 1.5,
        max_uncertain_turns: 3,
        max_inference_timeouts: 3,
    }
}

// Catches: the host's seconds and counts reaching the machine as other values than `main`'s
// defaults.
#[test]
fn main_defaults_convert_to_the_machine_defaults() {
    assert_eq!(options(&host_options()).expect("valid"), AmdOptions::default());
}

// Catches: a value `AMD.__init__` refuses accepted, or refused with another message. A negative
// float cannot be a `Duration`, so the conversion owns that check.
#[test]
fn invalid_options_raise_mains_messages() {
    let positive = "amd timeouts and max_uncertain_turns must be positive";
    type Change = fn(&mut abi::AmdOptions);
    let cases: Vec<(Change, &str)> = vec![
        (|o| o.timeout = 0.0, positive),
        (|o| o.idle_timeout = -1.0, positive),
        (|o| o.voicemail_idle_timeout = f64::NAN, positive),
        (|o| o.max_uncertain_turns = 0, positive),
        (|o| o.max_inference_timeouts = -2, "max_inference_timeouts must be positive"),
        (|o| o.machine_silence_threshold = -0.5, "machine_silence_threshold must be non-negative"),
        (|o| o.opening_text = Some(" ".into()), "opening_text must not be empty"),
    ];
    for (change, message) in cases {
        let mut host = host_options();
        change(&mut host);
        assert_eq!(options(&host), Err(Invalid(message.into())));
    }
    let mut zero = host_options();
    zero.machine_silence_threshold = 0.0;
    assert_eq!(
        options(&zero).expect("zero disables the hold").machine_silence_threshold,
        Duration::ZERO
    );
}

// Catches: a session refusal raised as the wrong Python class: `main` raises `RuntimeError` for
// the session's state and `ValueError` for a model that does not fit it.
#[test]
fn refusals_keep_mains_exception_classes() {
    let runtime = [AmdAttachRefused::AlreadyAttached, AmdAttachRefused::HandoffInFlight];
    for why in runtime {
        assert!(matches!(refused(why.into()), abi::AmdEnterRefused::Refused(_)));
    }
    let value = [
        AmdAttachRefused::RealtimeServerTurnDetection,
        AmdAttachRefused::RealtimeTools,
        AmdAttachRefused::RealtimeNoStt,
    ];
    for why in value {
        assert!(matches!(refused(why.into()), abi::AmdEnterRefused::Invalid(_)));
    }
    assert!(matches!(refused(AmdEnterError::NotStreaming), abi::AmdEnterRefused::Invalid(_)));
    assert_eq!(
        refused(AmdAttachRefused::Closing.into()),
        abi::AmdEnterRefused::Session(SessionRefused::Closed)
    );
}

// Catches: an event field lost or mis-scaled between the machine and the host's record.
#[test]
fn events_keep_every_field_in_seconds() {
    let prediction = Prediction {
        turn_id: 3,
        category: Category::Wait,
        stage: Stage::MachineVm,
        reason: AmdReason::Prediction,
        transcript: "please hold".into(),
        speech_duration: Duration::from_millis(1500),
        delay: Duration::from_millis(250),
        inference_duration: Some(Duration::from_millis(400)),
    };
    let outcome = Outcome {
        category: Stage::MachineVm,
        reason: AmdReason::IdleTimeout,
        voicemail_message_played: true,
        last_prediction: Some(prediction.clone()),
    };
    let host = completed(&outcome);
    assert_eq!(host.category, abi::AmdCategory::MachineVm);
    assert_eq!(host.reason, abi::AmdReason::IdleTimeout);
    assert!(host.voicemail_message_played);
    let last = host.last_prediction.expect("the last prediction");
    assert_eq!((last.category, last.stage), (abi::AmdCategory::Wait, abi::AmdCategory::MachineVm));
    assert_eq!((last.speech_duration, last.delay), (1.5, 0.25));
    assert_eq!(last.inference_duration, Some(0.4));
    assert_eq!((last.turn_id, last.transcript.as_str()), (3, "please hold"));

    let menu = MenuObserved {
        session_id: "amd_1".into(),
        turn_id: 2,
        menu: "main".into(),
        options: vec![IvrMenuOption {
            label: "sales".into(),
            dtmf: "1".into(),
            ..Default::default()
        }],
        extraction_duration: Duration::from_millis(100),
    };
    let abi::AmdEvent::MenuObserved(menu) = event(&AmdEvent::MenuObserved(menu)) else {
        panic!("a menu stays a menu");
    };
    assert_eq!(menu.options[0].dtmf, "1");
    assert_eq!(menu.extraction_duration, 0.1);
}

// Catches: a reason's error flag drifting from `main`'s `_ERROR_REASONS`.
#[test]
fn only_mains_error_reasons_are_errors() {
    use abi::AmdRules as _;
    let errors = [
        abi::AmdReason::InternalError,
        abi::AmdReason::ParticipantMissing,
        abi::AmdReason::AgentChanged,
        abi::AmdReason::Cancelled,
    ];
    for reason in abi::AmdReason::ALL {
        let event = abi::AmdCompletedEvent {
            category: abi::AmdCategory::Uncertain,
            reason,
            voicemail_message_played: false,
            last_prediction: None,
        };
        assert_eq!(Rules::completed_is_error(event), errors.contains(&reason), "{reason:?}");
    }
}
