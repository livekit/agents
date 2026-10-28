use lk_agents_abi::Ts;
use lk_agents_abi::voice::amd::{AmdCategory, TranscriptSource};
use lk_agents_voice::amd::classifier::{
    Amd, AmdConfig, AmdTimer, Effect, Human, In, Start, Verdict,
};
use std::time::Duration;

#[test]
fn a_transcript_replaces_the_short_greeting_deadline_without_extending_the_silence() {
    let config = AmdConfig {
        human_silence_threshold: Duration::from_millis(100),
        machine_silence_threshold: Duration::from_millis(300),
        ..AmdConfig::default()
    };
    let (amd, _) = Amd::attached(config);
    let (amd, _) = amd.step(Ts::ZERO, In::Start(Start::Listening));
    let (amd, _) = amd.step(Ts::ZERO, In::Human(Human::SpeechStarted));
    let (amd, _) = amd.step(
        Ts::from_nanos(50_000_000),
        In::Human(Human::SpeechEnded { silence: Duration::ZERO }),
    );
    let (_, output) = amd.step(
        Ts::from_nanos(90_000_000),
        In::Human(Human::Transcript { text: "hello".into(), source: TranscriptSource::SessionStt }),
    );
    assert!(output.effects.contains(&Effect::Cancel(AmdTimer::Silence)));
    assert!(output.effects.contains(&Effect::Arm(AmdTimer::Silence, Ts::from_nanos(350_000_000))));
    assert!(!output.effects.iter().any(|effect| matches!(effect, Effect::Prediction(_))));
}

#[test]
fn a_machine_verdict_waits_for_both_the_silence_and_the_end_of_turn_in_either_order() {
    for eot_first in [false, true] {
        let (amd, _) = Amd::attached(AmdConfig::default());
        let (amd, _) = amd.step(Ts::ZERO, In::Start(Start::Listening));
        let (amd, _) = amd.step(
            Ts::ZERO,
            In::Human(Human::Transcript {
                text: "you have reached voicemail".into(),
                source: TranscriptSource::SessionStt,
            }),
        );
        let (amd, _) = amd.step(Ts::ZERO, In::Human(Human::SpeechStarted));
        let (amd, _) = amd.step(
            Ts::from_nanos(1_000_000_000),
            In::Human(Human::SpeechEnded { silence: Duration::ZERO }),
        );
        let (amd, output) = amd.step(
            Ts::from_nanos(1_100_000_000),
            In::Verdict(Verdict::Llm {
                category: AmdCategory::MachineVm,
                transcript: "you have reached voicemail".into(),
            }),
        );
        assert!(!output.effects.iter().any(|effect| matches!(effect, Effect::Prediction(_))));
        let first =
            if eot_first { In::Human(Human::EndOfTurn) } else { In::Timeout(AmdTimer::Silence) };
        let first_at = if eot_first { 2_000_000_000 } else { 2_500_000_000 };
        let (amd, output) = amd.step(Ts::from_nanos(first_at), first);
        assert!(!output.effects.iter().any(|effect| matches!(effect, Effect::Prediction(_))));
        let second =
            if eot_first { In::Timeout(AmdTimer::Silence) } else { In::Human(Human::EndOfTurn) };
        let (_, output) = amd.step(Ts::from_nanos(3_000_000_000), second);
        let predictions: Vec<_> = output
            .effects
            .iter()
            .filter_map(|effect| match effect {
                Effect::Prediction(prediction) => Some(prediction.category),
                _ => None,
            })
            .collect();
        assert_eq!(predictions, [AmdCategory::MachineVm]);
    }
}
