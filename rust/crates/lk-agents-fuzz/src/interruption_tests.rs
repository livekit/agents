//! Interruption gates: what user speech may cut the agent, and what it may not.

use lk_agents_abi::{ChatItem, Role};
use lk_agents_fuzz::drive::{RoomLegs, Run, Setup, simulate};
use lk_agents_fuzz::input::{
    AgentText, Confidence, Count, Input, Knob, Millis, Ref, SayInput, TurnMode, UserText,
};
use lk_agents_fuzz::record::Kind;
use lk_agents_voice::events::SessionEvent;

/// A reply prepared on a preflight transcript plays no audio while the caller resumes speaking.
/// https://github.com/livekit/agents/pull/6745 https://github.com/livekit/agents/issues/7948
#[test]
fn resumed_user_speech_holds_pending_audio() {
    let setup =
        Setup { vad: true, stt: true, preemptive: true, turn: TurnMode::Stt, ..Setup::default() };
    let run = simulate(async {
        let mut run = Run::start(&setup, 0, Default::default()).await;
        for input in [
            Input::VadStart { vad: Ref::new(0), speech_ms: Millis(300) },
            Input::SttStart { stt: Ref::new(0) },
            Input::SttFinal {
                utterance: Ref::new(0),
                text: UserText("I would like".into()),
                speaker: None,
            },
            Input::VadEnd { vad: Ref::new(0), silence_ms: Millis(250) },
            Input::SttPreflight {
                utterance: Ref::new(0),
                text: UserText("I would like".into()),
                speaker: None,
            },
            Input::Advance { ms: Millis(100) },
            Input::VadStart { vad: Ref::new(0), speech_ms: Millis(300) },
            Input::LlmText { chat: Ref::new(0), text: AgentText("How can I help?".into()) },
            Input::LlmEnd { chat: Ref::new(0) },
            Input::Advance { ms: Millis(1_000) },
        ] {
            run.apply(&input).await;
        }
        run
    });
    assert_eq!(run.record.resources(Kind::Chat).count(), 1, "a speculative reply was generated");
    assert!(
        !run.record
            .sink()
            .iter()
            .any(|(_, sink)| matches!(sink, lk_agents_fuzz::record::Sink::Audio { .. })),
        "audio escaped during resumed speech"
    );
}

/// Interim words interrupt the agent only once local VAD confirms the minimum speech, whatever
/// the word gate. https://github.com/livekit/agents/pull/6599 https://github.com/livekit/agents/issues/3515
#[test]
fn interim_words_wait_for_local_vad_before_interrupting() {
    for min_words in [0, 3] {
        let setup = Setup { stt: true, vad: true, ..Setup::default() };
        let run = simulate(async {
            let mut run = Run::start(&setup, 0, Default::default()).await;
            for input in [
                Input::Configure {
                    knob: Knob::MinInterruptionWords,
                    mode: TurnMode::Default,
                    ms: Millis(0),
                    on: true,
                    words: Count(min_words),
                },
                Input::Say {
                    text: AgentText("Hold the line.".into()),
                    input: SayInput::Audio,
                    allow: None,
                    to_ctx: true,
                },
                Input::FeedAudio { stream: Ref::new(0), chunks: Count(400) },
                Input::Advance { ms: Millis(100) },
                Input::PlayoutStarted { segment: Ref::new(0) },
                Input::Advance { ms: Millis(4_000) },
                Input::SttStart { stt: Ref::new(0) },
                Input::SttInterim {
                    utterance: Ref::new(0),
                    text: UserText(
                        if min_words == 0 { "wait stop please" } else { "wait stop" }.into(),
                    ),
                    speaker: None,
                },
                Input::Advance { ms: Millis(100) },
                Input::VadStart { vad: Ref::new(0), speech_ms: Millis(100) },
                Input::VadInference {
                    vad: Ref::new(0),
                    probability: Confidence(90),
                    speaking: true,
                    duration: Millis(100),
                },
                Input::Advance { ms: Millis(100) },
                Input::VadInference {
                    vad: Ref::new(0),
                    probability: Confidence(90),
                    speaking: true,
                    duration: Millis(800),
                },
                Input::Advance { ms: Millis(100) },
                Input::SttInterim {
                    utterance: Ref::new(0),
                    text: UserText("wait stop please".into()),
                    speaker: None,
                },
                Input::VadInference {
                    vad: Ref::new(0),
                    probability: Confidence(90),
                    speaking: true,
                    duration: Millis(900),
                },
                Input::Advance { ms: Millis(100) },
            ] {
                run.apply(&input).await;
            }
            run
        });
        let cuts: Vec<_> = run
            .record
            .journal()
            .iter()
            .filter_map(|(at, step)| match step {
                lk_agents_voice::trace::Step::Interrupt(
                    lk_agents_voice::trace::InterruptStep::Interrupted { .. },
                ) => Some(at.at_ms),
                _ => None,
            })
            .collect();
        assert_eq!(cuts.len(), 1, "{cuts:?}");
        let earliest = if min_words == 0 { 4_300 } else { 4_400 };
        assert!(
            cuts[0] >= earliest,
            "interruption bypassed a word or duration gate: min_words={min_words}, {cuts:?}"
        );
    }
}

/// A VAD end of speech without a speech onset (noise) does not cancel the pending reply.
/// https://github.com/livekit/agents/pull/6274 https://github.com/livekit/agents/issues/4243
#[test]
fn noise_only_endpoint_events_keep_the_pending_reply() {
    let setup = Setup { vad: true, stt: true, tts: false, ..Setup::default() };
    let run = simulate(async {
        let mut run = Run::start(&setup, 0, Default::default()).await;
        for input in [
            Input::GenerateReply { user_input: None, instructions: None, allow: None },
            Input::Advance { ms: Millis(100) },
            Input::VadInference {
                vad: Ref::new(0),
                probability: Confidence(0),
                speaking: false,
                duration: Millis(300),
            },
            Input::VadEnd { vad: Ref::new(0), silence_ms: Millis(500) },
            Input::Advance { ms: Millis(100) },
            Input::LlmText {
                chat: Ref::new(0),
                text: AgentText("Your booking is confirmed.".into()),
            },
            Input::LlmEnd { chat: Ref::new(0) },
            Input::Advance { ms: Millis(500) },
        ] {
            run.apply(&input).await;
        }
        run
    });
    assert_eq!(
        run.record
            .host_view()
            .events()
            .into_iter()
            .filter_map(|(_, event)| match event {
                SessionEvent::ConversationItemAdded { item, .. } => item
                    .as_message()
                    .filter(|m| m.role == Role::Assistant)
                    .and_then(|m| m.raw_text_content()),
                _ => None,
            })
            .collect::<Vec<_>>(),
        ["Your booking is confirmed."]
    );
}

/// The next user turn after a resumed false interruption starts at its own speech onset.
/// https://github.com/livekit/agents/issues/7063 https://github.com/livekit/agents/issues/4498
#[test]
#[ignore = "https://github.com/livekit/agents/issues/7063 reproduces: false interruption retains the abandoned speech onset"]
fn false_interruption_discards_the_abandoned_user_anchor() {
    let setup = Setup {
        stt: true,
        vad: true,
        pausable_output: true,
        room_legs: Some(RoomLegs { pre_connect: None }),
        ..Setup::default()
    };
    let mut inputs = vec![
        Input::Advance { ms: Millis(100) },
        Input::Say {
            text: AgentText("A long answer.".into()),
            input: SayInput::Audio,
            allow: None,
            to_ctx: true,
        },
    ];
    for _ in 0..8 {
        inputs.extend([
            Input::FeedAudio { stream: Ref::new(0), chunks: Count(50) },
            Input::Advance { ms: Millis(450) },
        ]);
    }
    inputs.extend([
        Input::VadStart { vad: Ref::new(0), speech_ms: Millis(100) },
        Input::VadInference {
            vad: Ref::new(0),
            probability: Confidence(90),
            speaking: true,
            duration: Millis(800),
        },
        Input::Advance { ms: Millis(100) },
        Input::VadEnd { vad: Ref::new(0), silence_ms: Millis(250) },
        Input::Advance { ms: Millis(3_000) },
        Input::FeedEnd { stream: Ref::new(0) },
        Input::Advance { ms: Millis(5_000) },
        Input::VadStart { vad: Ref::new(0), speech_ms: Millis(100) },
        Input::SttStart { stt: Ref::new(0) },
        Input::Advance { ms: Millis(300) },
        Input::SttFinal {
            utterance: Ref::new(0),
            text: UserText("a fresh question".into()),
            speaker: None,
        },
        Input::VadEnd { vad: Ref::new(0), silence_ms: Millis(250) },
        Input::SttEnd { utterance: Ref::new(0) },
        Input::Advance { ms: Millis(2_000) },
    ]);
    let run = simulate(async {
        let mut run = Run::start(&setup, 0, Default::default()).await;
        for input in &inputs {
            run.apply(input).await;
        }
        run
    });
    assert!(
        run.record.host_view().events().iter().any(|(_, event)| matches!(
            event,
            SessionEvent::AgentFalseInterruption { resumed: true }
        )),
        "must actually resume a false interruption"
    );
    let users: Vec<_> = run
        .record
        .host_view()
        .events()
        .into_iter()
        .filter_map(|(_, event)| match event {
            SessionEvent::ConversationItemAdded { item, .. } => match item.as_ref() {
                ChatItem::Message(message) if message.role == Role::User => Some(message),
                _ => None,
            },
            _ => None,
        })
        .collect();
    assert_eq!(users.len(), 1);
    assert_eq!(users[0].raw_text_content().as_deref(), Some("a fresh question"));
    let started = users[0].metrics.started_speaking_at.expect("fresh onset");
    assert!(started >= 1_704_067_211.0, "stale false-interruption anchor: {started}");
}

/// When the adaptive interruption detector fails, local VAD interrupts the agent at once.
/// https://github.com/livekit/agents/pull/7923 https://github.com/livekit/agents/pull/5142
#[test]
fn failed_adaptive_detection_falls_back_to_local_vad() {
    let setup = Setup { stt: true, vad: true, interruption: true, ..Setup::default() };
    let run = simulate(async {
        let mut run = Run::start(&setup, 0, Default::default()).await;
        for input in [
            Input::Say {
                text: AgentText("A long answer.".into()),
                input: SayInput::Audio,
                allow: None,
                to_ctx: true,
            },
            Input::FeedAudio { stream: Ref::new(0), chunks: Count(400) },
            Input::Advance { ms: Millis(100) },
            Input::PlayoutStarted { segment: Ref::new(0) },
            Input::Advance { ms: Millis(4_000) },
            Input::VadStart { vad: Ref::new(0), speech_ms: Millis(300) },
            Input::Advance { ms: Millis(10) },
            Input::OverlapFail { port: Ref::new(0) },
            Input::Advance { ms: Millis(10) },
            Input::VadInference {
                vad: Ref::new(0),
                probability: Confidence(90),
                speaking: true,
                duration: Millis(800),
            },
            Input::Advance { ms: Millis(100) },
        ] {
            run.apply(&input).await;
        }
        run
    });
    assert_eq!(run.record.resources(Kind::Interruption).count(), 1, "exercise adaptive detection");
    let cuts: Vec<_> = run
        .record
        .journal()
        .iter()
        .filter_map(|(at, step)| match step {
            lk_agents_voice::trace::Step::Interrupt(
                lk_agents_voice::trace::InterruptStep::Interrupted { .. },
            ) => Some(at.at_ms),
            _ => None,
        })
        .collect();
    assert_eq!(cuts.len(), 1, "local VAD must take over without waiting for detector reconnection");
    assert!(cuts[0] <= 4_220, "VAD fallback was delayed: {cuts:?}");
}

/// A filler the recogniser finalizes as nothing leaves the false interruption to resume.
#[test]
fn an_empty_final_transcript_keeps_a_false_interruption_false() {
    let setup = Setup { stt: true, vad: true, pausable_output: true, ..Setup::default() };
    let mut inputs = vec![
        Input::Advance { ms: Millis(100) },
        Input::Say {
            text: AgentText("A long answer.".into()),
            input: SayInput::Audio,
            allow: None,
            to_ctx: true,
        },
    ];
    for _ in 0..8 {
        inputs.extend([
            Input::FeedAudio { stream: Ref::new(0), chunks: Count(50) },
            Input::Advance { ms: Millis(450) },
        ]);
    }
    inputs.extend([
        Input::VadStart { vad: Ref::new(0), speech_ms: Millis(100) },
        Input::VadInference {
            vad: Ref::new(0),
            probability: Confidence(90),
            speaking: true,
            duration: Millis(800),
        },
        Input::SttStart { stt: Ref::new(0) },
        Input::Advance { ms: Millis(100) },
        Input::VadEnd { vad: Ref::new(0), silence_ms: Millis(250) },
        Input::SttFinal { utterance: Ref::new(0), text: UserText(String::new()), speaker: None },
        Input::SttEnd { utterance: Ref::new(0) },
        Input::Advance { ms: Millis(3_000) },
        Input::FeedEnd { stream: Ref::new(0) },
        Input::Advance { ms: Millis(5_000) },
    ]);
    let run = simulate(async {
        let mut run = Run::start(&setup, 0, Default::default()).await;
        for input in &inputs {
            run.apply(input).await;
        }
        run
    });
    let events = run.record.host_view().events();
    assert!(
        events.iter().any(|(_, event)| matches!(
            event,
            SessionEvent::AgentFalseInterruption { resumed: true }
        )),
        "the empty final turned the false interruption into a real one"
    );
}
