//! End of turn: preflight, end-of-turn prediction and endpointing deadlines decide when a user
//! turn commits and which reply answers it.

use lk_agents_abi::Role;
use lk_agents_fuzz::drive::{Run, Setup, simulate};
use lk_agents_fuzz::input::{
    AgentText, Confidence, Count, Input, Knob, Millis, Ref, TurnMode, UserText,
};
use lk_agents_fuzz::record::Kind;
use lk_agents_voice::events::SessionEvent;

/// What `role` said, as the conversation recorded it.
fn said(run: &Run, role: Role) -> Vec<String> {
    let events = run.record.host_view().events();
    let messages = events.into_iter().filter_map(|(_, event)| match event {
        SessionEvent::ConversationItemAdded { item, .. } => item.as_message(),
        _ => None,
    });
    messages.filter(|m| m.role == role).filter_map(|m| m.raw_text_content()).collect()
}

/// A final that only adds punctuation to the preflight transcript reuses the preflight reply, in
/// every script. https://github.com/livekit/agents/issues/7506 https://github.com/livekit/agents/issues/4219
#[test]
#[ignore = "https://github.com/livekit/agents/issues/7506 reproduces: CJK punctuation merging whitespace-separated words discards preflight"]
fn punctuation_only_final_reuses_the_preflight_request() {
    for (preflight, final_text) in [
        ("hello how are you", "Hello, how are you?"),
        ("مرحبا كيف حالك", "مرحبا، كيف حالك؟"),
        ("आप कैसे हैं", "आप कैसे हैं।"),
        ("你好 我很好", "你好，我很好。"),
    ] {
        let setup = Setup { stt: true, preemptive: true, turn: TurnMode::Stt, ..Setup::default() };
        let run = simulate(async {
            let mut run = Run::start(&setup, 0, Default::default()).await;
            for input in [
                Input::SttStart { stt: Ref::new(0) },
                Input::SttPreflight {
                    utterance: Ref::new(0),
                    text: UserText(preflight.into()),
                    speaker: None,
                },
                Input::Advance { ms: Millis(100) },
                Input::SttFinal {
                    utterance: Ref::new(0),
                    text: UserText(final_text.into()),
                    speaker: None,
                },
                Input::SttEnd { utterance: Ref::new(0) },
                Input::Advance { ms: Millis(1_000) },
            ] {
                run.apply(&input).await;
            }
            run
        });
        assert_eq!(run.record.resources(Kind::Chat).count(), 1, "{preflight:?} -> {final_text:?}");
        assert_eq!(said(&run, Role::User), [final_text]);
    }
}

/// A final that changes the preflight's words retires the preflight reply, and only the
/// corrected turn is answered. https://github.com/livekit/agents/issues/1303 https://github.com/livekit/agents/pull/5806
#[test]
fn corrected_final_retires_the_stale_reply() {
    let setup =
        Setup { stt: true, preemptive: true, turn: TurnMode::Stt, tts: false, ..Setup::default() };
    let run = simulate(async {
        let mut run = Run::start(&setup, 0, Default::default()).await;
        for input in [
            Input::SttStart { stt: Ref::new(0) },
            Input::SttPreflight {
                utterance: Ref::new(0),
                text: UserText("book Tuesday".into()),
                speaker: None,
            },
            Input::Advance { ms: Millis(100) },
            Input::LlmText { chat: Ref::new(0), text: AgentText("Tuesday booked.".into()) },
            Input::LlmEnd { chat: Ref::new(0) },
            Input::Advance { ms: Millis(100) },
            Input::SttFinal {
                utterance: Ref::new(0),
                text: UserText("book Thursday".into()),
                speaker: None,
            },
            Input::SttEnd { utterance: Ref::new(0) },
            Input::Advance { ms: Millis(1_000) },
            Input::LlmText { chat: Ref::new(1), text: AgentText("Thursday booked.".into()) },
            Input::LlmEnd { chat: Ref::new(1) },
            Input::Advance { ms: Millis(500) },
        ] {
            run.apply(&input).await;
        }
        run
    });

    assert_eq!(said(&run, Role::User), ["book Thursday"]);
    assert_eq!(said(&run, Role::Assistant), ["Thursday booked."]);
}

/// A failed end-of-turn prediction still commits and answers the turn.
/// https://github.com/livekit/agents/issues/6177 https://github.com/livekit/agents/pull/8171
#[test]
fn a_failed_eot_prediction_still_commits_the_turn() {
    let setup = Setup { vad: true, stt: true, eot: true, ..Setup::default() };
    let run = simulate(async {
        let mut run = Run::start(&setup, 0, Default::default()).await;
        for input in [
            Input::Configure {
                knob: Knob::MinEndpointing,
                mode: TurnMode::Default,
                ms: Millis(200),
                on: true,
                words: Count(0),
            },
            Input::Configure {
                knob: Knob::MaxEndpointing,
                mode: TurnMode::Default,
                ms: Millis(800),
                on: true,
                words: Count(0),
            },
            Input::VadStart { vad: Ref::new(0), speech_ms: Millis(300) },
            Input::SttStart { stt: Ref::new(0) },
            Input::Advance { ms: Millis(500) },
            Input::SttFinal {
                utterance: Ref::new(0),
                text: UserText("are you open today".into()),
                speaker: None,
            },
            Input::VadInference {
                vad: Ref::new(0),
                probability: Confidence(0),
                speaking: false,
                duration: Millis(224),
            },
            Input::Advance { ms: Millis(10) },
            Input::EotFail { prediction: Ref::new(0) },
            Input::VadEnd { vad: Ref::new(0), silence_ms: Millis(250) },
            Input::SttEnd { utterance: Ref::new(0) },
            Input::Advance { ms: Millis(1_000) },
        ] {
            run.apply(&input).await;
        }
        run
    });

    assert_eq!(run.record.resources(Kind::Prediction).count(), 1, "the prediction ran and failed");
    assert_eq!(said(&run, Role::User), ["are you open today"]);
    assert_eq!(run.record.resources(Kind::Chat).count(), 1, "the turn is answered");
}

/// Resumed speech restarts the endpointing delay, so the turn commits once, after the last
/// pause. https://github.com/livekit/agents/issues/4325 https://github.com/livekit/agents/pull/6820
#[test]
fn resumed_speech_restarts_the_endpointing_delay() {
    let setup = Setup { vad: true, stt: true, ..Setup::default() };
    let run = simulate(async {
        let mut run = Run::start(&setup, 0, Default::default()).await;
        for input in [
            Input::Configure {
                knob: Knob::MinEndpointing,
                mode: TurnMode::Default,
                ms: Millis(500),
                on: true,
                words: Count(0),
            },
            Input::VadStart { vad: Ref::new(0), speech_ms: Millis(300) },
            Input::SttStart { stt: Ref::new(0) },
            Input::SttFinal {
                utterance: Ref::new(0),
                text: UserText("I need".into()),
                speaker: None,
            },
            Input::VadEnd { vad: Ref::new(0), silence_ms: Millis(0) },
            Input::Advance { ms: Millis(300) },
            Input::VadStart { vad: Ref::new(0), speech_ms: Millis(100) },
            Input::Advance { ms: Millis(300) },
            Input::SttFinal {
                utterance: Ref::new(0),
                text: UserText("a table".into()),
                speaker: None,
            },
            Input::VadEnd { vad: Ref::new(0), silence_ms: Millis(0) },
            Input::SttEnd { utterance: Ref::new(0) },
            Input::Advance { ms: Millis(600) },
        ] {
            run.apply(&input).await;
        }
        run
    });

    let committed: Vec<_> = run
        .record
        .host_view()
        .events()
        .into_iter()
        .filter_map(|(at, event)| match event {
            SessionEvent::ConversationItemAdded { item, .. } => item
                .as_message()
                .filter(|m| m.role == Role::User)
                .map(|m| (at.at_ms, m.raw_text_content())),
            _ => None,
        })
        .collect();
    let [(at_ms, text)] = committed.as_slice() else { panic!("one user turn: {committed:?}") };
    assert_eq!(text.as_deref(), Some("I need a table"));
    assert!(*at_ms >= 1_100, "committed on the first pause's deadline, at {at_ms} ms");
}

/// A draining close keeps the final transcript that arrives after it.
/// https://github.com/livekit/agents/issues/6889 https://github.com/livekit/agents/pull/3377
#[test]
fn draining_close_keeps_the_trailing_final_transcript() {
    let setup = Setup { stt: true, vad: true, ..Setup::default() };
    let run = simulate(async {
        let mut run = Run::start(&setup, 0, Default::default()).await;
        for input in [
            Input::VadStart { vad: Ref::new(0), speech_ms: Millis(300) },
            Input::SttStart { stt: Ref::new(0) },
            Input::Advance { ms: Millis(500) },
            Input::VadEnd { vad: Ref::new(0), silence_ms: Millis(250) },
            Input::Close { drain: true },
            Input::Advance { ms: Millis(100) },
            Input::SttFinal {
                utterance: Ref::new(0),
                text: UserText("please cancel the booking".into()),
                speaker: None,
            },
            Input::SttEnd { utterance: Ref::new(0) },
            Input::Advance { ms: Millis(5_000) },
        ] {
            run.apply(&input).await;
        }
        run
    });

    assert_eq!(said(&run, Role::User), ["please cancel the booking"]);
}
