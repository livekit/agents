//! Activation and the opening (proposal "Activation and the opening"; spec §10.6; invariant 15).

use super::{Call, listening};
use lk_agents_fuzz::fakes::amd::{OPENING_INSTRUCTIONS, OPENING_TEXT};
use lk_agents_fuzz::input::{AmdOpening, AmdVerdict, Input, Millis, Ref};
use lk_agents_voice::agent_session::machine::activity::scheduling::SpeechKind;

const LINE: &str = "Hello, this is the clinic.";

/// The speeches of this kind the session created.
fn speeches(call: &Call, kind: SpeechKind) -> Vec<u64> {
    call.run
        .record
        .host_view()
        .speeches()
        .into_iter()
        .filter(|s| s.kind() == Some(kind))
        .map(|s| s.id)
        .collect()
}

/// The opening plays while every turn stays held. Python:
/// `test_opening_text_plays_while_other_replies_stay_held`.
#[test]
fn the_opening_plays_while_turns_stay_held() {
    let call = Call::run(listening(), async |call| {
        call.attach(Some(AmdOpening::Text)).await;
        call.advance(3_000).await;
        call.says("hello").await;
        call.advance(2_000).await;
        assert!(call.unanswered("hello"), "a turn during the opening waits");
        call.decide(AmdVerdict::Allow).await;
        call.advance(50).await;
        call.model_answers(LINE).await;
        call.advance(3_000).await;
    });
    let said = speeches(&call, SpeechKind::Say);
    assert_eq!(said.len(), 1, "the opening is said");
    assert!(call.heard(Some(said[0])).is_some(), "and heard");
    assert!(call.recorded(lk_agents_abi::Role::Assistant).iter().any(|t| t == OPENING_TEXT));
    assert_eq!(call.heard_requests("hello").len(), 1);
    assert!(call.failures().is_empty(), "{:?}", call.failures());
}

/// Opening instructions generate the opening. Python:
/// `test_opening_instructions_generate_a_reply_without_tools`.
#[test]
fn opening_instructions_generate_the_opening() {
    let call = Call::run(listening(), async |call| {
        call.attach(Some(AmdOpening::Instructions)).await;
        call.advance(50).await;
        call.model_answers(LINE).await;
        call.advance(3_000).await;
    });
    let asked: Vec<_> = call
        .run
        .record
        .amd_view()
        .requests()
        .into_iter()
        .filter(|q| q.carries(OPENING_INSTRUCTIONS))
        .collect();
    assert_eq!(asked.len(), 1, "the opening is generated once");
    assert!(call.heard(asked[0].speech).is_some());
    assert!(call.failures().is_empty(), "{:?}", call.failures());
}

/// Proposal story: participant speech starting right after `Activated`, before the opening's
/// first playout, cancels it; speech already under way when AMD attached drops it (audio is
/// discarded while AMD is pending, spec §5.2, so that is the only speech at activation). Python:
/// `test_participant_speech_before_the_opening_plays_cancels_it`.
#[test]
fn speech_before_the_opening_plays_cancels_it() {
    for already_speaking in [false, true] {
        let call = Call::run(listening(), async |call| {
            if already_speaking {
                call.send(Input::VadStart { vad: Ref::new(0), speech_ms: Millis(0) }).await;
            }
            call.send(Input::AmdAttach {}).await;
            call.send(Input::AmdActivate { opening: Some(AmdOpening::Text) }).await;
            if !already_speaking {
                call.send(Input::VadStart { vad: Ref::new(0), speech_ms: Millis(0) }).await;
            }
            call.advance(500).await;
            call.send(Input::VadEnd { vad: Ref::new(0), silence_ms: Millis(0) }).await;
            call.advance(3_000).await;
        });
        let said = speeches(&call, SpeechKind::Say);
        assert!(
            said.iter().all(|s| call.heard(Some(*s)).is_none()),
            "already_speaking={already_speaking}: the opening was heard"
        );
        // Not vacuous: the gate was attached and AMD heard the participant.
        assert!(
            call.observed()
                .iter()
                .any(|(_, o)| matches!(o, lk_agents_voice::amd::AmdObservation::UserSpeech(_))),
            "already_speaking={already_speaking}: AMD heard the participant"
        );
        if !already_speaking {
            assert_eq!(said.len(), 1, "the opening was created at activation, then cancelled");
        }
        assert!(
            call.failures().is_empty(),
            "already_speaking={already_speaking}: {:?}",
            call.failures()
        );
    }
}

/// `say()` and `generate_reply()` from the host are refused while AMD is attached (proposal
/// "Session rules while attached").
#[test]
fn host_speech_is_refused_while_attached() {
    let call = Call::run(listening(), async |call| {
        call.attach(None).await;
        call.send(Input::Say {
            text: lk_agents_fuzz::input::AgentText("Hello there.".into()),
            input: lk_agents_fuzz::input::SayInput::Words,
            allow: None,
            to_ctx: true,
        })
        .await;
        call.advance(2_000).await;
    });
    assert!(speeches(&call, SpeechKind::Say).iter().all(|s| call.heard(Some(*s)).is_none()));
    assert!(!call.recorded(lk_agents_abi::Role::Assistant).iter().any(|t| t == "Hello there."));
    assert!(
        call.observed()
            .iter()
            .any(|(_, o)| matches!(o, lk_agents_voice::amd::AmdObservation::AgentBusy { .. }))
    );
}
