//! Completion (spec §12.3).

use super::{Call, listening};
use lk_agents_fuzz::drive::Setup;
use lk_agents_fuzz::fakes::amd::STAGE;
use lk_agents_fuzz::input::{AmdVerdict, Count, Input, Knob, Millis, TurnMode};

const LINE: &str = "Hello, this is the clinic.";
/// About six seconds of speech: still playing when the box hangs up.
const MESSAGE: &str = "Hello, this is the clinic calling about your appointment on Tuesday at \
nine. Please call us back at five five five, one two three four, at your convenience. Thank you.";

fn hooked() -> Setup {
    Setup { hook_bound_ms: Some(30_000), ..listening() }
}

/// Completion into a stage the application handles while a reply is generating
/// but not yet playing cancels it. Python: `test_cancel_pending_speech_includes_preemptive_and_queued_speech`.
#[test]
fn completion_into_a_handled_stage_cancels_a_generating_reply() {
    let call = Call::run(listening(), async |call| {
        call.attach(None).await;
        call.says("hello").await;
        call.decide(AmdVerdict::Allow).await;
        call.advance(50).await;
        assert!(!call.asked("hello").is_empty(), "the reply is generating");
        call.detach(AmdVerdict::Allow, true).await;
        call.model_answers(LINE).await;
        call.advance(3_000).await;
    });
    assert_eq!(call.heard_reply("hello"), None, "an unplayed reply is cancelled");
    assert!(call.failures().is_empty(), "{:?}", call.failures());
}

/// Completion after a decision is stored but before it is applied replaces it
/// with the "AMD completed" row: the reply loses its stage instructions.
#[test]
fn completion_replaces_a_stored_decision() {
    let call = Call::run(hooked(), async |call| {
        call.attach(None).await;
        call.says("you have reached the screener").await;
        call.decide(AmdVerdict::Stage).await;
        call.detach(AmdVerdict::Allow, false).await;
        call.hook_returns().await;
        call.advance(50).await;
        call.model_answers(LINE).await;
        call.advance(3_000).await;
    });
    let heard = call.heard_requests("you have reached the screener");
    assert_eq!(heard.len(), 1);
    assert!(!heard[0].carries(STAGE), "the stored stage decision was replaced");
    assert!(call.failures().is_empty(), "{:?}", call.failures());
}

/// A voicemail box hanging up while the message plays: the message keeps
/// playing (`_cancel_pending_speeches` keeps a playing speech). Python:
/// `test_late_unavailable_does_not_cancel_reply`.
#[test]
fn a_voicemail_message_keeps_playing_when_the_box_hangs_up() {
    let call = Call::run(listening(), async |call| {
        call.attach(None).await;
        call.says("please leave a message after the tone").await;
        call.decide(AmdVerdict::Silence).await;
        call.advance(50).await;
        call.model_answers(MESSAGE).await;
        call.advance(2_500).await;
        assert!(call.heard_reply("please leave a message after the tone").is_some(), "playing");
        call.detach(AmdVerdict::Allow, true).await;
        call.advance(15_000).await;
    });
    let heard = call.heard_requests("please leave a message after the tone");
    let speech = heard[0].speech.unwrap();
    let view = call.run.record.host_view().speech(speech);
    assert!(view.is_done() && !view.is_interrupted(), "the message played to its end");
    assert!(call.failures().is_empty(), "{:?}", call.failures());
}

/// A `wait` turn in its hook at completion, and at `aclose()`. Python:
/// `test_wait_does_not_block_completion_or_allow_a_reply_afterward`.
#[test]
fn a_wait_turn_in_its_hook_is_never_answered() {
    for close in [false, true] {
        let call = Call::run(hooked(), async |call| {
            call.attach(None).await;
            call.says("please hold").await;
            call.decide(AmdVerdict::Wait).await;
            if close {
                call.send(Input::Close { drain: false }).await;
            } else {
                call.detach(AmdVerdict::Wait, false).await;
            }
            call.hook_returns().await;
            call.advance(50).await;
            call.model_answers(LINE).await;
            call.advance(3_000).await;
            if close {
                // The close waits on the hosted `on_exit` it issued until the hook bound, as
                // without AMD; past it, the close finishes and the streams it owes are closed.
                call.advance(30_000).await;
            }
        });
        assert!(call.unanswered("please hold"), "close={close}");
        assert!(call.saved_user_words().contains(&"please hold".to_string()), "close={close}");
        assert!(call.failures().is_empty(), "close={close}: {:?}", call.failures());
    }
}

/// A `machine-ivr` stage followed by a direct `human` completion, with an agent
/// that sets `allow_interruptions=False`. AMD forces interruptions on while attached and
/// restores them at detach; the IVR reply offers the DTMF tool, the human's reply does not.
#[test]
fn ivr_then_human_with_an_uninterruptible_agent() {
    let call = Call::run(listening(), async |call| {
        call.send(Input::Configure {
            knob: Knob::Interruptions,
            mode: TurnMode::Default,
            ms: Millis(0),
            on: false,
            words: Count(0),
        })
        .await;
        call.attach(None).await;
        call.says("press one for appointments").await;
        call.decide(AmdVerdict::Ivr).await;
        call.advance(50).await;
        call.model_answers(LINE).await;
        call.advance(4_000).await;
        call.says("hello this is Sam").await;
        call.detach(AmdVerdict::Allow, false).await;
        call.advance(50).await;
        call.model_answers(LINE).await;
        call.advance(3_000).await;
    });
    let interruptible = |words: &str| {
        let speech = call.heard_requests(words).first()?.speech?;
        call.run.record.handles().get(&speech).map(|h| h.allow_interruptions)
    };
    let ivr = call.made("press one for appointments");
    assert_eq!(ivr[0].offers(lk_agents_fuzz::fakes::amd::SEND_DTMF_TOOL), 1);
    assert!(ivr[0].carries(STAGE));
    assert_eq!(interruptible("press one for appointments"), Some(true), "forced on");
    let human = call.heard_requests("hello this is Sam");
    assert_eq!(human.len(), 1);
    assert_eq!(human[0].offers(lk_agents_fuzz::fakes::amd::SEND_DTMF_TOOL), 0);
    assert!(!human[0].carries(STAGE), "no human instructions after a plain completion");
    assert_eq!(interruptible("hello this is Sam"), Some(false), "restored at detach");
    assert!(call.failures().is_empty(), "{:?}", call.failures());
}

/// A draining close ends the run as its completion would: a greeting queued before AMD, behind
/// one already playing, and held at the head plays during the drain (spec §12.3), and the run
/// hears `Detached`. Python: `test_a_greeting_queued_before_amd_is_in_the_history_when_it_plays`,
/// closed instead of completed.
#[test]
fn a_draining_close_plays_the_held_greeting() {
    let say = |text: &str| Input::Say {
        text: lk_agents_fuzz::input::AgentText(text.into()),
        input: lk_agents_fuzz::input::SayInput::Words,
        allow: None,
        to_ctx: true,
    };
    let mut closed = 0;
    let call = Call::run(listening(), async |call| {
        call.send(say("Hi.")).await;
        call.send(say(LINE)).await;
        call.attach(None).await;
        call.advance(3_000).await;
        closed = call.now_ms();
        call.send(Input::Close { drain: true }).await;
        call.advance(3_000).await;
    });
    let said: Vec<u64> = call.run.record.host_view().speeches().into_iter().map(|s| s.id).collect();
    let [_, greeting] = said[..] else { panic!("two greetings: {said:?}") };
    let heard = call.heard(Some(greeting)).expect("the drain plays the held greeting");
    assert!(heard >= closed, "held until the close, heard at {heard} ms");
    assert!(call.detached_at().is_some(), "the run hears its gate released");
    assert!(call.failures().is_empty(), "{:?}", call.failures());
}
