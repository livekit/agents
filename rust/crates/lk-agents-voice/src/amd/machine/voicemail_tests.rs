//! The voicemail record (spec §10.4): one message at most, retries after a failed attempt, and
//! what a played message changes. Python's tests are in `tests/test_amd.py`.

use super::harness::*;
use super::*;
use crate::agent_session::machine::activity::scheduling::SpeechKind;
use crate::amd::{AmdObservation, DetachCancel, SpeechObservation, TurnEnd};

const MESSAGE: &str = "Hi Sam, this is Alex from Acme. Please call me back.";
const GREETING: &str = "Please leave a message after the tone.";

fn voicemail() -> crate::amd::ReplyDecision {
    staged(DEFAULT_VOICEMAIL_INSTRUCTIONS)
}

fn played_ivr() -> crate::amd::ReplyDecision {
    ivr(&format!("{DEFAULT_IVR_INSTRUCTIONS}{VOICEMAIL_PLAYED_NOTE}"))
}

/// A run whose voicemail message played on turn 1.
fn after_message(options: AmdOptions) -> H {
    let mut h = H::started(options);
    let turn = h.turn(GREETING, Category::MachineVm);
    assert_eq!(h.decision(turn), voicemail());
    h.reply_after_silence(turn, MESSAGE);
    h
}

#[test]
fn test_voicemail_sends_one_delivered_message_and_records_playback() {
    let mut h = after_message(AmdOptions::default());
    let again = h.turn(GREETING, Category::MachineVm);
    assert_eq!(h.decision(again), deny());
    assert!(h.detaches().is_empty());
    h.step(In::Close);
    assert!(h.outcome().voicemail_message_played);
}

#[test]
fn test_superseded_voicemail_reply_does_not_consume_the_stage_message() {
    let mut h = H::started(AmdOptions::default());
    let first = h.turn(GREETING, Category::MachineVm);
    let speech = h.reply(first);
    h.silence_held(first);
    // a newer turn cancels the held reply
    let second = h.commit("Please leave your message.");
    h.ended(first, TurnEnd::Superseded);
    h.speech(speech, SpeechObservation::Done { interrupted: true, failed: false });

    h.classify(second, Category::MachineVm);
    assert_eq!(h.decision(second), voicemail());
    h.reply_after_silence(second, MESSAGE);

    let third = h.turn(GREETING, Category::MachineVm);
    assert_eq!(h.decision(third), deny());
    h.step(In::Close);
    assert!(h.outcome().voicemail_message_played);
}

#[test]
fn test_stale_voicemail_handle_does_not_clear_a_newer_reservation() {
    // the reservation belongs to the turn allowed the message; a newer commit releases it
    for stale_first in [true, false] {
        let mut h = H::started(AmdOptions::default());
        let first = h.turn(GREETING, Category::MachineVm);
        assert_eq!(h.decision(first), voicemail());
        let second = h.turn(GREETING, Category::MachineVm);
        assert_eq!(h.decision(second), voicemail(), "the first turn made no attempt");

        let old = h.created(SpeechKind::TurnReply);
        let new = h.created(SpeechKind::TurnReply);
        if stale_first {
            h.observe(AmdObservation::ReplyCreated { turn: first, speech: old });
        }
        h.observe(AmdObservation::ReplyCreated { turn: second, speech: new });
        if !stale_first {
            h.observe(AmdObservation::ReplyCreated { turn: first, speech: old });
        }
        h.play(new, MESSAGE);
        h.speech(old, SpeechObservation::Done { interrupted: true, failed: false });

        let third = h.turn(GREETING, Category::MachineVm);
        assert_eq!(h.decision(third), deny(), "stale first: {stale_first}");
        h.step(In::Close);
        assert!(h.outcome().voicemail_message_played, "stale first: {stale_first}");
    }
}

#[test]
fn test_voicemail_playback_is_recorded_by_its_handle() {
    // played only if the speech finished, not interrupted or failed, no item was cut (a TTS
    // failure partway), playout advanced, and the run was still active
    for outcome in ["played", "interrupted", "failed", "truncated", "no_audio", "closed"] {
        let mut h = H::started(AmdOptions::default());
        let turn = h.turn(GREETING, Category::MachineVm);
        let speech = h.reply(turn);
        h.silence_held(turn);
        h.silence_released(turn);
        if outcome != "no_audio" {
            h.speech(speech, SpeechObservation::PlayoutAdvanced);
        }
        let text = Some(if outcome == "truncated" { "Hi Sam, your" } else { MESSAGE }.to_owned());
        let interrupted = outcome == "truncated";
        h.speech(speech, SpeechObservation::ItemAdded { text, interrupted });
        if outcome == "closed" {
            h.step(In::Close);
        }
        let done = SpeechObservation::Done {
            interrupted: outcome == "interrupted",
            failed: outcome == "failed",
        };
        h.speech(speech, done);

        let played = outcome == "played";
        if outcome != "closed" {
            let next = h.turn(GREETING, Category::MachineVm);
            let expected = if played { deny() } else { voicemail() };
            assert_eq!(h.decision(next), expected, "{outcome}: a failed attempt can be retried");
            h.step(In::Close);
        }
        assert_eq!(h.outcome().voicemail_message_played, played, "{outcome}");
    }
}

#[test]
fn test_voicemail_delivery_guard_survives_transitions_and_allows_retry() {
    for bridge in
        [None, Some(Category::Uncertain), Some(Category::Wait), Some(Category::MachineIvr)]
    {
        for outcome in ["played", "interrupted", "failed", "no_audio"] {
            let case = format!("{bridge:?}, {outcome}");
            let mut h = H::started(AmdOptions::default());
            let first = h.turn(GREETING, Category::MachineVm);
            assert_eq!(h.decision(first), voicemail(), "{case}");
            let speech = h.reply(first);
            if let Some(bridge) = bridge {
                h.turn("One moment.", bridge);
            }
            // an attempt that has not finished blocks another
            let overlapping = h.turn(GREETING, Category::MachineVm);
            assert_eq!(h.decision(overlapping), deny(), "{case}");

            if outcome != "no_audio" {
                h.speech(speech, SpeechObservation::PlayoutAdvanced);
            }
            h.speech(
                speech,
                SpeechObservation::ItemAdded { text: Some(MESSAGE.to_owned()), interrupted: false },
            );
            let done = SpeechObservation::Done {
                interrupted: outcome == "interrupted",
                failed: outcome == "failed",
            };
            h.speech(speech, done);

            if let Some(bridge) = bridge {
                h.turn("One moment.", bridge);
            }
            // the record survives other stages; only a played message ends the retries
            let retry = h.turn(GREETING, Category::MachineVm);
            let expected = if outcome == "played" { deny() } else { voicemail() };
            assert_eq!(h.decision(retry), expected, "{case}");
        }
    }
}

#[test]
fn test_a_corrected_voicemail_reply_moves_tracking_to_the_replacement() {
    // a correction remakes the unheard reply; the first speech's interrupted end is ignored
    let mut h = H::started(AmdOptions::default());
    let turn = h.turn(GREETING, Category::MachineVm);
    let first = h.reply(turn);
    let replacement = h.reply(turn);
    h.speech(first, SpeechObservation::Done { interrupted: true, failed: false });

    let overlapping = h.turn(GREETING, Category::MachineVm);
    assert_eq!(h.decision(overlapping), deny(), "the replacement is still in flight");
    h.play(replacement, MESSAGE);

    let next = h.turn(GREETING, Category::MachineVm);
    assert_eq!(h.decision(next), deny());
    h.step(In::Close);
    assert!(h.outcome().voicemail_message_played);
}

#[test]
fn test_a_goodbye_after_the_voicemail_message_keeps_the_stage() {
    // the box's goodbye is not an unreachable call; a person who picks up is human (spec §12.2)
    let cases = [
        (false, Category::MachineUnavailable, Stage::MachineVm),
        (true, Category::MachineUnavailable, Stage::MachineIvr),
        (true, Category::Human, Stage::Human),
    ];
    for (menu, last, expected) in cases {
        let mut h = after_message(AmdOptions::default());
        if menu {
            let turn = h.turn("To replay your message, press 1.", Category::MachineIvr);
            // after a played message, the IVR instructions say so (spec §10.2)
            assert_eq!(h.decision(turn), played_ivr());
        }
        h.turn("Goodbye.", last);
        if menu && last == Category::Human {
            h.turn("Hello?", Category::Human);
        }
        let outcome = h.outcome();
        assert_eq!(outcome.reason, AmdReason::Finished);
        assert_eq!(outcome.category, expected, "menu {menu}, {last:?}");
        assert!(outcome.voicemail_message_played);
    }
}

#[test]
fn test_late_post_voicemail_menu_uses_the_normal_ivr_idle_timeout() {
    let options = AmdOptions {
        idle_timeout: secs(3),
        voicemail_idle_timeout: secs(10),
        ..AmdOptions::default()
    };
    let mut h = after_message(options);
    // the voicemail idle timer restarts after playback
    assert_eq!(h.armed(AmdTimer::Idle), Some(h.now.saturating_add(secs(10))));
    h.advance(secs(5));
    assert!(h.detaches().is_empty());

    h.speech_started(h.now);
    assert_eq!(h.armed(AmdTimer::Idle), None, "participant speech keeps the run busy");
    h.advance(secs(1));
    h.speech_ended(h.now);
    let menu = "To replay your message, press 1. To send this message now, press pound.";
    let turn = h.turn(menu, Category::MachineIvr);
    assert_eq!(h.decision(turn), played_ivr());
    h.reply_after_silence(turn, "Pound.");
    assert_eq!(h.armed(AmdTimer::Idle), Some(h.now.saturating_add(secs(3))));

    h.advance(secs(3));
    let outcome = h.outcome();
    assert_eq!(outcome.reason, AmdReason::IdleTimeout);
    assert_eq!(outcome.category, Stage::MachineIvr);
    assert!(outcome.voicemail_message_played);
}

#[test]
fn test_unchanged_voicemail_rearms_idle_without_another_reply() {
    for reason in [AmdReason::InferenceError, AmdReason::Reused, AmdReason::Prediction] {
        let mut h =
            after_message(AmdOptions { voicemail_idle_timeout: secs(3), ..AmdOptions::default() });
        h.speech_started(h.now);
        h.speech_ended(h.now);
        let turn = match reason {
            AmdReason::Reused => h.commit(""),
            AmdReason::InferenceError => {
                let turn = h.commit("Beep.");
                h.respond(h.request_for(turn).req, Err(ClassifyError::InvalidOutput));
                turn
            }
            _ => h.turn(GREETING, Category::MachineVm),
        };
        assert_eq!(h.decision(turn), deny(), "{reason:?}");
        h.advance(secs(3));
        let outcome = h.outcome();
        assert_eq!(outcome.reason, AmdReason::IdleTimeout, "{reason:?}");
        assert_eq!(outcome.category, Stage::MachineVm, "{reason:?}");
    }
}

#[test]
fn test_voicemail_idle_does_not_extend_the_hard_timeout() {
    let options =
        AmdOptions { timeout: secs(3), voicemail_idle_timeout: secs(10), ..AmdOptions::default() };
    let mut h = H::started(options);
    let turn = h.turn(GREETING, Category::MachineVm);
    h.silence_held(turn);
    h.advance(secs(3));
    // completion releases the held reply as an "AMD completed" allow (spec §10.3, §12.3)
    assert_eq!(h.detach(), completed_with(plain(), DetachCancel::Keep));
    let outcome = h.outcome();
    assert_eq!((outcome.reason, outcome.category), (AmdReason::Timeout, Stage::MachineVm));
}

#[test]
fn test_correction_preserves_delivered_voicemail_and_dtmf_history() {
    let mut h = after_message(AmdOptions::default());
    let sent = h.dtmf("1");
    h.executed(std::slice::from_ref(&sent));
    let turn = h.commit("Please state your name and why you are calling.");
    assert_eq!(dtmf_calls(&h.request_for(turn)), vec![sent.0]);
    h.classify(turn, Category::MachineScreening);
    assert_eq!(h.decision(turn), staged(DEFAULT_SCREENING_INSTRUCTIONS));

    // a correction does not reset a played message (spec §9.3)
    let back = h.turn(GREETING, Category::MachineVm);
    assert_eq!(h.decision(back), deny());
    assert_eq!(h.last_prediction().stage, Stage::MachineVm);
}

#[test]
fn test_the_opening_is_not_a_voicemail_message() {
    // spec §10.6: the opening never sets `voicemail_message_played`
    let options = AmdOptions { opening_text: Some("Hi!".to_owned()), ..AmdOptions::default() };
    let mut h = H::started(options);
    h.say("Hi!");
    let turn = h.turn(GREETING, Category::MachineVm);
    assert_eq!(h.decision(turn), voicemail());
    h.step(In::Close);
    assert!(!h.outcome().voicemail_message_played);
}
