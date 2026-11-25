//! Turns, requests and classifier history: who owns the request, the late-final wait and the
//! right to reply (spec §6, §7, §8). Python's tests are in `tests/test_amd.py`.

use super::harness::*;
use super::*;
use crate::agent_session::machine::activity::scheduling::SpeechKind;
use crate::amd::{AmdObservation, DetachCancel, SpeechObservation};
use serde_json::json;

fn amd_stt() -> AmdSetup {
    AmdSetup { amd_stt: true, ..setup() }
}

/// AMD's STT without session STT.
fn amd_stt_only() -> AmdSetup {
    AmdSetup { amd_stt: true, session_stt: false, ..setup() }
}

/// A user message with the session transcript alone, as the default setup has.
fn user(turn_id: u64, text: &str) -> ClassifierItem {
    let transcript = TurnTranscript { session: Some(text.to_owned()), amd: None };
    ClassifierItem::User { turn_id, transcript }
}

fn both(session: &str, amd: &str) -> serde_json::Value {
    json!({ "transcript_1": session, "transcript_2": amd })
}

/// A user message in the dual-transcript format, decoded.
fn decoded(text: &str) -> serde_json::Value {
    serde_json::from_str(text).unwrap_or_else(|_| panic!("not the dual-transcript JSON: {text}"))
}

fn current_text(request: &ClassifyRequest) -> String {
    match request.history.last() {
        Some(ClassifierItem::User { transcript, .. }) => transcript.for_classifier(),
        other => panic!("the request ends with {other:?}, not its turn"),
    }
}

// Requests (spec §7.1, §8.1).

#[test]
fn test_a_turn_with_text_starts_one_request() {
    let mut h = H::started(AmdOptions::default());
    h.advance(ms(500));
    let at = h.now;
    let turn = h.commit("hello");

    assert_eq!(h.requests_for(turn), 1);
    let request = h.request_for(turn);
    assert_eq!(request.turn_id, 1);
    assert_eq!(request.stage, Stage::Uncertain);
    assert_eq!(request.previous_prediction, None);
    assert!(!request.likely_human);
    assert_eq!(listed(&request), spec_recommended(Stage::Uncertain));
    assert_eq!(request.history, vec![user(1, "hello")]);
    assert_eq!(h.armed(AmdTimer::Inference(request.req)), Some(at.saturating_add(secs(2))));
    assert!(h.predictions().is_empty());
    assert!(h.decisions(turn).is_empty(), "no decision before the turn's prediction");
}

#[test]
fn test_turn_ids_are_local_to_each_amd_run() {
    // AMD numbers its turns from 1 in each run; answers name the session's turn
    for first in [41, 42] {
        let mut h = H::started(AmdOptions::default());
        h.turns_from(first);
        let turn = h.commit("hello");
        assert_eq!(turn, TurnId(first));
        assert_eq!(h.request_for(turn).turn_id, 1);
        h.classify(turn, Category::MachineScreening);
        assert_eq!(h.last_prediction().turn_id, 1);
        assert_eq!(h.decision(turn), staged(DEFAULT_SCREENING_INSTRUCTIONS));
    }
}

#[test]
fn test_request_state_json_carries_the_stage_previous_prediction_and_speech() {
    // `_inference.classify` (Python `_inference.py:86`): the request's system message
    let mut h = H::started(AmdOptions::default());
    let first = h.commit("hello");
    assert_eq!(
        state(&h.request_for(first)),
        json!({
            "stage": "uncertain",
            "previous_prediction": null,
            // `sorted(_fsm.RECOMMENDED[stage])`: by wire name
            "recommended_next_categories": [
                "human", "machine-ivr", "machine-screening", "machine-unavailable", "machine-vm",
                "uncertain", "wait",
            ],
            "speech_duration": 0.0,
        })
    );
    h.classify(first, Category::MachineVm);

    h.advance(secs(1));
    h.speech_started(h.now);
    h.advance(secs(2));
    h.speech_ended(h.now);
    let second = h.commit("Leave a message after the tone.");
    assert_eq!(
        state(&h.request_for(second)),
        json!({
            "stage": "machine-vm",
            "previous_prediction": { "turn_id": 1, "category": "machine-vm", "reason": "prediction" },
            "recommended_next_categories": [
                "human", "machine-ivr", "machine-unavailable", "machine-vm", "uncertain", "wait",
            ],
            "speech_duration": 2.0,
        })
    );
}

#[test]
fn test_superseded_inference_keeps_the_classification_context() {
    let mut h = H::started(AmdOptions::default());
    let sent = h.dtmf("1");
    h.executed(std::slice::from_ref(&sent));
    let first = h.commit("Hello, can you hear me?");
    let previous = h.request_for(first);
    let second = h.commit("Yes, let's schedule that.");
    let request = h.request_for(second);

    // a newer turn with text cancels the pending request (spec §7.4)
    assert_eq!(h.cancelled(), vec![previous.req]);
    assert_eq!(
        request.history,
        vec![
            ClassifierItem::DtmfCall(sent.0),
            ClassifierItem::DtmfOutput(sent.1),
            user(1, "Hello, can you hear me?"),
            user(2, "Yes, let's schedule that."),
        ]
    );
    assert_eq!((previous.stage, request.stage), (Stage::Uncertain, Stage::Uncertain));

    // a stale request never decides (spec §14 rule 2)
    let human = Classification { category: Category::Human, inference_duration: ms(10) };
    h.respond(previous.req, Ok(human));
    assert!(h.predictions().is_empty());
    assert!(h.detaches().is_empty());

    h.classify(second, Category::Human);
    assert!(!h.allowed(first));
    let outcome = h.outcome();
    let last = outcome.last_prediction.expect("a last prediction");
    assert_eq!((last.turn_id, last.transcript.as_str()), (2, "Yes, let's schedule that."));
    assert_eq!(h.reasons(), vec![(2, AmdReason::Prediction)]);
}

// Empty turns (spec §7.3).

#[test]
fn test_empty_turns_do_not_cancel_or_count_against_pending_inference() {
    for reason in [AmdReason::Prediction, AmdReason::InferenceError, AmdReason::InferenceTimeout] {
        let timeout = reason == AmdReason::InferenceTimeout;
        let options = AmdOptions {
            inference_timeout: if timeout { ms(100) } else { secs(10) },
            max_uncertain_turns: 2,
            ..AmdOptions::default()
        };
        let mut h = H::started(options);
        let first = h.commit("hello");
        let req = h.request_for(first).req;
        let second = h.commit("");
        let current = h.commit("");
        assert!(h.cancelled().is_empty(), "{reason:?}: an empty turn cancels nothing");
        assert_eq!(h.requests().len(), 1, "{reason:?}");
        assert!(h.predictions().is_empty(), "{reason:?}: the empty turns wait for the result");

        match reason {
            AmdReason::Prediction => {
                let uncertain =
                    Classification { category: Category::Uncertain, inference_duration: ms(5) };
                h.respond(req, Ok(uncertain));
            }
            AmdReason::InferenceError => {
                h.respond(req, Err(ClassifyError::InvalidOutput));
            }
            _ => h.advance(ms(100)),
        }
        assert_eq!(
            h.reasons(),
            vec![(1, reason), (2, AmdReason::Reused), (3, AmdReason::Reused)],
            "{reason:?}"
        );
        let predictions = h.predictions();
        for reused in &predictions[1..] {
            assert_eq!(reused.category, predictions[0].category, "{reason:?}");
            assert_eq!(reused.stage, predictions[0].stage, "{reason:?}");
            assert_eq!(reused.transcript, "", "{reason:?}");
            assert_eq!(reused.inference_duration, None, "{reason:?}");
        }
        // only the latest turn may reply
        assert!(!h.allowed(first) && !h.allowed(second), "{reason:?}");
        assert_eq!(h.decision(current), plain(), "{reason:?}");
        assert_eq!(h.requests().len(), 1, "{reason:?}");

        // reused predictions do not count toward max_uncertain_turns; errors and timeouts neither
        let last_turn = if reason == AmdReason::Prediction { 4 } else { 5 };
        for turn_id in 4..=last_turn {
            h.turn("hello", Category::Uncertain);
            let done = !h.detaches().is_empty();
            assert_eq!(done, turn_id == last_turn, "{reason:?}: turn {turn_id}");
        }
        let outcome = h.outcome();
        assert_eq!(outcome.reason, AmdReason::MaxUncertainTurns, "{reason:?}");
        assert_eq!(outcome.last_prediction.map(|p| p.turn_id), Some(last_turn), "{reason:?}");
    }
}

#[test]
fn test_a_fallback_is_uncertain_and_a_reuse_repeats_the_prediction() {
    for reason in [AmdReason::Reused, AmdReason::InferenceError, AmdReason::InferenceTimeout] {
        let mut h = H::started(AmdOptions { inference_timeout: ms(20), ..AmdOptions::default() });
        h.turn("hello", Category::MachineVm);
        let first = h.last_prediction();
        assert_eq!(first.stage, Stage::MachineVm);

        if reason == AmdReason::Reused {
            h.commit("");
        } else {
            let turn = h.commit("hello");
            if reason == AmdReason::InferenceError {
                h.respond(h.request_for(turn).req, Err(ClassifyError::Api));
            } else {
                h.advance(ms(20));
            }
        }
        let fallback = h.last_prediction();
        assert_eq!(h.predictions().len(), 2, "{reason:?}");
        assert_eq!(fallback.reason, reason);
        // a turn with no prediction is unknown; an empty turn repeats the last one
        let expected =
            if reason == AmdReason::Reused { Category::MachineVm } else { Category::Uncertain };
        assert_eq!(fallback.category, expected, "{reason:?}");
        assert_eq!(fallback.stage, first.stage, "{reason:?}");

        h.step(In::Close);
        let outcome = h.outcome();
        // a reused event does not change the latest prediction
        let latest = if reason == AmdReason::Reused { first } else { fallback };
        assert_eq!(outcome.last_prediction, Some(latest), "{reason:?}");
        assert_eq!(outcome.reason, AmdReason::Cancelled);
    }
}

#[test]
fn test_empty_turn_does_not_cancel_a_pending_terminal_prediction() {
    let mut h = H::started(AmdOptions::default());
    let first = h.commit("hello");
    let second = h.commit("");
    assert!(h.predictions().is_empty());
    h.classify(first, Category::Human);

    // a prediction that completes AMD sends no `ReplyDecided`; `Detach` settles the latest turn
    assert!(h.decisions(second).is_empty());
    assert!(!h.allowed(first));
    assert_eq!(h.detach(), completed_with(plain(), DetachCancel::Keep));
    let outcome = h.outcome();
    let last = outcome.last_prediction.expect("a last prediction");
    assert_eq!((last.turn_id, last.category), (1, Category::Human));
    assert_eq!((outcome.category, outcome.reason), (Stage::Human, AmdReason::Finished));
}

#[test]
fn test_reused_turns_keep_the_corrected_stage() {
    let mut h = H::started(AmdOptions::default());
    h.turn("hello", Category::MachineVm);
    let correction = h.commit("Please state your name.");
    let empty = h.commit("");
    h.classify(correction, Category::MachineScreening);

    assert!(!h.allowed(correction), "the empty turn took the right to reply");
    assert_eq!(h.decision(empty), staged(DEFAULT_SCREENING_INSTRUCTIONS));
    let reused = h.last_prediction();
    assert_eq!((reused.turn_id, reused.reason), (3, AmdReason::Reused));
    assert_eq!(reused.stage, Stage::MachineScreening);

    let next = h.commit("");
    assert_eq!(h.decision(next), staged(DEFAULT_SCREENING_INSTRUCTIONS));
    let reused = h.last_prediction();
    assert_eq!((reused.turn_id, reused.reason), (4, AmdReason::Reused));
    assert_eq!(reused.stage, Stage::MachineScreening);
    assert_eq!(h.requests().len(), 2);
}

#[test]
fn test_empty_turns_preserve_wait_until_new_classification() {
    for pending in [false, true] {
        let mut h = H::started(AmdOptions::default());
        h.turn("Press 1 for sales.", Category::MachineIvr);
        let second = h.commit("Please hold.");
        let latest = if pending { h.commit("") } else { second };
        h.classify(second, Category::Wait);
        assert_eq!(h.decision(latest), deny(), "pending={pending}");

        for _ in 0..2 {
            let turn = h.commit("");
            assert_eq!(h.decision(turn), deny(), "pending={pending}: a reused wait denies");
            let reused = h.last_prediction();
            assert_eq!(reused.reason, AmdReason::Reused);
            assert_eq!((reused.category, reused.stage), (Category::Wait, Stage::MachineIvr));
        }
        assert_eq!(h.requests().len(), 2, "pending={pending}");

        // a new classification ends the wait: a first human after a menu is likely human
        let human = h.turn("Hello?", Category::Human);
        assert_eq!(h.decision(human), held(true), "pending={pending}");
    }
}

// Tool calls in classifier history (spec §8.2, §11.1).

#[test]
fn test_dtmf_on_an_empty_turn_is_context_not_an_inference_trigger() {
    let mut h = H::started(AmdOptions::default());
    let sent = h.dtmf("1");
    h.executed(std::slice::from_ref(&sent));
    h.commit("");
    assert!(h.requests().is_empty());
    // no request is pending: the empty turn reuses the latest prediction at once
    assert_eq!(h.reasons(), vec![(1, AmdReason::Reused)]);
    assert_eq!(h.last_prediction().category, Category::Uncertain);

    let turn = h.commit("Hello.");
    assert_eq!(
        h.request_for(turn).history,
        vec![
            ClassifierItem::DtmfCall(sent.0),
            ClassifierItem::DtmfOutput(sent.1),
            user(2, "Hello.")
        ]
    );
}

#[test]
fn test_dtmf_results_are_collected_once_without_changing_inflight_requests() {
    let mut h = H::started(AmdOptions::default());
    let first_send = h.dtmf("12#");
    h.executed(std::slice::from_ref(&first_send));
    h.executed(std::slice::from_ref(&first_send));
    assert!(h.requests().is_empty(), "a sent digit starts no request");

    let first = h.commit("hello");
    let second_send = h.dtmf("3");
    h.executed(std::slice::from_ref(&second_send));
    let second = h.commit("hello");

    assert_eq!(
        h.request_for(first).history,
        vec![
            ClassifierItem::DtmfCall(first_send.0.clone()),
            ClassifierItem::DtmfOutput(first_send.1.clone()),
            user(1, "hello"),
        ]
    );
    assert_eq!(
        h.request_for(second).history,
        vec![
            ClassifierItem::DtmfCall(first_send.0),
            ClassifierItem::DtmfOutput(first_send.1),
            user(1, "hello"),
            ClassifierItem::DtmfCall(second_send.0),
            ClassifierItem::DtmfOutput(second_send.1),
            user(2, "hello"),
        ]
    );
}

#[test]
fn test_amd_collects_only_successful_dtmf_calls_from_its_run() {
    let mut h = H::started(AmdOptions::default());
    let wall = h.wall();
    let failed = h.tool("send_dtmf_events", "1", true, wall);
    let other = h.tool("another_tool", "2", false, wall);
    let before = h.tool("send_dtmf_events", "3", false, setup().entered_at - secs(1));
    let mut silent = h.tool("send_dtmf_events", "5", false, wall);
    silent.1.output.clear();
    let success = h.dtmf("4");
    h.executed(&[failed, other, before, silent, success.clone()]);

    let turn = h.commit("hello");
    assert_eq!(
        h.request_for(turn).history,
        vec![
            ClassifierItem::DtmfCall(success.0),
            ClassifierItem::DtmfOutput(success.1),
            user(1, "hello")
        ]
    );
}

// Agent replies in classifier history (spec §8.2).

#[test]
fn test_classifier_sees_that_the_agent_spoke_but_not_its_words() {
    let mut h = H::started(AmdOptions::default());
    h.commit("Leave a message after the tone.");
    h.advance(ms(100));
    let wordless = h.created(SpeechKind::Say);
    h.speech(wordless, SpeechObservation::PlayoutAdvanced);
    h.speech(
        wordless,
        SpeechObservation::ItemAdded { text: Some(String::new()), interrupted: false },
    );
    h.speech(wordless, SpeechObservation::Done { interrupted: false, failed: false });
    h.advance(ms(100));
    h.say("Hi Sam, this is Alex.");
    h.advance(ms(100));
    let turn = h.commit("To erase it and start over, press two.");

    assert_eq!(
        messages(&h.request_for(turn)),
        vec![
            ("user", "Leave a message after the tone.".to_owned()),
            ("assistant", AGENT_REPLY.to_owned()),
            ("user", "To erase it and start over, press two.".to_owned()),
        ]
    );
}

#[test]
fn test_agent_reply_keeps_speaking_order_when_a_turn_commits_during_it() {
    let mut h = H::started(AmdOptions::default());
    h.commit("Please state your name.");
    h.advance(ms(100));
    // the reply starts between the two turns but is added when its speech ends
    let speech = h.created(SpeechKind::Say);
    h.speech(speech, SpeechObservation::PlayoutAdvanced);
    h.advance(ms(100));
    h.commit("Okay.");
    h.advance(ms(100));
    let text = Some("This is Alex from Acme.".to_owned());
    h.speech(speech, SpeechObservation::ItemAdded { text, interrupted: false });
    h.speech(speech, SpeechObservation::Done { interrupted: false, failed: false });
    h.advance(ms(100));
    let turn = h.commit("Please hold.");

    assert_eq!(
        messages(&h.request_for(turn)),
        vec![
            ("user", "Please state your name.".to_owned()),
            ("assistant", AGENT_REPLY.to_owned()),
            ("user", "Okay.".to_owned()),
            ("user", "Please hold.".to_owned()),
        ]
    );
}

#[test]
fn test_classifier_sees_the_agent_words_while_a_human_is_held() {
    let mut h = H::started(AmdOptions::default());
    h.turn("hello", Category::MachineIvr);
    h.advance(ms(100));
    h.say("Pressing one.");
    h.advance(ms(100));
    h.turn("Hello?", Category::Human);
    h.advance(ms(100));
    h.say("Hi Sam, can you make it tomorrow?");
    h.advance(ms(100));
    let turn = h.commit("Yes. Goodbye.");

    assert_eq!(
        messages(&h.request_for(turn)),
        vec![
            ("user", "hello".to_owned()),
            ("assistant", AGENT_REPLY.to_owned()),
            ("user", "Hello?".to_owned()),
            ("assistant", "Hi Sam, can you make it tomorrow?".to_owned()),
            ("user", "Yes. Goodbye.".to_owned()),
        ]
    );
}

#[test]
fn test_classifier_sees_the_agent_words_after_an_uncertain_turn_in_a_menu() {
    // wait keeps the placeholder: the agent should not answer hold music
    for (after, reply) in
        [(Category::Uncertain, "Hi Sam, can you make it tomorrow?"), (Category::Wait, AGENT_REPLY)]
    {
        let mut h = H::started(AmdOptions::default());
        h.turn("hello", Category::MachineIvr);
        h.advance(ms(100));
        h.turn("Sam.", after);
        h.advance(ms(100));
        h.say("Hi Sam, can you make it tomorrow?");
        h.advance(ms(100));
        let turn = h.commit("Yes. That's confirmed.");

        assert_eq!(
            messages(&h.request_for(turn)),
            vec![
                ("user", "hello".to_owned()),
                ("user", "Sam.".to_owned()),
                ("assistant", reply.to_owned()),
                ("user", "Yes. That's confirmed.".to_owned()),
            ],
            "after {after:?}"
        );
    }
}

#[test]
fn test_a_reply_shows_its_words_by_the_prediction_it_answered() {
    // the choice is made when the reply is allowed; a later turn's prediction does not change it
    for (after, next, reply) in [
        (Category::Uncertain, Category::MachineIvr, "Hi Sam, can you make it tomorrow?"),
        (Category::MachineIvr, Category::Uncertain, AGENT_REPLY),
    ] {
        let mut h = H::started(AmdOptions::default());
        h.turn("hello", Category::MachineIvr);
        h.advance(ms(100));
        let answered = h.commit("Sam.");
        h.advance(ms(100));
        // a preemptive reply exists before the prediction it answers
        let speech = h.created(SpeechKind::TurnReply);
        h.classify(answered, after);
        assert!(h.allowed(answered), "after {after:?}");
        h.observe(AmdObservation::ReplyCreated { turn: answered, speech });
        h.speech(speech, SpeechObservation::PlayoutAdvanced);
        h.advance(ms(100));

        // the next turn is classified while the reply still plays
        h.turn("Please hold.", next);
        h.advance(ms(100));
        let text = Some("Hi Sam, can you make it tomorrow?".to_owned());
        h.speech(speech, SpeechObservation::ItemAdded { text, interrupted: false });
        h.speech(speech, SpeechObservation::Done { interrupted: false, failed: false });
        h.advance(ms(100));
        let turn = h.commit("Yes. That's confirmed.");

        assert_eq!(
            messages(&h.request_for(turn)),
            vec![
                ("user", "hello".to_owned()),
                ("user", "Sam.".to_owned()),
                ("assistant", reply.to_owned()),
                ("user", "Please hold.".to_owned()),
                ("user", "Yes. That's confirmed.".to_owned()),
            ],
            "after {after:?}"
        );
    }
}

#[test]
fn test_a_greeting_queued_before_amd_is_in_the_history_when_it_plays() {
    let mut h = H::new(AmdOptions::default());
    // reported at attach, before listening starts
    let greeting = h.new_speech();
    let queued =
        SpeechObservation::Created { kind: SpeechKind::Say, add_to_chat_ctx: true, playing: false };
    h.speech(greeting, queued);
    h.observe(AmdObservation::AgentBusy { busy: true, at: h.now });
    h.step(In::Callee(Callee::Answered));

    h.turn("hello", Category::MachineIvr);
    h.advance(ms(100));
    h.play(greeting, "Hi, this is Alex.");
    h.advance(ms(100));
    let turn = h.commit("Press one for sales.");

    assert_eq!(
        messages(&h.request_for(turn)),
        vec![
            ("user", "hello".to_owned()),
            ("assistant", AGENT_REPLY.to_owned()),
            ("user", "Press one for sales.".to_owned()),
        ]
    );
}

#[test]
fn test_a_speech_outside_the_chat_context_has_no_history_entry() {
    // spec §8.2: the session does not record a reply created with `add_to_chat_ctx=False`
    let mut h = H::started(AmdOptions::default());
    h.commit("Please state your name.");
    h.advance(ms(100));
    let speech = h.new_speech();
    let event = SpeechObservation::Created {
        kind: SpeechKind::Say,
        add_to_chat_ctx: false,
        playing: false,
    };
    h.speech(speech, event);
    h.play(speech, "One moment.");
    h.advance(ms(100));
    let turn = h.commit("Hello?");

    assert_eq!(
        messages(&h.request_for(turn)),
        vec![("user", "Please state your name.".to_owned()), ("user", "Hello?".to_owned())]
    );
}

#[test]
fn test_classifier_history_is_bounded_and_saved_decisions_survive() {
    let mut h = H::started(AmdOptions::default());
    for turn_id in 1..24 {
        h.turn(&format!("prompt {turn_id}"), Category::MachineScreening);
    }
    let sent = h.dtmf("2#");
    h.executed(std::slice::from_ref(&sent));
    let turn = h.commit("latest prompt");
    let request = h.request_for(turn);

    assert_eq!(request.turn_id, 24);
    // a DTMF call and its output are two items, as in Python's chat context
    assert_eq!(request.history.len(), HISTORY_LIMIT);
    let ids: Vec<u64> = user_turns(&request).iter().map(|(id, _)| *id).collect();
    assert_eq!(ids, (7..=24).collect::<Vec<_>>());
    assert_eq!(dtmf_calls(&request), vec![sent.0]);
}

#[test]
fn test_empty_turns_preserve_transcript_and_dtmf_history() {
    let mut h = H::started(AmdOptions::default());
    h.turn("For sales press 1.", Category::MachineIvr);
    let sent = h.dtmf("1");
    h.executed(std::slice::from_ref(&sent));
    for _ in 0..22 {
        h.commit("");
    }
    assert_eq!(h.requests().len(), 1);
    let turn = h.commit("One moment please.");
    let request = h.request_for(turn);

    assert_eq!(request.turn_id, 24);
    assert_eq!(
        request.history,
        vec![
            user(1, "For sales press 1."),
            ClassifierItem::DtmfCall(sent.0),
            ClassifierItem::DtmfOutput(sent.1),
            user(24, "One moment please."),
        ]
    );
}

// Transcripts (spec §6.1).

#[test]
fn test_an_empty_commit_keeps_the_session_finals() {
    // the committed transcript replaces the session finals, unless it is empty
    let mut h = H::started(AmdOptions::default());
    h.session_final("Hello");
    h.session_final("there");
    let turn = h.commit("");
    assert_eq!(current_text(&h.request_for(turn)), "Hello there");
    h.classify(turn, Category::Uncertain);
    assert_eq!(h.last_prediction().transcript, "Hello there");

    h.session_final("helo");
    let turn = h.commit("hello");
    assert_eq!(current_text(&h.request_for(turn)), "hello");
}

#[test]
fn test_classifier_sees_both_transcripts_without_changing_agent_transcript() {
    let mut h = H::started_with(AmdOptions::default(), amd_stt());
    h.session_final("session transcript");
    h.amd_final("AMD transcript");
    let turn = h.commit("session transcript");
    assert_eq!(
        decoded(&current_text(&h.request_for(turn))),
        both("session transcript", "AMD transcript")
    );
    h.classify(turn, Category::Uncertain);
    // events use the session transcript when it is not empty
    assert_eq!(h.last_prediction().transcript, "session transcript");

    // without session STT the classifier sees AMD's transcript alone
    let mut h = H::started_with(AmdOptions::default(), amd_stt_only());
    h.amd_final("AMD transcript");
    let turn = h.commit("");
    assert_eq!(current_text(&h.request_for(turn)), "AMD transcript");
    h.classify(turn, Category::Uncertain);
    assert_eq!(h.last_prediction().transcript, "AMD transcript");
}

// The late AMD final (spec §6.2).

#[test]
fn test_the_classifier_waits_briefly_for_a_late_amd_final() {
    let mut h = H::started_with(AmdOptions::default(), amd_stt());
    let at = h.now;
    let turn = h.commit("Sam Lee speaking.");
    assert_eq!(h.armed(AmdTimer::LateFinal(turn)), Some(at.saturating_add(LATE_AMD_FINAL_WAIT)));
    h.advance(ms(100));
    assert_eq!(h.requests_for(turn), 0, "the request waits for the AMD final");

    h.amd_final("Sam Lee speaking.");
    assert_eq!(h.requests_for(turn), 1);
    let request = h.request_for(turn);
    assert_eq!(decoded(&current_text(&request)), both("Sam Lee speaking.", "Sam Lee speaking."));
    // the deadline starts at commit and includes the wait
    assert_eq!(h.armed(AmdTimer::Inference(request.req)), Some(at.saturating_add(secs(2))));
}

#[test]
fn test_a_late_amd_transcript_does_not_empty_the_turn() {
    let mut h = H::started_with(AmdOptions::default(), amd_stt());
    h.amd_final("Hi, you've reached Sam.");
    // a turn that has its AMD text needs no wait
    let first = h.commit("Hi, you've reached Sam.");
    assert_eq!(h.requests_for(first), 1);
    assert_eq!(h.armed(AmdTimer::LateFinal(first)), None);
    let first_text = current_text(&h.request_for(first));
    assert_eq!(decoded(&first_text), both("Hi, you've reached Sam.", "Hi, you've reached Sam."));
    h.classify(first, Category::Uncertain);

    let second = h.commit("Leave your name and number.");
    assert_eq!(h.requests_for(second), 0);
    h.advance(LATE_AMD_FINAL_WAIT);
    let request = h.request_for(second);
    assert_eq!(decoded(&current_text(&request)), both("Leave your name and number.", ""));
    assert_eq!(user_turns(&request)[0], (1, first_text));
}

#[test]
fn test_a_blank_final_keeps_the_turn_waiting() {
    let mut h = H::started_with(AmdOptions::default(), amd_stt());
    let turn = h.commit("Hello.");
    h.amd_final("  ");
    assert_eq!(h.requests_for(turn), 0);
    h.amd_final("Hello there.");
    assert_eq!(decoded(&current_text(&h.request_for(turn))), both("Hello.", "Hello there."));
}

#[test]
fn test_without_session_stt_a_turn_is_classified_when_its_final_arrives() {
    let mut h = H::started_with(AmdOptions::default(), amd_stt_only());
    let turn = h.commit("");
    h.advance(ms(100));
    assert!(h.requests().is_empty());
    assert!(h.predictions().is_empty());

    let heard = h.now;
    h.amd_final("Please state your name.");
    let request = h.request_for(turn);
    assert_eq!(current_text(&request), "Please state your name.");
    // without text at commit, the deadline starts when the final arrives
    assert_eq!(h.armed(AmdTimer::Inference(request.req)), Some(heard.saturating_add(secs(2))));
    h.classify(turn, Category::MachineScreening);
    assert_eq!(h.last_prediction().transcript, "Please state your name.");
}

#[test]
fn test_transcript_received_after_eot_is_kept_in_next_request() {
    let mut h = H::started_with(AmdOptions::default(), amd_stt_only());
    h.commit("");
    h.advance(LATE_AMD_FINAL_WAIT + ms(10));
    // after the wait, a final is the next turn's
    assert_eq!(h.reasons(), vec![(1, AmdReason::Reused)]);
    assert_eq!(h.last_prediction().transcript, "");
    h.amd_final("Please state your name.");
    assert!(h.requests().is_empty());

    let second = h.commit("");
    let first = h.request_for(second);
    assert_eq!(first.turn_id, 2);
    assert_eq!(current_text(&first), "Please state your name.");
    h.classify(second, Category::MachineScreening);
    assert_eq!(h.last_prediction().transcript, "Please state your name.");

    h.amd_final("Okay, connecting you.");
    let third = h.commit("");
    let request = h.request_for(third);
    assert_eq!(
        user_turns(&request),
        vec![(2, "Please state your name.".to_owned()), (3, "Okay, connecting you.".to_owned())]
    );
}

#[test]
fn test_a_reused_prediction_is_kept_when_the_turn_final_arrives() {
    // without session STT, turn 2 waits for its final while turn 1 is still classified
    let mut h = H::started_with(AmdOptions::default(), amd_stt_only());
    let first = h.commit("");
    h.amd_final("Please state your name.");
    h.request_for(first);
    h.commit("");
    h.classify(first, Category::MachineScreening);
    assert_eq!(h.reasons(), vec![(1, AmdReason::Prediction), (2, AmdReason::Reused)]);

    // a prediction is never revised: the final goes to history for the next request
    h.amd_final("Why are you calling?");
    assert_eq!(h.requests().len(), 1);
    assert_eq!(h.predictions().len(), 2);

    let third = h.commit("");
    h.amd_final("Hello?");
    assert_eq!(
        user_turns(&h.request_for(third)),
        vec![
            (1, "Please state your name.".to_owned()),
            (2, "Why are you calling?".to_owned()),
            (3, "Hello?".to_owned()),
        ]
    );
}

#[test]
fn test_an_older_turn_wait_does_not_decide_a_newer_waiting_turn() {
    let mut h = H::started_with(AmdOptions::default(), amd_stt_only());
    h.commit("");
    h.advance(ms(150));
    let second = h.commit("");
    // turn 1's wait has expired, turn 2's has not
    h.advance(ms(170));
    assert!(h.predictions().is_empty());
    h.amd_final("Hello, who is this?");
    let request = h.request_for(second);
    assert_eq!(current_text(&request), "Hello, who is this?");
    assert_eq!(h.requests().len(), 1);
}

#[test]
fn test_an_older_turn_final_does_not_cancel_a_newer_turn() {
    let mut h = H::started_with(AmdOptions::default(), amd_stt_only());
    h.commit("");
    // turn 1's final, then the next turn's words before it commits
    h.amd_final("Please hold.");
    h.amd_final("Hello, who is this?");
    let second = h.commit("");
    let request = h.requests().pop().expect("a request");
    assert_eq!(request.turn, second);
    assert_eq!(current_text(&request), "Hello, who is this?");

    h.classify(second, Category::Human);
    let outcome = h.outcome();
    assert_eq!(outcome.last_prediction.map(|p| p.turn_id), Some(2));
    assert!(h.predictions().iter().all(|p| p.turn_id == 2));
}

#[test]
fn test_late_final_after_many_empty_turns_is_committed_at_the_next_eot() {
    let mut h = H::started_with(AmdOptions::default(), amd_stt_only());
    let sent = h.dtmf("1");
    h.executed(std::slice::from_ref(&sent));
    for _ in 0..21 {
        h.commit("");
    }
    h.advance(LATE_AMD_FINAL_WAIT + ms(10));
    // only the latest turn takes a reused prediction; older turns never reply
    assert_eq!(h.reasons(), vec![(21, AmdReason::Reused)]);

    h.amd_final("Hello, can you hear me?");
    assert!(h.requests().is_empty());
    let turn = h.commit("");
    let request = h.request_for(turn);
    assert_eq!(request.turn_id, 22);
    assert_eq!(user_turns(&request), vec![(22, "Hello, can you hear me?".to_owned())]);
    assert_eq!(dtmf_calls(&request), vec![sent.0]);
    h.classify(turn, Category::Human);
    assert_eq!(h.outcome().category, Stage::Human);
}

// AMD STT failure (spec §6.3).

#[test]
fn test_amd_failure_leaves_the_session_transcript_for_the_open_turn_on() {
    let mut h = H::started_with(AmdOptions::default(), amd_stt());
    h.amd_final("AMD words");
    h.step(In::SttFailed);
    let first = h.commit("session words");
    // no wait, and the session transcript alone
    assert_eq!(current_text(&h.request_for(first)), "session words");
    let second = h.commit("more words");
    assert_eq!(current_text(&h.request_for(second)), "more words");
}

#[test]
fn test_stt_failure_during_the_late_final_wait_keeps_the_committed_turn() {
    // committed turns stay unchanged (spec §6.3): the turn keeps its empty AMD transcript. The
    // proposal stops the wait at once; Python waits it out. Either way the request follows.
    let mut h = H::started_with(AmdOptions::default(), amd_stt());
    let turn = h.commit("Hello.");
    assert_eq!(h.requests_for(turn), 0);
    h.step(In::SttFailed);
    h.advance(LATE_AMD_FINAL_WAIT);
    assert_eq!(h.requests_for(turn), 1);
    assert_eq!(decoded(&current_text(&h.request_for(turn))), both("Hello.", ""));
    // later turns use the session transcript alone, with no wait
    let next = h.commit("Hi.");
    assert_eq!(current_text(&h.request_for(next)), "Hi.");
}

// Speech duration (spec §7.1, §16.2).

#[test]
fn test_speech_timing_is_independent_of_event_creation() {
    let mut h = H::started(AmdOptions::default());
    // each boundary reaches AMD after the time it describes
    h.advance(ms(1200));
    h.speech_started(T0.saturating_add(secs(1)));
    h.advance(ms(1300));
    h.speech_ended(T0.saturating_add(secs(2)));
    h.advance(ms(500));
    let turn = h.commit("hello");
    assert_eq!(h.request_for(turn).speech_duration, secs(1));

    h.advance(ms(200));
    let req = h.request_for(turn).req;
    let screening =
        Classification { category: Category::MachineScreening, inference_duration: ms(150) };
    h.respond(req, Ok(screening));
    let prediction = h.last_prediction();
    assert_eq!(prediction.speech_duration, secs(1));
    assert_eq!(prediction.delay, ms(200));
    assert_eq!(prediction.inference_duration, Some(ms(150)));
}

#[test]
fn test_speech_across_a_commit_is_split_at_the_commit() {
    let mut h = H::started(AmdOptions::default());
    h.advance(secs(1));
    h.speech_started(h.now);
    h.advance(secs(1));
    let first = h.commit_with(Commit { speaking: true, ..Commit::text("Please hold") });
    assert_eq!(h.request_for(first).speech_duration, secs(1));

    h.advance(secs(1));
    h.speech_ended(h.now);
    h.advance(secs(1));
    let second = h.commit("the line.");
    assert_eq!(h.request_for(second).speech_duration, secs(1));
}
