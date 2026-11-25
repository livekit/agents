//! AMD's executor on a call: the real executor answers the gate from what the fake models answer
//! (proposal, implementation step 3). Each story names the `tests/test_amd.py` scenario on `main`
//! it ports, and asserts what the caller and the host would notice: what the models were asked,
//! what AMD told the host, what `execute()` returned, and what was said and heard.

use super::{Call, detecting};
use lk_agents_fuzz::fakes::amd::{
    AmdSetup, HUMAN_AFTER_MACHINE, IVR, MENU, SCREENING, SEND_DTMF_TOOL, VOICEMAIL,
};
use lk_agents_fuzz::input::{
    AgentText, AmdSays, Count, Input, Millis, Ref, RoomIn, SayInput, UserText,
};
use lk_agents_fuzz::record::amd_run::AskKind;
use lk_agents_fuzz::record::{Detail, Kind, Tag};
use lk_agents_runtime::voice::amd::classifier::{CLASSIFY_PROMPT, MENU_PROMPT, RECORD_RESULT};
use lk_agents_voice::amd::events::{AmdEvent, AmdReason, Category, Stage};
use lk_agents_voice::amd::options::AmdOptions;
use lk_agents_voice::trace::{AmdStep, GateStep};
use std::time::Duration;

const LINE: &str = "Hi, this is the clinic calling.";
/// About six seconds of speech.
const MESSAGE: &str = "Hello, this is the clinic calling about your appointment on Tuesday at \
nine. Please call us back at five five five, one two three four, at your convenience. Thank you.";

fn amd() -> AmdSetup {
    AmdSetup::default()
}

fn with(options: AmdOptions) -> AmdSetup {
    AmdSetup { options, ..AmdSetup::default() }
}

impl Call {
    /// The Ref of AMD's own recognition stream, while it is open.
    fn amd_stream(&self) -> Ref<lk_agents_fuzz::input::Stt> {
        let live = self.run.record.live().of(Kind::Stt);
        let nth = (self.run.record.resources(Kind::Stt))
            .filter(|r| *r.detail() == Detail::AmdStt && live.contains(&r.nth()))
            .map(|r| r.nth())
            .last()
            .expect("AMD's recogniser is open");
        Ref::new(nth)
    }

    /// The person says `session` as the session's recogniser hears it and `amd` as AMD's does,
    /// both finals before the turn commits.
    async fn says_to_both(&mut self, session: &str, amd: &str) {
        let amd_stream = self.amd_stream();
        let (mine, theirs) = (Ref::new(self.utterances), Ref::new(self.utterances + 1));
        self.utterances += 2;
        self.send(Input::VadStart { vad: Ref::new(0), speech_ms: Millis(0) }).await;
        self.send(Input::SttStart { stt: Ref::new(0) }).await;
        self.send(Input::SttStart { stt: amd_stream }).await;
        self.advance(300).await;
        let speaker = None;
        self.send(Input::SttFinal { utterance: mine, text: UserText(session.into()), speaker })
            .await;
        self.send(Input::SttFinal { utterance: theirs, text: UserText(amd.into()), speaker }).await;
        self.send(Input::VadEnd { vad: Ref::new(0), silence_ms: Millis(0) }).await;
        self.send(Input::SttEnd { utterance: mine }).await;
        self.send(Input::SttEnd { utterance: theirs }).await;
        self.advance(600).await;
    }

    /// The last user message of each classification, in order.
    fn classified_words(&self) -> Vec<String> {
        (self.amd_asks_of(AskKind::Classify).into_iter())
            .filter_map(|a| a.messages.last().map(|m| m.text.clone()))
            .collect()
    }

    /// The outcome every `execute()` returned, which must be one.
    fn the_outcome(&self) -> lk_agents_voice::amd::events::Outcome {
        let outcomes = self.outcomes();
        assert_eq!(outcomes.len(), 1, "execute() returned once: {outcomes:?}");
        outcomes[0].clone()
    }

    fn assert_clean(&self) {
        let failures = self.failures();
        assert!(failures.is_empty(), "{failures:?}");
    }
}

// Classification and completion.

/// A person answers at once: one classification, completion as `human`, and the turn's reply
/// under the agent's own instructions. Python: `test_a_human_at_call_start_completes_at_once`,
/// `test_human_instructions_are_skipped_without_a_machine_stage`.
#[test]
fn a_direct_human_answer_completes_and_gets_a_plain_reply() {
    let call = Call::run(detecting(amd()), async |call| {
        call.enter().await;
        call.says("hello who is this").await;
        call.classifier_says(AmdSays::Human).await;
        call.advance(100).await;
        call.model_answers(LINE).await;
        call.advance(3_000).await;
    });
    let asks = call.amd_asks_of(AskKind::Classify);
    assert_eq!(asks.len(), 1, "one classification");
    let ask = &asks[0];
    assert_eq!(ask.messages.first().map(|m| m.text.as_str()), Some(CLASSIFY_PROMPT));
    assert!(ask.state().is_some_and(|s| s.starts_with(r#"{"stage": "uncertain""#)), "{ask:?}");
    assert_eq!(ask.messages.last().map(|m| m.text.as_str()), Some("hello who is this"));
    assert_eq!(ask.tools, vec![RECORD_RESULT.to_owned()]);
    assert!(ask.required && !ask.tagged, "required, and no header for a plugin model");

    let outcome = call.the_outcome();
    assert_eq!((outcome.category, outcome.reason), (Stage::Human, AmdReason::Finished));
    assert!(!outcome.voicemail_message_played && !outcome.is_error());
    let last = outcome.last_prediction.clone().expect("the turn's prediction");
    assert_eq!((last.turn_id, last.category, last.stage), (1, Category::Human, Stage::Human));
    assert_eq!(last.reason, AmdReason::Prediction);
    assert_eq!(last.transcript, "hello who is this");
    assert_eq!(call.predictions(), vec![last], "one `amd_prediction`");
    assert_eq!(call.completions(), vec![outcome], "one `amd_completed`, as `execute()` returned");

    let heard = call.heard_requests("hello who is this");
    assert_eq!(heard.len(), 1, "the turn is answered once");
    assert!(!heard[0].carries(HUMAN_AFTER_MACHINE), "no stage instructions for a direct answer");
    call.assert_clean();
}

/// A screener, then a first `human` held as likely human, then `human` again: the held turn
/// replies with no stage instructions, the next request tells the model the turn is likely a
/// person, and the completing reply carries the human instructions. Python:
/// `test_a_first_human_after_a_menu_or_screener_waits_for_the_next_turn`.
#[test]
fn a_first_human_after_a_screener_waits_for_the_next_turn() {
    let call = Call::run(detecting(amd()), async |call| {
        call.enter().await;
        call.says("hi this is call assist please state your name").await;
        call.classifier_says(AmdSays::Screening).await;
        call.advance(2_000).await;
        call.model_answers(LINE).await;
        call.advance(4_000).await;
        call.says("hello").await;
        call.classifier_says(AmdSays::Human).await;
        call.advance(2_000).await;
        call.model_answers(LINE).await;
        call.advance(3_000).await;
        call.says("hello who is this").await;
        call.classifier_says(AmdSays::Human).await;
        call.advance(2_000).await;
        call.model_answers(LINE).await;
        call.advance(3_000).await;
    });
    let predictions = call.predictions();
    let got: Vec<_> = predictions.iter().map(|p| (p.turn_id, p.category, p.stage)).collect();
    assert_eq!(
        got,
        vec![
            (1, Category::MachineScreening, Stage::MachineScreening),
            (2, Category::Uncertain, Stage::MachineScreening),
            (3, Category::Human, Stage::Human),
        ]
    );
    let asks = call.amd_asks_of(AskKind::Classify);
    assert_eq!(asks.len(), 3);
    let state = asks[2].state().unwrap_or_default();
    assert!(state.contains(r#""category": "uncertain (likely human)""#), "{state}");
    assert!(state.contains(r#""stage": "machine-screening""#), "{state}");

    let screened = call.heard_requests("hi this is call assist please state your name");
    assert!(screened.iter().any(|q| q.carries(SCREENING)), "the screener is answered by stage");
    let held = call.heard_requests("hello");
    assert_eq!(held.len(), 1, "the held turn is answered");
    assert!(!held[0].carries(SCREENING) && !held[0].carries(HUMAN_AFTER_MACHINE), "no stage");
    let person = call.heard_requests("hello who is this");
    assert_eq!(person.len(), 1);
    assert!(person[0].carries(HUMAN_AFTER_MACHINE), "the human instructions after a machine");

    let outcome = call.the_outcome();
    assert_eq!((outcome.category, outcome.reason), (Stage::Human, AmdReason::Finished));
    assert_eq!(outcome.last_prediction.map(|p| p.turn_id), Some(3));
    call.assert_clean();
}

/// `wait` keeps the stage and the turn gets no reply; the next turn is classified on its own.
/// Python: `test_wait_skips_reply_and_allows_a_fresh_prediction`.
#[test]
fn a_wait_turn_is_not_answered_and_the_next_turn_is_classified() {
    let call = Call::run(detecting(amd()), async |call| {
        call.enter().await;
        call.says("please hold while we connect you").await;
        call.classifier_says(AmdSays::Wait).await;
        call.advance(3_000).await;
        call.model_answers(LINE).await;
        call.says("hello you are speaking with dana").await;
        call.classifier_says(AmdSays::Human).await;
        call.advance(100).await;
        call.model_answers(LINE).await;
        call.advance(3_000).await;
    });
    assert!(call.unanswered("please hold while we connect you"), "wait never replies");
    assert_eq!(call.heard_requests("hello you are speaking with dana").len(), 1);
    let got: Vec<_> = call.predictions().iter().map(|p| (p.category, p.stage)).collect();
    assert_eq!(got, vec![(Category::Wait, Stage::Uncertain), (Category::Human, Stage::Human)]);
    assert_eq!(call.saved_user_words().len(), 2, "the wait turn keeps its words");
    call.assert_clean();
}

/// Three `uncertain` turns in a row complete the run. Python:
/// `test_three_uncertain_predictions_complete_detection`.
#[test]
fn three_uncertain_turns_complete_the_run() {
    let call = Call::run(detecting(amd()), async |call| {
        call.enter().await;
        for words in ["hello", "hello hello", "is anyone there"] {
            call.says(words).await;
            call.classifier_says(AmdSays::Uncertain).await;
            call.advance(100).await;
            call.model_answers(LINE).await;
            call.advance(3_000).await;
        }
    });
    let outcome = call.the_outcome();
    assert_eq!(
        (outcome.category, outcome.reason),
        (Stage::Uncertain, AmdReason::MaxUncertainTurns)
    );
    assert_eq!(call.predictions().len(), 3);
    call.assert_clean();
}

/// Requests the model never answers time out after `inference_timeout`, each an `uncertain`
/// prediction; the third completes the run. Python: `test_inference_timeout_limit_completes_detection`.
#[test]
fn three_inference_timeouts_complete_the_run() {
    let call = Call::run(detecting(amd()), async |call| {
        call.enter().await;
        for words in ["hello", "hello hello", "is anyone there"] {
            call.says(words).await;
            call.advance(2_500).await;
            call.model_answers(LINE).await;
            call.advance(2_000).await;
        }
    });
    let reasons: Vec<_> = call.predictions().iter().map(|p| (p.reason, p.category)).collect();
    assert_eq!(reasons, vec![(AmdReason::InferenceTimeout, Category::Uncertain); 3]);
    assert!(call.predictions().iter().all(|p| p.inference_duration.is_none()));
    let outcome = call.the_outcome();
    assert_eq!(outcome.reason, AmdReason::InferenceTimeout);
    call.assert_clean();
}

/// An answer that is not a valid `record_result` call is an `inference_error`: the turn is
/// uncertain and still answered. Python: `test_invalid_model_output_falls_back_and_releases_the_reply`.
#[test]
fn a_garbled_answer_is_an_inference_error_and_the_turn_is_answered() {
    let call = Call::run(detecting(amd()), async |call| {
        call.enter().await;
        call.says("hello who is this").await;
        call.classifier_says(AmdSays::Garbled).await;
        call.advance(100).await;
        call.model_answers(LINE).await;
        call.advance(3_000).await;
    });
    let predictions = call.predictions();
    assert_eq!(predictions.len(), 1);
    assert_eq!(
        (predictions[0].reason, predictions[0].category),
        (AmdReason::InferenceError, Category::Uncertain)
    );
    assert_eq!(predictions[0].inference_duration, None);
    assert_eq!(call.heard_requests("hello who is this").len(), 1, "the reply is released");
    assert!(call.outcomes().is_empty(), "the run goes on");
    call.assert_clean();
}

/// A classifier failure outside the provider taxonomy ends the run with `internal_error`.
/// Python: `test_unexpected_inference_errors_finish_with_internal_error`.
#[test]
fn a_crashing_classifier_ends_the_run_with_internal_error() {
    let call = Call::run(detecting(amd()), async |call| {
        call.enter().await;
        call.says("hello who is this").await;
        call.classifier_says(AmdSays::Crashed).await;
        call.advance(3_000).await;
    });
    let outcome = call.the_outcome();
    assert_eq!(outcome.reason, AmdReason::InternalError);
    assert!(outcome.is_error());
    assert!(call.predictions().is_empty(), "no prediction for the crashed request");
    call.assert_clean();
}

/// A newer turn with text cancels the request in flight; its late answer decides nothing, and
/// the newer turn is decided by its own request. Python:
/// `test_classifier_that_ignores_cancellation_cannot_change_state`,
/// `test_timeout_rearms_and_late_result_cannot_change_a_newer_turn`.
#[test]
fn a_stale_answer_cannot_decide_a_newer_turn() {
    let call = Call::run(detecting(amd()), async |call| {
        call.enter().await;
        call.says("hello").await;
        let first = call.open_ask(AskKind::Classify).expect("the first request");
        call.says("please leave a message after the tone").await;
        call.send(Input::AmdReply { ask: first, says: AmdSays::Human }).await;
        call.advance(100).await;
        call.classifier_says(AmdSays::Voicemail).await;
        call.advance(2_000).await;
        call.model_answers(MESSAGE).await;
        call.advance(8_000).await;
    });
    let replies = call.run.record.amd_run().replies().to_vec();
    assert!(!replies[0].accepted, "the cancelled request took no answer: {replies:?}");
    let got: Vec<_> = call.predictions().iter().map(|p| (p.turn_id, p.category)).collect();
    assert_eq!(got, vec![(2, Category::MachineVm)], "turn 1's request was cancelled");
    let completed_as_human = call.outcomes().iter().any(|o| o.category == Stage::Human);
    assert!(!completed_as_human, "the stale `human` never completed the run");
    call.assert_clean();
}

/// A stage the application handles completes the run with no reply. Python:
/// `test_stage_without_instructions_completes_without_a_reply`.
#[test]
fn a_stage_without_instructions_completes_without_a_reply() {
    let options =
        AmdOptions { screening_instructions: None, ..lk_agents_fuzz::fakes::amd::options() };
    let call = Call::run(detecting(with(options)), async |call| {
        call.enter().await;
        call.says("hi this is call assist please state your name").await;
        call.classifier_says(AmdSays::Screening).await;
        call.advance(100).await;
        call.model_answers(LINE).await;
        call.advance(3_000).await;
    });
    let outcome = call.the_outcome();
    assert_eq!((outcome.category, outcome.reason), (Stage::MachineScreening, AmdReason::Finished));
    assert_eq!(outcome.last_prediction.map(|p| p.turn_id), Some(1));
    assert!(call.unanswered("hi this is call assist please state your name"));
    assert!(call.amd_asks_of(AskKind::Menu).is_empty());
    call.assert_clean();
}

// Voicemail (spec §10.4).

/// A voicemail greeting gets one message after the participant's silence; once it plays to its
/// end, the run reports it and completes when the box goes idle. Python:
/// `test_voicemail_sends_one_delivered_message_and_records_playback`.
#[test]
fn the_voicemail_message_plays_once_and_is_reported() {
    let options = AmdOptions {
        voicemail_idle_timeout: Duration::from_secs(3),
        ..lk_agents_fuzz::fakes::amd::options()
    };
    let call = Call::run(detecting(with(options)), async |call| {
        call.enter().await;
        call.says("please leave a message after the tone").await;
        call.classifier_says(AmdSays::Voicemail).await;
        call.advance(2_000).await;
        call.model_answers(MESSAGE).await;
        call.advance(15_000).await;
    });
    let heard = call.heard_requests("please leave a message after the tone");
    assert_eq!(heard.len(), 1, "one message");
    assert!(heard[0].carries(VOICEMAIL), "the voicemail instructions");
    let played: Vec<_> = (call.run.record.amd_run().steps().into_iter())
        .filter(|(_, s)| matches!(s, AmdStep::VoicemailPlayed { .. }))
        .collect();
    assert_eq!(played.len(), 1, "{played:?}");
    let outcome = call.the_outcome();
    assert_eq!((outcome.category, outcome.reason), (Stage::MachineVm, AmdReason::IdleTimeout));
    assert!(outcome.voicemail_message_played);
    call.assert_clean();
}

// Menus (spec §11.2).

/// A menu stage extracts the turn's menu with the menu prompt; the menu reaches the host, and
/// the turn's reply offers AMD's DTMF tool with the IVR instructions. Python:
/// `test_menu_is_observability_only_and_dtmf_tool_is_temporary`.
#[test]
fn a_menu_is_observed_and_the_reply_offers_dtmf() {
    let call = Call::run(detecting(amd()), async |call| {
        call.enter().await;
        call.says("for billing press one for support press two").await;
        call.classifier_says(AmdSays::Ivr).await;
        call.advance(50).await;
        call.menu_says(AmdSays::Menu).await;
        call.advance(2_000).await;
        call.model_answers(LINE).await;
        call.advance(3_000).await;
    });
    let menus = call.amd_asks_of(AskKind::Menu);
    assert_eq!(menus.len(), 1, "one extraction");
    assert_eq!(menus[0].messages.first().map(|m| m.text.as_str()), Some(MENU_PROMPT));
    let transcript = r#"{"transcript": "for billing press one for support press two"}"#;
    assert_eq!(menus[0].messages.last().map(|m| m.text.as_str()), Some(transcript));
    assert!(menus[0].required && menus[0].tools == vec![RECORD_RESULT.to_owned()]);

    let run = call.amd_run();
    let observed: Vec<_> = (call.run.record.amd_run().events(run).into_iter())
        .filter_map(|(_, e)| match e {
            AmdEvent::MenuObserved(menu) => Some(menu),
            _ => None,
        })
        .collect();
    assert_eq!(observed.len(), 1);
    assert_eq!((observed[0].turn_id, observed[0].menu.as_str()), (1, MENU));
    let keys: Vec<_> = observed[0].options.iter().map(|o| o.dtmf.as_str()).collect();
    assert_eq!(keys, vec!["1", "2"]);

    let heard = call.heard_requests("for billing press one for support press two");
    assert_eq!(heard.len(), 1);
    assert!(heard[0].carries(IVR) && heard[0].offers(SEND_DTMF_TOOL) == 1, "{:?}", heard[0]);
    call.assert_clean();
}

/// A newer turn with text stops the menu extraction in flight; classification goes on. Python:
/// `test_slow_menu_does_not_block_classification_or_next_turn`.
#[test]
fn a_newer_turn_stops_a_slow_menu() {
    let call = Call::run(detecting(amd()), async |call| {
        call.enter().await;
        call.says("for billing press one").await;
        call.classifier_says(AmdSays::Ivr).await;
        call.advance(100).await;
        // The next turn comes within the extraction's 5 s.
        let menu = call.open_ask(AskKind::Menu).expect("the extraction waits");
        call.says("hello this is dana").await;
        assert!(call.open_ask(AskKind::Menu).is_none(), "the newer turn stopped the extraction");
        call.send(Input::AmdReply { ask: menu, says: AmdSays::Menu }).await;
        // After a menu, a first `human` is held as likely human; a second confirms it
        // (Python's `confirm_human`).
        call.classifier_says(AmdSays::Human).await;
        call.advance(2_000).await;
        call.model_answers(LINE).await;
        call.advance(3_000).await;
        call.says("yes this is dana speaking").await;
        call.classifier_says(AmdSays::Human).await;
        call.advance(100).await;
        call.model_answers(LINE).await;
        call.advance(3_000).await;
    });
    let replies = call.run.record.amd_run().replies().to_vec();
    let stopped = replies.iter().find(|r| r.says == AmdSays::Menu).expect("the late menu answer");
    assert!(!stopped.accepted, "the stopped extraction took no answer");
    let run = call.amd_run();
    assert_eq!(call.run.record.amd_run().menus(run), 0, "the stopped menu is never observed");
    let outcome = call.the_outcome();
    assert_eq!((outcome.category, outcome.reason), (Stage::Human, AmdReason::Finished));
    call.assert_clean();
}

/// A decision model classifies with one choice per turn and never extracts a menu. Python:
/// `test_decision_model_classifies_with_one_choice_and_skips_menus`.
#[test]
fn a_decision_model_chooses_and_never_extracts_menus() {
    let call = Call::run(detecting(AmdSetup { decision: true, ..amd() }), async |call| {
        call.enter().await;
        call.says("thanks for calling for billing press one").await;
        call.classifier_says(AmdSays::Ivr).await;
        call.advance(2_000).await;
        call.model_answers(LINE).await;
        call.advance(3_000).await;
        call.says("for billing press one").await;
        call.classifier_says(AmdSays::Unavailable).await;
        call.advance(3_000).await;
    });
    let choices = call.amd_asks_of(AskKind::Choice);
    assert_eq!(choices.len(), 2, "one choice per turn");
    let first = choices[0].input.clone().unwrap_or_default();
    assert!(first.starts_with(r#"Request: {"stage": "uncertain""#), "{first}");
    assert!(
        first.ends_with("\nTranscript:\nuser: thanks for calling for billing press one"),
        "{first}"
    );
    let second = choices[1].input.clone().unwrap_or_default();
    assert!(second.starts_with(r#"Request: {"stage": "machine-ivr""#), "{second}");
    assert!(
        second.contains("user: thanks for calling for billing press one\n")
            && second.ends_with("user: for billing press one"),
        "{second}"
    );
    let every: Vec<String> = Category::ALL.iter().map(|c| c.as_str().to_owned()).collect();
    assert_eq!(choices[0].options, every);
    assert!(
        call.amd_asks_of(AskKind::Menu).is_empty()
            && call.amd_asks_of(AskKind::Classify).is_empty()
    );
    let run = call.amd_run();
    assert_eq!(call.run.record.amd_run().menus(run), 0);
    let outcome = call.the_outcome();
    assert_eq!(
        (outcome.category, outcome.reason),
        (Stage::MachineUnavailable, AmdReason::Finished)
    );
    call.assert_clean();
}

/// An Inference model's requests carry AMD's client feature header. Python:
/// `test_inference_decision_model_is_tagged`, `client_feature_kwargs`.
#[test]
fn an_inference_classifier_is_tagged() {
    let call = Call::run(detecting(AmdSetup { inference: true, ..amd() }), async |call| {
        call.enter().await;
        call.says("hello who is this").await;
        call.classifier_says(AmdSays::Human).await;
        call.advance(3_000).await;
    });
    let asks = call.amd_asks_of(AskKind::Classify);
    assert_eq!(asks.len(), 1);
    assert!(asks[0].tagged);
}

// AMD's own STT (spec §6).

/// With AMD's own recogniser, the classifier reads both transcripts, while the turn the agent
/// saves is the session's. Python:
/// `test_classifier_sees_both_transcripts_without_changing_agent_transcript`.
#[test]
fn the_classifier_reads_both_transcripts_and_the_session_keeps_its_own() {
    let call = Call::run(detecting(AmdSetup { own_stt: true, ..amd() }), async |call| {
        call.enter().await;
        call.says_to_both("hello who is this", "hello, who is this?").await;
        call.classifier_says(AmdSays::Human).await;
        call.advance(100).await;
        call.model_answers(LINE).await;
        call.advance(3_000).await;
    });
    let dual = r#"{"transcript_1": "hello who is this", "transcript_2": "hello, who is this?"}"#;
    assert_eq!(call.classified_words(), vec![dual.to_owned()]);
    assert_eq!(call.saved_user_words(), vec!["hello who is this".to_owned()]);
    let outcome = call.the_outcome();
    assert_eq!(outcome.last_prediction.map(|p| p.transcript), Some("hello who is this".into()));
    call.assert_clean();
}

/// Once AMD's recogniser fails, turns are classified on the session's transcript alone, at once.
/// Python: `test_amd_failure_leaves_the_session_transcript_for_the_open_turn_on`; the proposal
/// ends every late-final wait at once (`SttFailed`).
#[test]
fn after_amds_stt_fails_the_session_transcript_is_classified_alone() {
    let call = Call::run(detecting(AmdSetup { own_stt: true, ..amd() }), async |call| {
        call.enter().await;
        let stream = call.amd_stream();
        call.send(Input::SttFail { stt: stream }).await;
        call.advance(100).await;
        call.says("hello who is this").await;
        assert!(call.open_ask(AskKind::Classify).is_some(), "no wait for a final that cannot come");
        call.classifier_says(AmdSays::Human).await;
        call.advance(3_000).await;
    });
    assert_eq!(call.classified_words(), vec!["hello who is this".to_owned()]);
    call.assert_clean();
}

/// AMD's recogniser hears the person's real audio while the session's own recogniser hears
/// silence behind an uninterruptible speech (`agent_activity.push_audio`: AMD, like VAD, keeps
/// the real frame). The proposal leaves the feed's mechanism open, not its result.
#[test]
fn amds_stt_hears_the_person_while_the_session_stt_hears_silence() {
    let call = Call::run(detecting(AmdSetup { own_stt: true, ..amd() }), async |call| {
        // Queued before AMD attaches, it keeps its interruptibility.
        call.send(Input::Say {
            text: AgentText(MESSAGE.into()),
            input: SayInput::Words,
            allow: Some(false),
            to_ctx: true,
        })
        .await;
        call.enter().await;
        call.advance(500).await;
        let mic = Ref::new(*call.run.record.live().of(Kind::Mic).first().expect("a microphone"));
        call.send(Input::MicFrames { mic, chunks: Count(20) }).await;
        call.advance(300).await;
    });
    let amd_stream = (call.run.record.resources(Kind::Stt))
        .find(|r| *r.detail() == Detail::AmdStt)
        .map(|r| r.nth())
        .expect("AMD's recogniser opened");
    let mic_frames = |stt: u32| {
        (call.run.record.forwarded().iter())
            .filter(|(_, kind, nth, _)| *kind == Kind::Stt && *nth == stt)
            .flat_map(|(_, _, _, tags)| tags)
            .filter(|tag| matches!(tag, Tag::Mic { .. }))
            .count()
    };
    assert!(mic_frames(amd_stream) > 0, "AMD's recogniser heard the person");
    assert_eq!(mic_frames(0), 0, "the session's recogniser heard silence");
}

// The callee (spec §5.2).

/// No participant to listen to within 5 s: the run completes `participant_missing`, and the
/// session's gate is gone. Python: `test_missing_audio_subscription_times_out_and_releases_session`.
#[test]
fn no_participant_within_five_seconds_is_participant_missing() {
    let setup = lk_agents_fuzz::drive::Setup {
        room: Some(lk_agents_fuzz::drive::RoomSetup {
            close_on_disconnect: false,
            delete_room_on_close: false,
        }),
        amd: Some(amd()),
        ..super::listening()
    };
    let call = Call::run(setup, async |call| {
        call.send(Input::AmdEnter {}).await;
        call.advance(4_900).await;
        assert!(call.outcomes().is_empty(), "still waiting at 4.9 s");
        call.advance(200).await;
    });
    let outcome = call.the_outcome();
    assert_eq!(
        (outcome.category, outcome.reason),
        (Stage::Uncertain, AmdReason::ParticipantMissing)
    );
    assert!(outcome.is_error());
    assert!(call.detached_at().is_some(), "the gate detached");
    call.assert_clean();
}

/// A SIP callee still ringing past the 5 s track wait keeps AMD pending, with no deadline; once
/// answered, AMD listens and its overall limit counts from there. Python:
/// `test_sip_answer_gating_and_early_media`.
#[test]
fn a_ringing_sip_callee_is_waited_for_without_a_timeout() {
    let options =
        AmdOptions { timeout: Duration::from_secs(4), ..lk_agents_fuzz::fakes::amd::options() };
    let setup = detecting(AmdSetup { sip_status: Some("ringing".into()), ..with(options) });
    let call = Call::run(setup, async |call| {
        call.enter().await;
        call.advance(8_000).await;
        assert!(call.outcomes().is_empty(), "a call still ringing is waited for");
        assert!(call.run.answer_sip_call(), "the callee answers");
        call.advance(3_900).await;
        assert!(call.outcomes().is_empty(), "the overall limit counts from the answer");
        call.advance(200).await;
    });
    let outcome = call.the_outcome();
    assert_eq!(outcome.reason, AmdReason::Timeout);
    call.assert_clean();
}

/// The participant leaving after AMD listens completes it `participant_disconnected`. Python:
/// `test_without_an_identity_amd_listens_to_the_linked_participant`.
#[test]
fn the_participant_leaving_is_participant_disconnected() {
    let call = Call::run(detecting(amd()), async |call| {
        call.enter().await;
        call.says("hello").await;
        call.classifier_says(AmdSays::Uncertain).await;
        call.advance(100).await;
        call.model_answers(LINE).await;
        call.advance(2_000).await;
        call.send(Input::Room { event: RoomIn::Unlinked }).await;
        call.advance(1_000).await;
    });
    let outcome = call.the_outcome();
    assert_eq!(outcome.reason, AmdReason::ParticipantDisconnected);
    assert!(!outcome.is_error(), "a disconnect is a normal outcome");
    assert_eq!(outcome.last_prediction.map(|p| p.category), Some(Category::Uncertain));
}

// Completion and cancellation (spec §12).

/// `aclose()` with a request in flight: the request is abandoned, its late answer is refused,
/// the gate detaches, and `execute()` returns `cancelled` once, after `Detached`. Python:
/// `test_cancelled_execute_waiter_does_not_cancel_detection`, `_finish(CANCELLED)`.
#[test]
fn closing_the_run_abandons_its_request_and_completes_once_after_detached() {
    let call = Call::run(detecting(amd()), async |call| {
        call.enter().await;
        call.says("hello who is this").await;
        let ask = call.open_ask(AskKind::Classify).expect("a request in flight");
        call.close_amd().await;
        call.advance(100).await;
        assert!(call.open_ask(AskKind::Classify).is_none(), "the close abandoned the request");
        call.send(Input::AmdReply { ask, says: AmdSays::Human }).await;
        call.advance(100).await;
        call.model_answers(LINE).await;
        call.advance(3_000).await;
    });
    let outcome = call.the_outcome();
    assert_eq!((outcome.category, outcome.reason), (Stage::Uncertain, AmdReason::Cancelled));
    assert_eq!(call.completions(), vec![outcome]);
    assert!(call.predictions().is_empty());
    let run = call.amd_run();
    let detached = call.run.record.amd_run().detached(run).expect("the gate detached");
    let returned = call.run.record.amd_run().outcomes(run)[0].0;
    assert!(detached < returned, "execute() returns after the session's Detached");
    call.assert_clean();
}

/// Turn ids are local to each run: a second run numbers its first turn 1. Python:
/// `test_turn_ids_are_local_to_each_amd_run`.
#[test]
fn a_second_run_numbers_its_turns_from_one() {
    let call = Call::run(detecting(amd()), async |call| {
        call.enter().await;
        call.says("hello").await;
        call.classifier_says(AmdSays::Uncertain).await;
        call.advance(100).await;
        call.model_answers(LINE).await;
        call.advance(2_000).await;
        call.close_amd().await;
        call.advance(500).await;
        call.enter().await;
        call.says("hello who is this").await;
        call.classifier_says(AmdSays::Human).await;
        call.advance(100).await;
        call.model_answers(LINE).await;
        call.advance(3_000).await;
    });
    let entered = call.run.record.amd_run().entered();
    assert_eq!(entered.len(), 2, "two runs attached");
    assert_ne!(entered[0].1, entered[1].1, "each run has its own id");
    let first = call.run.record.amd_run().predictions(entered[0].1);
    let second = call.run.record.amd_run().predictions(entered[1].1);
    assert_eq!(first.iter().map(|(_, p)| p.turn_id).collect::<Vec<_>>(), vec![1]);
    assert_eq!(second.iter().map(|(_, p)| p.turn_id).collect::<Vec<_>>(), vec![1]);
    call.assert_clean();
}

// The executor's own account: the steps it journals and the runs it refuses.

/// The run journals its steps in the order they happened: the turn bound, its request issued,
/// the prediction that request decided, then the completion, which follows the session's
/// `Detached`. The DST rules read these steps; a run that journals them out of order proves
/// nothing.
#[test]
fn the_runs_steps_are_journalled_in_order_and_complete_after_detached() {
    let call = Call::run(detecting(amd()), async |call| {
        call.enter().await;
        call.says("hello who is this").await;
        call.classifier_says(AmdSays::Human).await;
        call.advance(100).await;
        call.model_answers(LINE).await;
        call.advance(3_000).await;
    });
    let steps: Vec<_> = call.run.record.amd_run().steps();
    let position = |pick: &dyn Fn(&AmdStep) -> bool| steps.iter().position(|(_, s)| pick(s));
    let bound = position(&|s| matches!(s, AmdStep::TurnBound { turn_id: 1, has_text: true, .. }));
    let issued = position(&|s| matches!(s, AmdStep::RequestIssued { .. }));
    let predicted = position(&|s| {
        matches!(s, AmdStep::Predicted { category: Category::Human, source_req: Some(_), .. })
    });
    let completed = position(&|s| {
        matches!(s, AmdStep::Completed { category: Stage::Human, reason: AmdReason::Finished, .. })
    });
    assert!(
        bound.is_some() && bound < issued && issued < predicted && predicted < completed,
        "{steps:?}"
    );
    let run = call.amd_run();
    let detached = call.run.record.amd_run().detached(run).expect("the gate detached");
    let completed_at = steps[completed.expect("completed")].0;
    assert!(detached < completed_at, "`Completed` after the session's `Detached`");
    call.assert_clean();
}

/// A voicemail greeting's reply carries the voicemail stage's instructions, and `aclose()` after
/// it completes the run in that stage with `cancelled`. Python: `_finish(CANCELLED)` reports the
/// stage it was in.
#[test]
fn a_close_in_the_voicemail_stage_completes_it_cancelled() {
    let call = Call::run(detecting(amd()), async |call| {
        call.enter().await;
        call.says("please leave a message after the tone").await;
        call.classifier_says(AmdSays::Voicemail).await;
        call.advance(2_000).await;
        call.model_answers(LINE).await;
        call.advance(3_000).await;
        call.close_amd().await;
        call.advance(1_000).await;
    });
    let asked = call.made("please leave a message after the tone");
    assert!(asked.iter().any(|q| q.carries(VOICEMAIL)), "{asked:?}");
    let outcome = call.the_outcome();
    assert_eq!((outcome.category, outcome.reason), (Stage::MachineVm, AmdReason::Cancelled));
    call.assert_clean();
}

/// A decision model's `wait` keeps the turn unanswered, with one choice and no menu request.
/// Python: `_authorize_reply` denies a `wait` category.
#[test]
fn a_decision_model_holds_a_wait_turn_and_reads_no_menu() {
    let call = Call::run(detecting(AmdSetup { decision: true, ..amd() }), async |call| {
        call.enter().await;
        call.says("please hold while we connect you").await;
        call.classifier_says(AmdSays::Wait).await;
        call.advance(3_000).await;
        call.model_answers(LINE).await;
        call.advance(2_000).await;
        call.close_amd().await;
        call.advance(1_000).await;
    });
    assert!(call.unanswered("please hold while we connect you"));
    assert_eq!(call.amd_asks_of(AskKind::Choice).len(), 1, "one choice");
    assert!(call.amd_asks_of(AskKind::Menu).is_empty(), "a decision model reads no menu");
    assert_eq!(call.predictions().iter().map(|p| p.category).collect::<Vec<_>>(), [Category::Wait]);
    call.assert_clean();
}

/// A second run entered while one is attached is refused at the gate, and the first still runs
/// and closes. Spec §5.1: one run at a time.
#[test]
fn a_second_run_is_refused_while_one_is_attached() {
    let call = Call::run(detecting(amd()), async |call| {
        call.enter().await;
        call.enter().await;
        call.close_amd().await;
        call.advance(1_000).await;
    });
    let answers = call.run.record.amd_run().enter_answers().to_vec();
    assert_eq!(answers.len(), 2, "{answers:?}");
    assert!(answers[0].1.is_ok() && answers[1].1.is_err(), "{answers:?}");
    let refused = (call.run.record.journal().iter()).any(|(_, s)| {
        matches!(s, lk_agents_voice::trace::Step::Gate(GateStep::AttachRefused { .. }))
    });
    assert!(refused, "the gate refused the second attach");
    assert_eq!(call.the_outcome().reason, AmdReason::Cancelled);
    call.assert_clean();
}
