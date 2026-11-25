//! What an allowed reply carries (proposal "Reply payload"): stage instructions, the DTMF tool,
//! preemptive adoption and corrections.

use super::{Call, listening};
use lk_agents_fuzz::drive::Setup;
use lk_agents_fuzz::fakes::amd::SEND_DTMF_TOOL;
use lk_agents_fuzz::fakes::amd::STAGE;
use lk_agents_fuzz::input::{
    AgentText, AmdVerdict, Input, Millis, Ref, ToolAnswer, ToolArgs, TurnMode, UserText,
};
use lk_agents_fuzz::record::Kind;

const LINE: &str = "Hello, this is the clinic.";

/// Stage instructions reach the reply and its tool follow-up (spec §10.2), never the record
/// (invariant 8), and expire with the reply. Python:
/// `test_stage_instructions_expire_after_first_human_reply`.
#[test]
fn stage_instructions_reach_the_reply_and_its_follow_up_and_are_never_saved() {
    let setup = Setup { tools: vec!["lookup".into()], ..listening() };
    let call = Call::run(setup, async |call| {
        call.attach(None).await;
        call.says("you have reached the screener").await;
        call.decide(AmdVerdict::Stage).await;
        call.advance(50).await;
        // The reply's request, not the first one: the preemptive generation opened chat 0 and
        // the staged decision discarded it (proposal "Preemptive generation").
        let chat = Ref::new(*call.run.record.live().of(Kind::Chat).last().expect("the reply"));
        call.send(Input::LlmToolCall { chat, name: "lookup".into(), args: ToolArgs("{}".into()) })
            .await;
        call.send(Input::LlmEnd { chat }).await;
        call.advance(50).await;
        // A value, so the model is asked again: `Return` is a tool returning nothing, which
        // ends the reply with no follow-up (`ToolOutcome::Nothing`).
        call.send(Input::ToolAnswered {
            tool: Ref::new(0),
            answer: ToolAnswer::Value,
            text: AgentText("found".into()),
        })
        .await;
        call.advance(100).await;
        call.model_answers(LINE).await;
        call.advance(3_000).await;
        call.says("go ahead").await;
        call.decide(AmdVerdict::Allow).await;
        call.advance(50).await;
        call.model_answers(LINE).await;
        call.advance(3_000).await;
    });
    let staged = call.made("you have reached the screener");
    assert_eq!(staged.len(), 2, "the reply and its tool follow-up: {staged:?}");
    assert!(staged.iter().all(|q| q.carries(STAGE)), "both carry the stage");
    let next = call.heard_requests("go ahead");
    assert_eq!(next.len(), 1);
    assert!(call.asked("go ahead").iter().all(|q| !q.carries(STAGE)), "no inheritance");
    assert!(!call.recorded_anywhere(STAGE), "stage instructions are never recorded");
    assert!(call.failures().is_empty(), "{:?}", call.failures());
}

/// The DTMF tool is offered to the IVR reply only, once. Python:
/// `test_menu_is_observability_only_and_dtmf_tool_is_temporary`.
#[test]
fn an_ivr_reply_offers_the_dtmf_tool_once() {
    let call = Call::run(listening(), async |call| {
        call.attach(None).await;
        call.says("press one for appointments").await;
        call.decide(AmdVerdict::Ivr).await;
        call.advance(50).await;
        call.model_answers(LINE).await;
        call.advance(4_000).await;
        call.says("hello this is Sam").await;
        call.decide(AmdVerdict::Allow).await;
        call.advance(50).await;
        call.model_answers(LINE).await;
        call.advance(3_000).await;
    });
    let ivr = call.made("press one for appointments");
    assert_eq!(ivr.len(), 1);
    assert_eq!(ivr[0].offers(SEND_DTMF_TOOL), 1);
    assert_eq!(call.heard_requests("hello this is Sam").len(), 1);
    let human = call.asked("hello this is Sam");
    assert!(human.iter().all(|q| q.offers(SEND_DTMF_TOOL) == 0), "{human:?}");
    let preemptive = call.asked("press one for appointments");
    assert!(
        preemptive
            .iter()
            .filter(|q| !ivr.iter().any(|m| m.chat == q.chat))
            .all(|q| q.offers(SEND_DTMF_TOOL) == 0),
        "a preemptive generation never gets AMD's DTMF tool"
    );
    assert!(call.failures().is_empty(), "{:?}", call.failures());
}

/// Proposal story: an agent that defines its own `send_dtmf_events`, across IVR entry and exit:
/// its own tool is kept, never doubled.
#[test]
fn an_agents_own_dtmf_tool_is_kept() {
    let setup = Setup { tools: vec![SEND_DTMF_TOOL.into()], ..listening() };
    let call = Call::run(setup, async |call| {
        call.attach(None).await;
        call.says("press one for appointments").await;
        call.advance(1_000).await;
        assert!(call.unanswered("press one for appointments"), "held for its decision");
        call.decide(AmdVerdict::Ivr).await;
        call.advance(50).await;
        call.model_answers(LINE).await;
        call.advance(4_000).await;
        call.detach(AmdVerdict::Allow, false).await;
        call.says("hello this is Sam").await;
        call.advance(50).await;
        call.model_answers(LINE).await;
        call.advance(3_000).await;
    });
    let ivr = call.made("press one for appointments");
    assert_eq!(ivr.len(), 1);
    assert_eq!(ivr[0].offers(SEND_DTMF_TOOL), 1, "the agent's tool, once");
    assert!(ivr[0].carries(STAGE));
    let human = call.asked("hello this is Sam");
    assert!(
        !human.is_empty() && human.iter().all(|q| q.offers(SEND_DTMF_TOOL) == 1),
        "still the agent's"
    );
    assert!(call.failures().is_empty(), "{:?}", call.failures());
}

fn preemptive() -> Setup {
    Setup { stt: true, preemptive: true, turn: TurnMode::Stt, ..Setup::default() }
}

/// The person's words through a preflight: the preemptive generation starts on the preflight.
async fn says_with_preflight(call: &mut Call, words: &str) {
    call.send(Input::SttStart { stt: Ref::new(0) }).await;
    call.send(Input::SttPreflight {
        utterance: Ref::new(0),
        text: UserText(words.into()),
        speaker: None,
    })
    .await;
    call.advance(100).await;
    call.send(Input::SttFinal {
        utterance: Ref::new(0),
        text: UserText(words.into()),
        speaker: None,
    })
    .await;
    call.send(Input::SttEnd { utterance: Ref::new(0) }).await;
    call.advance(1_000).await;
}

/// Proposal story: a direct human answer adopts its preemptive generation, and a screening reply
/// discards it. Python: `test_preemptive_generation_commits_only_the_accepted_reply`.
#[test]
fn a_plain_allow_adopts_the_preemptive_generation_and_a_staged_one_discards_it() {
    for verdict in [AmdVerdict::Allow, AmdVerdict::Stage] {
        let call = Call::run(preemptive(), async |call| {
            call.attach(None).await;
            says_with_preflight(call, "hello this is Sam").await;
            assert!(
                call.heard_reply("hello this is Sam").is_none(),
                "{verdict:?}: nothing is heard before the decision"
            );
            call.decide(verdict).await;
            call.advance(50).await;
            call.model_answers(LINE).await;
            call.advance(3_000).await;
        });
        let asked = call.asked("hello this is Sam");
        let heard: Vec<_> = asked.iter().filter(|q| call.heard(q.speech).is_some()).collect();
        assert_eq!(heard.len(), 1, "{verdict:?}: one reply heard: {asked:?}");
        match verdict {
            AmdVerdict::Allow => {
                assert_eq!(asked.len(), 1, "adopted, not generated again");
                assert!(!heard[0].carries(STAGE));
            }
            _ => {
                assert_eq!(asked.len(), 2, "discarded and generated again");
                assert!(heard[0].carries(STAGE), "the heard reply carries the stage");
            }
        }
        assert!(call.failures().is_empty(), "{verdict:?}: {:?}", call.failures());
    }
}

/// A recogniser and manual turns, as `turns/correction.rs` sets them up.
fn manual() -> Setup {
    Setup { stt: true, turn: TurnMode::Manual, ..Setup::default() }
}

/// "book a table" committed as an interim; its final reads "book a cable". Returns when the
/// commit was asked.
// The commit lands after the endpointing delay, so this waits a second past `CommitTurn`, as
// `turns/correction.rs` does: a final before the commit replaces the words instead of
// correcting a committed turn.
async fn committed_as_interim(call: &mut Call) -> u64 {
    call.send(Input::SttStart { stt: Ref::new(0) }).await;
    call.send(Input::SttInterim {
        utterance: Ref::new(0),
        text: UserText("book a table".into()),
        speaker: None,
    })
    .await;
    call.advance(100).await;
    let asked = call.now_ms();
    call.send(Input::CommitTurn { skip_reply: false }).await;
    call.advance(1_000).await;
    asked
}

async fn corrected(call: &mut Call) {
    call.send(Input::SttFinal {
        utterance: Ref::new(0),
        text: UserText("book a cable".into()),
        speaker: None,
    })
    .await;
    call.advance(50).await;
}

/// Proposal story: a correction before and after the reply exists. Before, the reply answers the
/// corrected words; after, while unheard, the reply is made again from them, under the same
/// decision.
#[test]
fn a_correction_before_and_after_the_reply_exists() {
    for before in [true, false] {
        let call = Call::run(manual(), async |call| {
            call.attach(None).await;
            committed_as_interim(call).await;
            if before {
                corrected(call).await;
            }
            call.decide(AmdVerdict::Stage).await;
            call.advance(50).await;
            if !before {
                corrected(call).await;
            }
            call.model_answers(LINE).await;
            call.advance(3_000).await;
        });
        assert_eq!(call.heard_reply("book a table"), None, "before={before}");
        let fixed = call.made("book a cable");
        assert_eq!(fixed.len(), 1, "before={before}: {fixed:?}");
        assert!(
            fixed[0].carries(STAGE),
            "before={before}: the decision applies to the replacement"
        );
        assert!(call.heard(fixed[0].speech).is_some(), "before={before}");
        assert!(call.failures().is_empty(), "before={before}: {:?}", call.failures());
    }
}

/// Proposal story: a correction while the voicemail reply is held: the replacement keeps the
/// hold, and is heard only after the silence.
#[test]
fn a_correction_while_the_voicemail_reply_is_held() {
    let mut committed = 0;
    let call = Call::run(manual(), async |call| {
        call.attach(None).await;
        committed = committed_as_interim(call).await;
        call.decide(AmdVerdict::Silence).await;
        call.advance(50).await;
        corrected(call).await;
        call.model_answers(LINE).await;
        call.advance(Millis(4_000).0).await;
    });
    let fixed = call.made("book a cable");
    assert_eq!(fixed.len(), 1);
    assert!(fixed[0].carries(STAGE));
    let heard = call.heard(fixed[0].speech).expect("heard after the hold");
    // No detector on this call: the silence is timed from the commit, the latest boundary
    // (`detector.py:708`).
    assert!(heard >= committed + 1_500 - 200, "heard at {heard} ms, committed near {committed} ms");
    assert_eq!(call.heard_reply("book a table"), None);
    assert!(call.failures().is_empty(), "{:?}", call.failures());
}
