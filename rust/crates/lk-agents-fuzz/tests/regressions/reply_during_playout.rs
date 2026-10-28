//! A reply asked for while the previous one is still playing makes its own model request.
use lk_agents_fuzz::drive::{Run, Setup, simulate};
use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref};
use lk_agents_fuzz::record::Kind;

// Catches: since c43d27219, `generate_reply` while the previous reply still spoke interrupted it
// and never issued its own LLM request, its handle never settling (coordinator report, real
// room). A reply asked for during a long playout must reach its model.
#[test]
fn a_reply_asked_during_playout_reaches_its_model() {
    let setup = Setup::default();
    let inputs = [
        Input::GenerateReply { user_input: None, instructions: None, allow: None },
        Input::Advance { ms: Millis(100) },
        Input::LlmText { chat: Ref::new(0), text: AgentText("A long first answer that keeps playing for a while.".into()) },
        Input::LlmEnd { chat: Ref::new(0) },
        Input::Advance { ms: Millis(100) },
        Input::PlayoutStarted { segment: Ref::new(0) },
        Input::Advance { ms: Millis(500) },
        Input::GenerateReply { user_input: None, instructions: Some(AgentText("say something else".into())), allow: None },
        Input::Advance { ms: Millis(2_000) },
    ];
    let run = simulate(async {
        let mut run = Run::start(&setup, 0, Default::default()).await;
        for input in &inputs {
            run.apply(input).await;
        }
        run
    });
    let chats = run.record.resources(Kind::Chat).count();
    assert_eq!(chats, 2, "the second reply never asked its model");
}
