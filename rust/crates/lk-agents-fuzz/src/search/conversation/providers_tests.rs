use super::*;
use crate::drive::simulate;
use lk_agents_abi::session::{InputCandidate, InputRequestRef, InputRoute};
use lk_agents_abi::{ChatMessage, Role};
use lk_agents_core::session::input_routing;
use std::time::UNIX_EPOCH;

#[test]
fn survey_input_verdicts_cover_valid_ambiguous_and_invalid_provider_outputs() {
    let candidates: Vec<_> = (0..2)
        .map(|index| InputCandidate {
            reference: InputRequestRef::Agent {
                agent_id: "agent".into(),
                request_id: format!("question_{index}"),
            },
            question: "Confirm?".into(),
            presentation: ChatMessage::text(
                format!("presented_{index}"),
                Role::Assistant,
                "Confirm?",
                UNIX_EPOCH,
            ),
            response: None,
        })
        .collect();
    assert_eq!(
        input_routing::parse(&input_verdict(2, 0).0, &candidates).unwrap(),
        InputRoute::NoMatch
    );
    assert_eq!(
        input_routing::parse(&input_verdict(2, 1).0, &candidates).unwrap(),
        InputRoute::Matched(vec![candidates[0].reference.clone()])
    );
    assert_eq!(
        input_routing::parse(&input_verdict(2, 2).0, &candidates).unwrap(),
        InputRoute::Matched(
            candidates.iter().map(|candidate| candidate.reference.clone()).collect()
        )
    );
    assert_eq!(
        input_routing::parse(&input_verdict(2, 3).0, &candidates).unwrap(),
        InputRoute::Unclear
    );
    assert!(input_routing::parse(&input_verdict(2, 4).0, &candidates).is_err());
}

#[test]
fn the_routing_profile_uses_real_inference_to_admit_an_approved_tool_body() {
    let setup = Setup {
        input_routing: true,
        tts: false,
        tools: vec![crate::fakes::tools::APPROVAL_TOOL.into()],
        ..Setup::default()
    };
    assert!(!Setup::default().input_routing, "authored cases opt into a separate routing model");
    simulate(async {
        let mut run = Run::start(&setup, 0, Default::default()).await;
        for input in [
            Input::SetAudioOutput { attach: false },
            Input::Advance { ms: Millis(20) },
            Input::GenerateReply {
                user_input: Some(UserText("Transfer me to the desk.".into())),
                instructions: None,
                allow: None,
            },
            Input::Advance { ms: Millis(100) },
            Input::LlmToolCall {
                chat: Ref::new(0),
                name: crate::fakes::tools::APPROVAL_TOOL.into(),
                args: ToolArgs(r#"{"label":"desk"}"#.into()),
            },
            Input::LlmEnd { chat: Ref::new(0) },
            Input::Advance { ms: Millis(1_000) },
        ] {
            run.apply(&input).await;
        }
        assert!(
            run.record.host_view().tools_run().is_empty(),
            "the first call must wait for approval before running its body"
        );
        run.apply(&Input::GenerateReply {
            user_input: Some(UserText("Yes, transfer me to that desk.".into())),
            instructions: None,
            allow: None,
        })
        .await;
        run.apply(&Input::Advance { ms: Millis(100) }).await;
        let (chat, count) = run
            .record
            .resources(Kind::Chat)
            .find_map(|chat| match chat.detail() {
                Detail::Chat { tools, input_candidates: Some(count), .. } => {
                    assert_eq!(tools, &[input_routing::TOOL.to_string()]);
                    Some((chat.nth(), *count))
                }
                _ => None,
            })
            .expect("the separate fake must observe the real routing request");
        assert_eq!(count, 1, "the fixture records only the actually offered occurrence count");
        let verdict = Input::LlmToolCall {
            chat: Ref::new(chat),
            name: input_routing::TOOL.into(),
            args: input_verdict(count, 1),
        };
        run.apply(&verdict).await;
        run.apply(&Input::LlmEnd { chat: Ref::new(chat) }).await;
        run.apply(&Input::Advance { ms: Millis(100) }).await;
        let frontend = run
            .record
            .resources(Kind::Chat)
            .filter(|opened| opened.nth() > chat)
            .find(|opened| matches!(opened.detail(), Detail::Chat { input_candidates: None, .. }))
            .expect("accepted routing must release the frontend model")
            .nth();
        for input in [
            Input::LlmToolCall {
                chat: Ref::new(frontend),
                name: crate::fakes::tools::APPROVAL_TOOL.into(),
                args: ToolArgs(r#"{"label":"desk"}"#.into()),
            },
            Input::LlmEnd { chat: Ref::new(frontend) },
            Input::Advance { ms: Millis(100) },
        ] {
            run.apply(&input).await;
        }
        assert_eq!(
            run.record.host_view().tools_run().len(),
            1,
            "production correlation and approval admission must release exactly one body"
        );
        for input in [
            Input::ToolAnswered {
                tool: Ref::new(0),
                answer: ToolAnswer::Return,
                text: AgentText(String::new()),
            },
            Input::Advance { ms: Millis(100) },
            Input::Close { drain: false },
            Input::Advance { ms: Millis(5_000) },
            Input::AwaitExit {},
        ] {
            run.apply(&input).await;
        }
        let (failures, _) = crate::check::check_all(&run.record);
        assert!(failures.is_empty(), "{failures:?}");
    });
}
