mod input_required_preserves_model_arguments {
    //! A human message cannot silently replace the arguments of a later tool call.
    use lk_agents_abi::ChatItem;
    use lk_agents_fuzz::drive::{Run, Setup, simulate};
    use lk_agents_fuzz::input::{
        AgentText, Input, Millis, Ref, ToolAnswer, ToolArgs, TurnMode, UserText,
    };
    use lk_agents_voice::events::SessionEvent;

    // Catches: legacy named-answer injection rewriting the first string argument merely because
    // the person spoke while an earlier question was open.
    #[test]
    fn a_pending_question_does_not_rewrite_a_later_tools_arguments() {
        let setup = Setup {
            vad: true,
            stt: true,
            turn: TurnMode::Stt,
            tools: vec!["lookup".into()],
            ..Setup::default()
        };
        let call = |chat| Input::LlmToolCall {
            chat: Ref::new(chat),
            name: "lookup".into(),
            args: ToolArgs(r#"{"label":"guess"}"#.into()),
        };
        let inputs = [
            Input::GenerateReply { user_input: None, instructions: None, allow: None },
            Input::Advance { ms: Millis(100) },
            call(0),
            Input::LlmEnd { chat: Ref::new(0) },
            Input::Advance { ms: Millis(100) },
            Input::ToolAnswered {
                tool: Ref::new(0),
                answer: ToolAnswer::AskVerbatim,
                text: AgentText("Which day?".into()),
            },
            Input::Advance { ms: Millis(3_000) },
            Input::VadStart { vad: Ref::new(0), speech_ms: Millis(300) },
            Input::SttStart { stt: Ref::new(0) },
            Input::Advance { ms: Millis(300) },
            Input::SttFinal {
                utterance: Ref::new(0),
                text: UserText("Monday".into()),
                speaker: None,
            },
            Input::SttEnd { utterance: Ref::new(0) },
            Input::VadEnd { vad: Ref::new(0), silence_ms: Millis(500) },
            Input::Advance { ms: Millis(3_000) },
            // The next turn's explicit call retains the arguments the model supplied.
            call(1),
            Input::LlmEnd { chat: Ref::new(1) },
            Input::Advance { ms: Millis(100) },
            Input::ToolAnswered {
                tool: Ref::new(1),
                answer: ToolAnswer::Value,
                text: AgentText("Booked.".into()),
            },
            Input::Advance { ms: Millis(1_000) },
        ];
        let run = simulate(async {
            let mut run = Run::start(&setup, 0, Default::default()).await;
            for input in &inputs {
                run.apply(input).await;
            }
            run
        });
        let calls: Vec<String> = run
            .record
            .host_view()
            .events()
            .into_iter()
            .filter_map(|(_, ev)| match ev {
                SessionEvent::ConversationItemAdded { item, .. } => match item.as_ref() {
                    ChatItem::FunctionCall(c) => Some(c.arguments.clone()),
                    _ => None,
                },
                _ => None,
            })
            .collect();
        assert_eq!(calls, [r#"{"label":"guess"}"#, r#"{"label":"guess"}"#], "calls: {calls:?}");
    }
}

mod ordinary_completed_replies {
    //! An ordinary tool's `Completed(value)` is its call's output, and the agent replies to it.
    use lk_agents_abi::ChatItem;
    use lk_agents_fuzz::drive::{Run, Setup, simulate};
    use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref, ToolAnswer, ToolArgs};
    use lk_agents_fuzz::record::Kind;
    use lk_agents_voice::events::SessionEvent;

    // Catches: a `Completed` from a tool outside any VoiceTask taken as a task's completion, its
    // output asking for no reply and the step ending there: the agent went silent. The fake host
    // hands a value with every completion, so the core alone decides.
    #[test]
    fn an_ordinary_completion_is_replied_to() {
        let setup = Setup { tools: vec!["lookup".into()], ..Setup::default() };
        let inputs = [
            Input::GenerateReply { user_input: None, instructions: None, allow: None },
            Input::Advance { ms: Millis(100) },
            Input::LlmToolCall {
                chat: Ref::new(0),
                name: "lookup".into(),
                args: ToolArgs(r#"{"label":"order"}"#.into()),
            },
            Input::LlmEnd { chat: Ref::new(0) },
            Input::Advance { ms: Millis(100) },
            Input::ToolAnswered {
                tool: Ref::new(0),
                answer: ToolAnswer::Completed,
                text: AgentText("booked".into()),
            },
            Input::Advance { ms: Millis(1_000) },
        ];
        let run = simulate(async {
            let mut run = Run::start(&setup, 0, Default::default()).await;
            for input in &inputs {
                run.apply(input).await;
            }
            run
        });
        let outputs: Vec<(String, bool)> = run
            .record
            .host_view()
            .events()
            .into_iter()
            .filter_map(|(_, ev)| match ev {
                SessionEvent::ConversationItemAdded { item, .. } => match item.as_ref() {
                    ChatItem::FunctionCallOutput(o) => Some((o.output.clone(), o.reply_required)),
                    _ => None,
                },
                _ => None,
            })
            .collect();
        assert_eq!(outputs, [("booked".to_owned(), true)], "outputs: {outputs:?}");
        assert_eq!(run.record.resources(Kind::Chat).count(), 2, "the follow-up was never asked");
    }
}

mod stored_response_keeps_the_read_back {
    //! Human words and the original approval question are distinct stored records.
    use lk_agents_abi::ChatItem;
    use lk_agents_fuzz::drive::{Run, Setup, simulate};
    use lk_agents_fuzz::input::{Input, Millis, Ref, ToolArgs, TurnMode, UserText};
    use lk_agents_voice::events::SessionEvent;

    /// The fixture tool the runs declare with `needs_approval=True`.
    const APPROVAL_TOOL: &str = "transfer";

    // Catches: a preemptive or final transcript replacing a waiting tool's original read-back.
    #[test]
    fn the_human_response_does_not_replace_the_read_back() {
        // A store that keeps its lease for the whole call.
        let setup = Setup {
            tools: vec![APPROVAL_TOOL.into()],
            stt: true,
            vad: true,
            preemptive: true,
            turn: TurnMode::Stt,
            storage: Some(lk_agents_fuzz::drive::StorageSetup::default()),
            ..Setup::default()
        };
        let answer = || UserText("yes please".into());
        let inputs = [
            Input::GenerateReply {
                user_input: Some(UserText("transfer me".into())),
                instructions: None,
                allow: None,
            },
            Input::Advance { ms: Millis(20) },
            Input::LlmToolCall {
                chat: Ref::new(0),
                name: APPROVAL_TOOL.into(),
                args: ToolArgs("{}".into()),
            },
            Input::LlmEnd { chat: Ref::new(0) },
            Input::Advance { ms: Millis(5_000) },
            // The person answers; the reply to it is planned from their preflight words.
            Input::VadStart { vad: Ref::new(0), speech_ms: Millis(300) },
            Input::SttStart { stt: Ref::new(0) },
            Input::Advance { ms: Millis(600) },
            Input::VadEnd { vad: Ref::new(0), silence_ms: Millis(300) },
            Input::SttPreflight { utterance: Ref::new(0), text: answer(), speaker: None },
            Input::Advance { ms: Millis(50) },
            Input::SttFinal { utterance: Ref::new(0), text: answer(), speaker: None },
            Input::SttEnd { utterance: Ref::new(0) },
            Input::Advance { ms: Millis(5_000) },
        ];
        let run = simulate(async {
            let mut run = Run::start(&setup, 0, Default::default()).await;
            for input in &inputs {
                run.apply(input).await;
            }
            run
        });
        let outputs: Vec<String> = run
            .record
            .host_view()
            .events()
            .into_iter()
            .filter_map(|(_, ev)| match ev {
                SessionEvent::ConversationItemAdded { item, .. } => match item.as_ref() {
                    ChatItem::FunctionCallOutput(o) => Some(o.output.clone()),
                    _ => None,
                },
                _ => None,
            })
            .collect();
        let asked = outputs.iter().any(|o| o.starts_with("waiting for the person's answer"));
        assert!(asked, "the scenario: the call waits on its read-back: {outputs:?}");
        assert!(
            outputs.iter().all(|output| output.starts_with("waiting for the person's answer")),
            "human words must not replace the approval outcome: {outputs:?}"
        );
        let heard = run.record.host_view().events().into_iter().any(|(_, event)| {
            matches!(event, SessionEvent::ConversationItemAdded { item, .. }
                if matches!(&**item, ChatItem::Message(message)
                    if message.role == lk_agents_abi::Role::User
                        && message.raw_text_content().as_deref() == Some("yes please")))
        });
        assert!(heard, "the canonical human response was lost");
    }
}

mod cut_question {
    //! A later human turn cannot consume an unheard question without exact response admission.
    use lk_agents_abi::ChatItem;
    use lk_agents_abi::llm::tool_executor::WAITING_PREFIX;
    use lk_agents_fuzz::drive::{Run, Setup, simulate};
    use lk_agents_fuzz::input::{
        AgentText, Input, Millis, Ref, ToolAnswer, ToolArgs, TurnMode, UserText,
    };
    use lk_agents_voice::events::SessionEvent;

    // Catches: a step whose speech was cut keeping an implicit answer claim. No routing model
    // is configured here, so neither pending source outcome may be replaced by the next turn.
    #[test]
    fn an_unmatched_turn_preserves_both_questions() {
        let setup = Setup {
            vad: true,
            stt: true,
            turn: TurnMode::Stt,
            tools: vec!["lookup".into(), "reserve".into()],
            ..Setup::default()
        };
        let call = |chat, name: &str| Input::LlmToolCall {
            chat: Ref::new(chat),
            name: name.into(),
            args: ToolArgs("{}".into()),
        };
        let ask = |tool, question: &str| Input::ToolAnswered {
            tool: Ref::new(tool),
            answer: ToolAnswer::AskVerbatim,
            text: AgentText(question.into()),
        };
        let inputs = [
            Input::GenerateReply { user_input: None, instructions: None, allow: None },
            Input::Advance { ms: Millis(100) },
            call(0, "lookup"),
            Input::LlmEnd { chat: Ref::new(0) },
            Input::Advance { ms: Millis(100) },
            // The reply is cut while `lookup` runs; its question comes after and is never put.
            Input::Interrupt { force: false, wait: false },
            Input::Advance { ms: Millis(100) },
            ask(0, "Which day?"),
            Input::Advance { ms: Millis(1_000) },
            Input::GenerateReply { user_input: None, instructions: None, allow: None },
            Input::Advance { ms: Millis(100) },
            call(1, "reserve"),
            Input::LlmEnd { chat: Ref::new(1) },
            Input::Advance { ms: Millis(100) },
            ask(1, "Which city?"),
            Input::Advance { ms: Millis(3_000) },
            Input::VadStart { vad: Ref::new(0), speech_ms: Millis(300) },
            Input::SttStart { stt: Ref::new(0) },
            Input::Advance { ms: Millis(300) },
            Input::SttFinal {
                utterance: Ref::new(0),
                text: UserText("Paris".into()),
                speaker: None,
            },
            Input::SttEnd { utterance: Ref::new(0) },
            Input::VadEnd { vad: Ref::new(0), silence_ms: Millis(500) },
            Input::Advance { ms: Millis(3_000) },
        ];
        let run = simulate(async {
            let mut run = Run::start(&setup, 0, Default::default()).await;
            for input in &inputs {
                run.apply(input).await;
            }
            run
        });
        let outputs: Vec<(String, String)> = run
            .record
            .host_view()
            .events()
            .into_iter()
            .filter_map(|(_, ev)| match ev {
                SessionEvent::ConversationItemAdded { item, .. } => match item.as_ref() {
                    ChatItem::FunctionCallOutput(o) => Some((o.name.clone(), o.output.clone())),
                    _ => None,
                },
                _ => None,
            })
            .collect();
        let waiting = |q: &str| format!("{WAITING_PREFIX}{q}");
        assert_eq!(
            outputs,
            [
                ("lookup".to_owned(), waiting("Which day?")),
                ("reserve".to_owned(), waiting("Which city?")),
            ],
            "human input cannot implicitly resolve either source question"
        );
    }
}

mod questions_in_recorded_order {
    //! Recorded question order cannot implicitly select a response destination.
    use lk_agents_abi::ChatItem;
    use lk_agents_abi::llm::tool_executor::WAITING_PREFIX;
    use lk_agents_fuzz::drive::{Run, Setup, simulate};
    use lk_agents_fuzz::input::{
        AgentText, Input, Millis, Ref, ToolAnswer, ToolArgs, TurnMode, UserText,
    };
    use lk_agents_voice::events::SessionEvent;

    // Catches: an output stamped from its run's anchor reordering recorded questions, or the
    // next human message implicitly consuming whichever question that order happens to select.
    #[test]
    fn recorded_question_order_does_not_choose_an_unmatched_responses_destination() {
        let setup = Setup {
            vad: true,
            stt: true,
            turn: TurnMode::Stt,
            tools: vec!["lookup".into(), "reserve".into()],
            ..Setup::default()
        };
        let call = |chat, name: &str| Input::LlmToolCall {
            chat: Ref::new(chat),
            name: name.into(),
            args: ToolArgs("{}".into()),
        };
        let ask = |tool, question: &str| Input::ToolAnswered {
            tool: Ref::new(tool),
            answer: ToolAnswer::AskVerbatim,
            text: AgentText(question.into()),
        };
        let inputs = [
            // The first reply's `reserve` runs on once its step gave the floor back.
            Input::GenerateReply { user_input: None, instructions: None, allow: None },
            Input::Advance { ms: Millis(100) },
            call(0, "reserve"),
            Input::LlmEnd { chat: Ref::new(0) },
            Input::Advance { ms: Millis(100) },
            Input::GenerateReply { user_input: None, instructions: None, allow: None },
            Input::Advance { ms: Millis(100) },
            // In one instant: the second reply calls `lookup`, which asks, then `reserve` asks.
            call(1, "lookup"),
            Input::LlmEnd { chat: Ref::new(1) },
            Input::Advance { ms: Millis(0) },
            ask(1, "Which day?"),
            Input::Advance { ms: Millis(0) },
            ask(0, "Which city?"),
            Input::Advance { ms: Millis(3_000) },
            Input::VadStart { vad: Ref::new(0), speech_ms: Millis(300) },
            Input::SttStart { stt: Ref::new(0) },
            Input::Advance { ms: Millis(300) },
            Input::SttFinal {
                utterance: Ref::new(0),
                text: UserText("Monday".into()),
                speaker: None,
            },
            Input::SttEnd { utterance: Ref::new(0) },
            Input::VadEnd { vad: Ref::new(0), silence_ms: Millis(500) },
            Input::Advance { ms: Millis(3_000) },
        ];
        let run = simulate(async {
            let mut run = Run::start(&setup, 0, Default::default()).await;
            for input in &inputs {
                run.apply(input).await;
            }
            run
        });
        let outputs: Vec<(String, String)> = run
            .record
            .host_view()
            .events()
            .into_iter()
            .filter_map(|(_, ev)| match ev {
                SessionEvent::ConversationItemAdded { item, .. } => match item.as_ref() {
                    ChatItem::FunctionCallOutput(o) => Some((o.name.clone(), o.output.clone())),
                    _ => None,
                },
                _ => None,
            })
            .collect();
        let waiting = |q: &str| format!("{WAITING_PREFIX}{q}");
        assert_eq!(
            outputs,
            [
                ("lookup".to_owned(), waiting("Which day?")),
                ("reserve".to_owned(), waiting("Which city?")),
            ],
            "unmatched human input preserves the recorded questions and their order"
        );
    }
}
