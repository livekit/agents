//! Host-handler latency cannot restore an interrupted step's question or reply claim.

use super::*;
use crate::agent_activity::scheduling::SpeechKind;
use crate::speech_handle::SpeechState;
use crate::tool_executor::{ToolOutcome, answer};
use crate::trace::InterruptCause;
use lk_agents_abi::{FunctionCall, FunctionCallOutput};
use std::time::UNIX_EPOCH;

#[test]
fn interrupt_while_tool_handler_waits_suppresses_the_old_question_and_claim() {
    let mut driver = crate::agent_session::tests::driver();
    let speech = SpeechId(41);
    let mut out = driver.out();
    assert!(driver.schedule_speech(
        Ts::ZERO,
        speech,
        SpeechState::new(true),
        SpeechKind::GenerateReply,
        true,
        true,
        &mut out,
    ));
    let call = FunctionCall {
        id: "flight-item".into(),
        call_id: "flight-call".into(),
        name: "book_flight".into(),
        arguments: "{}".into(),
        created_at: UNIX_EPOCH,
        extra: Default::default(),
        group_id: None,
    };
    let output =
        FunctionCallOutput::new("flight-output", "flight-call", "Which date?", false, UNIX_EPOCH);
    let execution = answer(
        &call,
        ToolOutcome::InputRequired {
            question: "Which date?".into(),
            verbatim: true,
            tool: Some("book_flight".into()),
        }
        .into(),
        UNIX_EPOCH,
    );
    let hook = HookId(7);
    driver.activity.tools.handling.insert(
        hook,
        Handling {
            speech,
            step: 1,
            // Both executor reads happened before the host handler started awaiting.
            end: StepEnd {
                executions: vec![execution],
                interrupted_before_tools: false,
                interrupted: false,
                forwarded_text_empty: true,
                max_tool_steps: 5,
                user_metrics: None,
            },
            items: vec![call.clone().into(), output.clone().into()],
            terminal_call: Some(call.call_id.clone()),
            event: ToolsExecutedEvent {
                function_calls: vec![call],
                function_call_outputs: vec![output],
                handoff_required: false,
            },
        },
    );
    // The person interrupts while function_tools_executed is still running at the host.
    let _ =
        driver.interrupt_speech(Ts::ZERO, speech, InterruptCause::Host, true, &mut out).unwrap();
    assert!(driver.activity.speech_scheduler.speech(speech).unwrap().interrupted());
    driver.on_tools_executed_handled(hook, None, &mut out);
    assert!(driver.activity.tools.claim.is_none(), "the old question cannot claim a new turn");
    assert!(!driver.session.runs.outputs.contains_key(&speech), "no old question was delivered");
    assert_eq!(
        driver.activity.speech_scheduler.speech_ids().count(),
        1,
        "no question speech was created"
    );
    let output = driver
        .session
        .chat_ctx
        .items()
        .find_map(|item| match item {
            ChatItem::FunctionCallOutput(output) => Some(output),
            _ => None,
        })
        .expect("the result remains recorded even though its question is suppressed");
    assert!(!output.reply_required);
}
