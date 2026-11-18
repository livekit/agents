use super::*;
use crate::agent_session::effects::VoiceEffect;
use crate::agent_session::machine::activity::reply_run::ReplyRun;
use crate::agent_session::machine::activity::scheduling::SpeechKind;
use crate::agent_session::machine::activity::speech::SpeechSpec;
use crate::generation::{PlanSpec, ReplySource};
use crate::tool_executor::{AgentFrame, MadeCall};
use lk_agents_abi::agent::{
    AgentCallbacks, AgentDefinition, FunctionToolInfo, OutputSpec, Retries,
};
use lk_agents_abi::llm::provider_format::tools::{DuplicateMode, DuplicateScope};
use lk_agents_abi::session::{InputCandidate, InputRequestRef};
use lk_agents_abi::{FunctionCall, HostRef, ToolSpec, Ts, WallAnchor};
use lk_agents_core::agent_run::{AgentRun, AgentRunEffect, AgentRunEvent, Opening, RunInput};
use std::sync::Arc;
use std::time::UNIX_EPOCH;

struct Frame(AgentDefinition);

impl AgentFrame for Frame {
    fn definition(&self) -> &AgentDefinition {
        &self.0
    }
    fn host(&self) -> &Arc<dyn AgentCallbacks> {
        unreachable!("the unhosted test call has no callback")
    }
    fn host_context(&self) -> &HostRef {
        unreachable!("the unhosted test call has no context")
    }
}

fn candidate(id: &str) -> InputCandidate {
    InputCandidate {
        reference: InputRequestRef::RemoteTask {
            remote_id: "travel".into(),
            task: lk_agents_abi::delegation::remote::RemoteTaskRef {
                context_id: "trip".into(),
                task_id: "flight".into(),
            },
            request_id: id.into(),
        },
        question: "Which day?".into(),
        presentation: ChatMessage::text("heard", Role::Assistant, "Which day?", UNIX_EPOCH),
        response: None,
    }
}

#[test]
fn realtime_calls_wait_for_exact_capture_and_response_admission_in_both_transcript_orders() {
    for order in [Some(false), Some(true), None] {
        let mut driver = crate::agent_session::tests::driver();
        let speech = driver.start_speech(
            Ts::ZERO,
            SpeechSpec::new(SpeechKind::Say, ReplySource::Text("response".into()), true),
        );
        driver.activity.speech_scheduler.speech_mut(speech).unwrap().authorize_generation();
        let spec = ToolSpec {
            name: "travel".into(),
            description: "Travel expert".into(),
            json_schema: serde_json::json!({"type":"object"}),
            flags: Default::default(),
            provider_tool: None,
        };
        let definition = AgentDefinition {
            name: "desk".into(),
            instructions: String::new(),
            dynamic_instructions: false,
            tools: vec![FunctionToolInfo {
                spec: spec.clone(),
                raw_schema: true,
                needs_approval: None,
                on_duplicate: DuplicateMode::Allow,
                duplicate_scope: DuplicateScope::Name,
                retries: None,
                remote: Some("https://travel.test".into()),
                local_agent: None,
            }],
            output: OutputSpec::Text { validated: false },
            retries: Retries::default(),
        };
        let mut run = AgentRun::new(
            definition.clone(),
            RunInput {
                invocation_id: "reply".into(),
                id_seed: 0,
                opening: Opening::Coordinated { approval_answers: Default::default() },
                history: vec![],
                delegated: false,
                mode: lk_agents_core::tool_outcome::Mode::Voice,
                llm: None,
                wall: WallAnchor { wall: 0.0, at: Ts::ZERO },
                cleanup_timeout: lk_agents_abi::HOST_TEARDOWN_TIMEOUT,
            },
        )
        .unwrap();
        let request = run
            .handle(AgentRunEvent::Start, Ts::ZERO)
            .into_iter()
            .find_map(|effect| match effect {
                AgentRunEffect::PlanRequest { request, .. } => Some(request),
                _ => None,
            })
            .unwrap();
        let mut reply = ReplyRun::new(
            run,
            Some(Arc::new(Frame(definition))),
            PlanSpec::default(),
            true,
            Some(request),
        );
        reply.input_gate = ReplyInputGate::Capturing { request, after: AfterCapture::Human };
        driver.activity.tools.replies.insert(speech, reply);
        let call = FunctionCall {
            id: "call".into(),
            call_id: "wire".into(),
            arguments: "{}".into(),
            name: "travel".into(),
            created_at: UNIX_EPOCH,
            extra: Default::default(),
            group_id: None,
        };
        driver.on_tool_made(
            Ts::ZERO,
            (speech, 1),
            MadeCall { call, spec: Some(spec), forbidden: false, agent: None },
            crate::agent_session::RequestId(700),
        );
        assert_eq!(
            driver.activity.tools.deferred.len(),
            1,
            "remote admission waits for its capture"
        );
        driver.reply_spoken(speech, None);
        let original = candidate("input_1");
        driver.input_offers(speech, ModelRequestId(request.0 + 1), vec![original.clone()]);
        assert!(
            !driver.remote_inputs_ready(speech, "travel"),
            "late captures cannot release another model turn"
        );
        let human = ChatMessage::text("human", Role::User, "Friday", UNIX_EPOCH);
        if order == Some(true) {
            driver.route_realtime_input_for(speech, human.clone());
        }
        driver.input_offers(speech, request, vec![original.clone()]);
        if order.is_none() {
            driver.realtime_input_ended(speech);
            assert!(driver.remote_inputs_ready(speech, "travel"));
            driver.settle_deferred_calls();
            assert!(
                driver.activity.tools.deferred.is_empty(),
                "an ended provider cannot leave calls waiting for an unowned transcript"
            );
            let reply = &driver.activity.tools.replies[&speech];
            let id = reply.run.step_call_id("wire").unwrap();
            let (call, _) = reply.run.step_call(id).unwrap();
            assert!(
                lk_agents_core::delegation::remote::routing::captured_offers(call).is_err(),
                "the call receives refusal feedback instead of dispatching guessed human words"
            );
            driver.activity.tools.replies.get_mut(&speech).unwrap().input_offers(vec![]);
            driver.expect_realtime_input(speech);
            assert!(
                driver.remote_inputs_ready(speech, "travel"),
                "new work needs no transcript when no pending response was offered"
            );
            continue;
        }
        if order == Some(false) {
            driver.route_realtime_input_for(speech, human.clone());
        }
        let request_id = driver
            .take_outbox()
            .into_iter()
            .find_map(|effect| match effect {
                VoiceEffect::RouteInput { request_id, .. } => Some(request_id),
                _ => None,
            })
            .unwrap();
        let mut held = original;
        held.response = Some(human);
        let admission = InputAdmission {
            offered: vec![held.clone(), candidate("input_2")],
            ..Default::default()
        };
        driver.input_routed(
            speech,
            "stale-attempt".into(),
            Some(Ok(admission.clone())),
            Default::default(),
        );
        assert!(!driver.remote_inputs_ready(speech, "travel"));
        driver.input_routed(speech, request_id, Some(Ok(admission)), Default::default());
        assert!(driver.remote_inputs_ready(speech, "travel"));
        driver.settle_deferred_calls();
        assert!(driver.activity.tools.deferred.is_empty());
        let reply = &driver.activity.tools.replies[&speech];
        let id = reply.run.step_call_id("wire").unwrap();
        let (call, _) = reply.run.step_call(id).unwrap();
        assert_eq!(
            lk_agents_core::delegation::remote::routing::captured_offers(call).unwrap(),
            vec![held],
            "a later question cannot become this already-started model turn's answer target"
        );
    }
}
