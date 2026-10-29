//! A coordinated invocation (VoiceSession): the coordinator plans every request and decides each
//! settled step's continuation.

use super::*;

fn coordinated(agent: AgentDefinition) -> Driver {
    let input = RunInput {
        invocation_id: "reply_1".into(),
        opening: Opening::Coordinated { approval_answers: Default::default() },
        history: vec![user("msg_1", "Book me Monday").into()],
        answer_tool: None,
        delegated: false,
        mode: crate::tool_outcome::Mode::Voice,
        llm: Some(LlmRef { id: LLM, provider: ProviderId::of_label("fake") }),
        wall: WallAnchor { wall: 1_700_000_000.0, at: Ts::ZERO },
        cleanup_timeout: Duration::from_secs(5),
    };
    Driver::with_input(agent, input)
}

struct Plan {
    request: RequestId,
    message_id: String,
    pending: Vec<ChatItem>,
    tools_forbidden: bool,
}

impl Driver {
    fn plans(&self) -> Vec<Plan> {
        (self.effects.iter())
            .filter_map(|e| match e {
                Effect::PlanRequest { request, message_id, pending, tools_forbidden, .. } => {
                    Some(Plan {
                        request: *request,
                        message_id: message_id.clone(),
                        pending: pending.clone(),
                        tools_forbidden: *tools_forbidden,
                    })
                }
                _ => None,
            })
            .collect()
    }

    fn respond(&mut self, request: RequestId, text: &str, calls: &[(&str, Value)]) {
        let calls = (calls.iter().enumerate())
            .map(|(i, (name, arguments))| FunctionToolCall {
                name: (*name).into(),
                arguments: arguments.to_string(),
                call_id: format!("call_{}_{i}", request.0),
                extra: None,
            })
            .collect();
        let text = (!text.is_empty()).then_some(text);
        self.send(Event::LlmChunk { request, chunk: chunk(text, calls, None) });
        self.send(Event::LlmEnded { request });
    }

    fn step_settled(&self) -> (StepId, Vec<ChatItem>, bool) {
        (self.effects.iter().rev())
            .find_map(|e| match e {
                Effect::StepSettled { step, items, continues } => {
                    Some((*step, items.clone(), *continues))
                }
                _ => None,
            })
            .expect("a settled step")
    }
}

// Catches: a coordinated run planning its own request (rendering instructions, pushing a
// message), the next step requested before the coordinator's verdict, the step's items missing
// from the next plan, or a forbidding verdict not reaching the next request.
#[test]
fn the_coordinator_plans_each_step_and_decides_its_continuation() {
    let mut d = coordinated(text_agent(vec![string_tool("find_slot", "day")]));
    assert!(d.effects.iter().all(|e| !matches!(e, Effect::LlmRequestStart { .. })));
    let [first] = &d.plans()[..] else { panic!("one plan: {:?}", d.effects) };
    assert!(first.pending.is_empty());
    assert!(!first.tools_forbidden);
    let (request, message_id) = (first.request, first.message_id.clone());

    d.respond(request, "Let me look.", &[("find_slot", json!({"day": "Monday"}))]);
    d.run_tools(|_| value("9:00 is free"));
    let (step, items, continues) = d.step_settled();
    assert!(continues);
    assert_eq!(
        transcript(&items),
        pairs(&[("call", r#"find_slot({"day":"Monday"})"#), ("output", "9:00 is free")])
    );
    assert_eq!(d.plans().len(), 1, "no step before the verdict");

    d.send(Event::StepVerdict { step, verdict: StepVerdict::Continue { tools_forbidden: true } });
    let plans = d.plans();
    let second = &plans[1];
    assert!(second.tools_forbidden);
    assert_eq!(second.pending[0].id(), message_id);
    assert_eq!(
        transcript(&second.pending),
        pairs(&[
            ("assistant", "Let me look."),
            ("call", r#"find_slot({"day":"Monday"})"#),
            ("output", "9:00 is free"),
        ])
    );

    d.respond(second.request, "Monday 9:00 is free.", &[]);
    assert_eq!(
        d.result().outcome,
        RunOutcome::Completed { output: Output::Text("Monday 9:00 is free.".into()) }
    );
}

// Catches: an interrupted reply requesting another step or settling as a plain completion, or
// an ending verdict leaving the run open.
#[test]
fn an_interrupted_or_ended_step_ends_the_run_without_another_request() {
    for (verdict, interrupted) in [(StepVerdict::Interrupted, true), (StepVerdict::End, false)] {
        let mut d = coordinated(text_agent(vec![string_tool("find_slot", "day")]));
        let request = d.plans()[0].request;
        d.respond(request, "", &[("find_slot", json!({"day": "Monday"}))]);
        d.run_tools(|_| value("9:00 is free"));
        let (step, ..) = d.step_settled();
        d.send(Event::StepVerdict { step, verdict });
        assert_eq!(d.plans().len(), 1);
        let result = d.result();
        assert_eq!(result.outcome, RunOutcome::Completed { output: Output::Nothing });
        assert_eq!(result.interrupted, interrupted);
    }
}

// Catches: a coordinator's interruption discarding an output the run already accepted.
#[test]
fn an_accepted_output_survives_an_interrupting_verdict() {
    let mut d =
        coordinated(agent(vec![], OutputSpec::Structured { schema: json!({"type": "integer"}) }));
    let request = d.plans()[0].request;
    d.respond(request, "", &[("final_result", json!({"result": 4}))]);
    let validation = (d.effects.iter())
        .find_map(|e| match e {
            Effect::OutputValidate { validation, .. } => Some(*validation),
            _ => None,
        })
        .expect("the candidate is validated");
    d.send(Event::OutputValidated { validation, verdict: OutputVerdict::Accepted { text: None } });
    let (step, _, continues) = d.step_settled();
    assert!(!continues);
    d.send(Event::StepVerdict { step, verdict: StepVerdict::Interrupted });
    assert_eq!(d.result().outcome, RunOutcome::Completed { output: Output::Validated(validation) });
}
