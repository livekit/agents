use super::*;
use crate::a2a::{PolicySpec, TurnOptions};
use lk_agents_abi::agent::run::{Output, RunOutcome, RunResult};
use lk_agents_abi::session::snapshot::SessionSnapshot;
use lk_agents_abi::text_session::{LiveTurnOrigin, TextSessionTurn};

fn machine() -> TaskMachine {
    let mut wire = a2a::Message::new(a2a::Role::User, vec![a2a::Part::text("start")]);
    wire.message_id = "initial".into();
    TaskMachine::new(
        "desk".into(),
        Message::from_a2a(&wire, 0.0, None).unwrap(),
        "task_A".into(),
        &TurnOptions { policy: PolicySpec::All(Policy::Live), ..Default::default() },
        0.0,
    )
}

fn admit(machine: &mut TaskMachine, task: &str) -> String {
    let mut wire = a2a::Message::new(a2a::Role::User, vec![a2a::Part::text(task)]);
    wire.message_id = format!("wire_{task}");
    machine.start_turn(
        vec![Message::from_a2a(&wire, 0.0, None).unwrap()],
        vec![task.into()],
        Path::Main,
        Duration::ZERO,
    );
    let Some(Operation::Handle(handle)) = machine
        .turn_step(Step::Pending { named: false }, Completion::Task(None), Duration::ZERO)
        .unwrap()
    else {
        panic!("a source opens its own handler")
    };
    assert_eq!(
        machine.turn_step(Step::Handle, Completion::Done, Duration::ZERO).unwrap(),
        Some(Operation::NextAll { should_yield: None, pause: PARTITION_POLL_PAUSE })
    );
    handle.message.id
}

fn report(source: &str) -> LiveTurnReport {
    let state = SessionSnapshot::new("session".into(), "task_A".into(), "desk".into());
    LiveTurnReport {
        turn: 1,
        origin: LiveTurnOrigin {
            request_id: source.into(),
            execution_id: "task_A".into(),
            task_ids: vec!["task_A".into()],
        },
        result: Box::new(Ok(TextSessionTurn {
            result: RunResult {
                outcome: RunOutcome::Completed { output: Output::Text("accepted A".into()) },
                interrupted: false,
                new_messages: Vec::new(),
                all_messages: Vec::new(),
                usage: Default::default(),
                verbatim: None,
                pending_call: None,
                input_requirements: Vec::new(),
            },
            state,
        })),
        working: false,
    }
}

#[test]
fn an_original_checkpoint_is_answered_for_its_task_while_a_side_handler_has_the_floor() {
    let mut machine = machine();
    let original = admit(&mut machine, "task_A");
    admit(&mut machine, "task_B");
    let next = machine
        .turn_step(
            Step::Next,
            Completion::Event(HandlerEvent::LiveCheckpoint { request_id: original.clone() }),
            Duration::ZERO,
        )
        .unwrap();
    assert_eq!(next, Some(Operation::Checkpoint(vec!["task_A".into()])));
    assert_eq!(machine.live.checkpoint_source.as_deref(), Some(original.as_str()));
    assert_eq!(machine.turn.as_ref().unwrap().answer_task, "task_B");
}

#[test]
fn another_sources_return_waits_for_the_exact_native_finalization_without_replacing_it() {
    let mut machine = machine();
    let original = admit(&mut machine, "task_A");
    let side = admit(&mut machine, "task_B");
    assert!(matches!(machine.managed_turn(report(&original), Duration::ZERO).unwrap(),
        Some(Operation::NextSource {request_id, ..}) if request_id == original));
    let stale = |text: &str| {
        Box::new(HandlerOutput {
            output: Some(text.into()),
            state: Some(serde_json::json!({"stale":true})),
            ..Default::default()
        })
    };
    assert!(matches!(machine.live_output(side.clone(), stale("side B"), Duration::ZERO).unwrap(),
        Some(Operation::NextSource {request_id, ..}) if request_id == original));
    assert_eq!(machine.turn.as_ref().unwrap().answer_task, "task_A");
    assert_eq!(machine.live.outputs.len(), 1);
    let Some(Operation::Stage(original_stage)) =
        machine.live_output(original.clone(), stale("stale A"), Duration::ZERO).unwrap()
    else {
        panic!("the original result is staged for its exact task")
    };
    assert_eq!(machine.answer, Some(("task_A".into(), Some(Value::String("accepted A".into())))));
    assert!(original_stage.end.as_ref().unwrap().state.is_none());
    assert!(original_stage.transitions.iter().any(|transition| {
        transition.task_id == "task_A" && transition.status == TaskState::Completed
    }));
    assert!(original_stage.transitions.iter().all(|transition| transition.task_id != "task_B"));

    machine.commit_live_turn();
    machine
        .live_step(
            LiveStep::Commit,
            Completion::Committed { task: None, run_id: "first".into() },
            Duration::ZERO,
        )
        .unwrap();
    machine.live_step(LiveStep::Acknowledged, Completion::Done, Duration::ZERO).unwrap();
    let Some(Operation::PrepareDialogue { source, messages }) = machine
        .live_step(
            LiveStep::Open,
            Completion::Opened { closed: false, picked_up: false, history_end: 0.0 },
            Duration::ZERO,
        )
        .unwrap()
    else {
        panic!("the deferred opaque result prepares its context through the existing owner")
    };
    assert_eq!(source, side);
    assert!(messages.iter().any(|message| message.raw_text_content().as_deref() == Some("side B")));
    let canonical = serde_json::json!({"native_owner_snapshot":true});
    let Some(Operation::Stage(side_stage)) = machine
        .live_step(
            LiveStep::DialoguePrepared,
            Completion::DialoguePrepared(canonical.clone()),
            Duration::ZERO,
        )
        .unwrap()
    else {
        panic!("the side result is staged with its native owner's state")
    };
    assert_eq!(machine.turn.as_ref().unwrap().answer_task, "task_B");
    assert_eq!(machine.answer, Some(("task_B".into(), Some(Value::String("side B".into())))));
    assert_eq!(side_stage.end.as_ref().unwrap().state, Some(canonical));
    assert!(side_stage.transitions.iter().any(|transition| {
        transition.task_id == "task_B" && transition.status == TaskState::Completed
    }));
    assert!(side_stage.transitions.iter().all(|transition| transition.task_id != "task_A"));
    assert!(machine.live.outputs.is_empty());
}

#[test]
fn an_unaddressed_message_opens_its_own_task_without_answering_a_waiting_question() {
    let mut machine = machine();
    let mut wire = a2a::Message::new(a2a::Role::User, vec![a2a::Part::text("When is check-in?")]);
    wire.message_id = "unaddressed".into();

    let first = machine.start_turn(
        vec![Message::from_a2a(&wire, 0.0, None).unwrap()],
        vec!["task_new".into()],
        Path::Main,
        Duration::ZERO,
    );

    // A queue or batch message would read the waiting question and complete it `mergedInto`
    // this one; under live only an explicit taskId continues a task.
    let Operation::Handle(handle) = first else {
        panic!("the unaddressed message opens its own handler, got {first:?}")
    };
    assert_eq!(handle.task_id, "task_new");
}

#[test]
fn an_explicit_task_id_still_reads_the_question_it_continues() {
    let mut machine = machine();
    let mut wire = a2a::Message::new(a2a::Role::User, vec![a2a::Part::text("The double.")]);
    wire.message_id = "answer".into();
    wire.task_id = Some("task_asked".into());

    let first = machine.start_turn(
        vec![Message::from_a2a(&wire, 0.0, None).unwrap()],
        vec!["task_received".into()],
        Path::Main,
        Duration::ZERO,
    );

    assert_eq!(first, Operation::ReadPending(Some("task_asked".into())));
}

#[test]
fn a_task_id_that_names_no_waiting_question_answers_no_other_question() {
    let mut machine = machine();
    let mut wire = a2a::Message::new(a2a::Role::User, vec![a2a::Part::text("Friday.")]);
    wire.message_id = "named".into();
    wire.task_id = Some("task_done".into());
    machine.start_turn(
        vec![Message::from_a2a(&wire, 0.0, None).unwrap()],
        vec!["task_received".into()],
        Path::Main,
        Duration::ZERO,
    );

    let next = machine
        .turn_step(Step::Pending { named: true }, Completion::Task(None), Duration::ZERO)
        .unwrap();

    let Some(Operation::Handle(handle)) = next else {
        panic!("the message opens its own handler, got {next:?}")
    };
    assert_eq!(handle.task_id, "task_received");
}
