use super::*;
use std::sync::atomic::{AtomicBool, Ordering};

use lk_agents_abi::a2a::client::AgentClientError;
use lk_agents_abi::a2a::remote::{DelegationInput, RemoteResponseClaim};
use lk_agents_abi::delegation::remote::{RemoteAgentState, RemoteRequestState};
use lk_agents_abi::session::ToolCallRef;
use lk_agents_abi::{ChatMessage, Role};

const REMOTE: &str = "booking";
const ENDPOINT: &str = "https://booking.test";

struct Driver {
    live: AtomicBool,
}

impl Driver {
    fn new() -> Self {
        Self { live: AtomicBool::new(true) }
    }

    fn close(&self) {
        self.live.store(false, Ordering::Release);
    }
}

impl RemoteTaskDriver for Driver {
    fn is_live(&self) -> bool {
        self.live.load(Ordering::Acquire)
    }

    fn reconcile(&self) -> lk_agents_abi::BoxFuture<'_, ()> {
        Box::pin(async {})
    }

    fn input<'a>(
        &'a self,
        _input: DelegationInput,
        _response_claim: Option<Arc<RemoteResponseClaim>>,
        _origin: Option<ToolCallRef>,
    ) -> lk_agents_abi::BoxFuture<'a, Result<(), AgentClientError>> {
        Box::pin(async { Ok(()) })
    }

    fn resume<'a>(
        &'a self,
        _request_id: String,
        _request: Box<lk_agents_abi::a2a::protocol::SendMessageRequest>,
        _streaming: bool,
        _input_request: Option<lk_agents_abi::session::InputRequestRef>,
        _origin: Option<ToolCallRef>,
    ) -> lk_agents_abi::BoxFuture<'a, Result<(), AgentClientError>> {
        Box::pin(async { panic!("the registry fixture never dispatches a stored request") })
    }
}

fn operation(sequence: u64, task: Option<RemoteTaskRef>) -> RemoteRequestState {
    RemoteRequestState {
        method: "message/send".into(),
        request_event: sequence,
        dispatch_started: true,
        ended: true,
        task,
        origin: None,
        input_request: None,
    }
}

fn booking() -> (SessionSnapshot, RemoteTaskRef) {
    let task = RemoteTaskRef { context_id: "meeting".into(), task_id: "booking".into() };
    let mut state = SessionSnapshot::new("conversation".into(), "run".into(), "desk".into());
    let mut remote = RemoteAgentState::new(ENDPOINT.into());
    remote.requests.insert("original".into(), operation(0, Some(task.clone())));
    state.remote_agents.insert(REMOTE.into(), remote);
    (state, task)
}

fn response(task: &RemoteTaskRef, words: &str) -> CallAdmission {
    CallAdmission::Respond {
        request: InputRequestRef::RemoteTask {
            remote_id: REMOTE.into(),
            task: task.clone(),
            request_id: "question-1".into(),
        },
        message: Box::new(ChatMessage::text("human", Role::User, words, std::time::UNIX_EPOCH)),
    }
}

#[test]
fn response_admissions_share_one_live_owner_before_any_journal_prepare() {
    let (state, task) = booking();
    let drivers = Drivers::default();
    let first: Arc<dyn RemoteTaskDriver> = Arc::new(Driver::new());
    let second: Arc<dyn RemoteTaskDriver> = Arc::new(Driver::new());
    let answer = response(&task, "Friday");

    let admitted =
        drivers.admit(&state, REMOTE, ENDPOINT, answer.clone(), Arc::downgrade(&first)).unwrap();
    assert_eq!(admitted.admission, answer);
    assert!(Arc::ptr_eq(admitted.driver.as_ref().unwrap(), &first));
    assert!(admitted.response_claim.is_some());
    let duplicate =
        drivers.admit(&state, REMOTE, ENDPOINT, answer, Arc::downgrade(&second)).unwrap();
    assert_eq!(duplicate.admission, CallAdmission::AlreadyForwarded { task: task.clone() });
    assert!(duplicate.driver.is_none(), "a duplicate must not acquire an independent observer");
    assert!(duplicate.response_claim.is_none());
    let owner = drivers.get(&state, REMOTE, ENDPOINT, &task).unwrap().unwrap();
    assert!(Arc::ptr_eq(&owner, &first));
    assert_eq!(state.remote_agents[REMOTE].requests.len(), 1);
}

#[test]
fn a_lost_admission_receipt_releases_its_answer_claim_while_the_driver_stays_live() {
    let (state, task) = booking();
    let drivers = Drivers::default();
    let first: Arc<dyn RemoteTaskDriver> = Arc::new(Driver::new());
    let retry: Arc<dyn RemoteTaskDriver> = Arc::new(Driver::new());
    let answer = response(&task, "Friday");

    let lost =
        drivers.admit(&state, REMOTE, ENDPOINT, answer.clone(), Arc::downgrade(&first)).unwrap();
    assert!(lost.response_claim.is_some());
    drop(lost);
    let admitted =
        drivers.admit(&state, REMOTE, ENDPOINT, answer.clone(), Arc::downgrade(&retry)).unwrap();
    assert_eq!(admitted.admission, answer, "an unsent response must remain admissible");
    assert!(admitted.response_claim.is_some());
    assert!(Arc::ptr_eq(admitted.driver.as_ref().unwrap(), &first));
}

#[test]
fn an_unsent_claim_can_release_without_ending_its_existing_task_observer() {
    let (state, task) = booking();
    let drivers = Drivers::default();
    let incumbent: Arc<dyn RemoteTaskDriver> = Arc::new(Driver::new());
    let candidate: Arc<dyn RemoteTaskDriver> = Arc::new(Driver::new());
    drivers.register(REMOTE, ENDPOINT, "original", Arc::downgrade(&incumbent));
    let answer = response(&task, "Friday");
    let mut admitted = drivers
        .admit(&state, REMOTE, ENDPOINT, answer.clone(), Arc::downgrade(&candidate))
        .unwrap();
    assert!(Arc::ptr_eq(admitted.driver.as_ref().unwrap(), &incumbent));
    let submission = admitted.response_claim.take().unwrap();
    drop(admitted);

    // After the receipt transfers ownership, only the concrete submission retains it.
    let duplicate = drivers
        .admit(&state, REMOTE, ENDPOINT, answer.clone(), Arc::downgrade(&candidate))
        .unwrap();
    assert_eq!(duplicate.admission, CallAdmission::AlreadyForwarded { task: task.clone() });
    drop(submission);
    let retry = drivers
        .admit(&state, REMOTE, ENDPOINT, answer.clone(), Arc::downgrade(&candidate))
        .unwrap();
    assert_eq!(retry.admission, answer);
    assert!(Arc::ptr_eq(retry.driver.as_ref().unwrap(), &incumbent));
    assert!(incumbent.is_live());
}

#[test]
fn dropping_an_old_claim_cannot_release_a_correction_with_the_same_message_id() {
    let (state, task) = booking();
    let drivers = Drivers::default();
    let candidate: Arc<dyn RemoteTaskDriver> = Arc::new(Driver::new());
    let original = drivers
        .admit(&state, REMOTE, ENDPOINT, response(&task, "Friday"), Arc::downgrade(&candidate))
        .unwrap();
    let correction = response(&task, "Actually Saturday");
    let corrected = drivers
        .admit(&state, REMOTE, ENDPOINT, correction.clone(), Arc::downgrade(&candidate))
        .unwrap();
    assert_eq!(corrected.admission, correction);
    assert!(!Arc::ptr_eq(
        original.response_claim.as_ref().unwrap(),
        corrected.response_claim.as_ref().unwrap(),
    ));
    drop(original);

    let duplicate = drivers
        .admit(&state, REMOTE, ENDPOINT, correction.clone(), Arc::downgrade(&candidate))
        .unwrap();
    assert_eq!(duplicate.admission, CallAdmission::AlreadyForwarded { task: task.clone() });
    drop(corrected);
    let retry = drivers
        .admit(&state, REMOTE, ENDPOINT, correction.clone(), Arc::downgrade(&candidate))
        .unwrap();
    assert_eq!(retry.admission, correction);
}

#[test]
fn ordinary_task_messages_remain_distinct_while_an_answer_is_claimed() {
    let (state, task) = booking();
    let drivers = Drivers::default();
    let first: Arc<dyn RemoteTaskDriver> = Arc::new(Driver::new());
    let next: Arc<dyn RemoteTaskDriver> = Arc::new(Driver::new());
    let answer = drivers
        .admit(&state, REMOTE, ENDPOINT, response(&task, "Friday"), Arc::downgrade(&first))
        .unwrap();
    for _ in 0..2 {
        let ordinary = drivers
            .admit(
                &state,
                REMOTE,
                ENDPOINT,
                CallAdmission::Continue(task.clone()),
                Arc::downgrade(&next),
            )
            .unwrap();
        assert_eq!(ordinary.admission, CallAdmission::Continue(task.clone()));
        assert!(ordinary.response_claim.is_none());
        assert!(Arc::ptr_eq(ordinary.driver.as_ref().unwrap(), &first));
    }
    drop(answer);
}

#[test]
fn a_closed_consumer_is_ineligible_while_its_sender_is_still_retained() {
    let (mut state, task) = booking();
    let drivers = Drivers::default();
    let first = Arc::new(Driver::new());
    let first_handle: Arc<dyn RemoteTaskDriver> = first.clone();
    let second: Arc<dyn RemoteTaskDriver> = Arc::new(Driver::new());
    state
        .remote_agents
        .get_mut(REMOTE)
        .unwrap()
        .requests
        .insert("second".into(), operation(1, Some(task.clone())));
    drivers.register(REMOTE, ENDPOINT, "original", Arc::downgrade(&first_handle));
    drivers.register(REMOTE, ENDPOINT, "second", Arc::downgrade(&second));
    let owner = drivers.get(&state, REMOTE, ENDPOINT, &task).unwrap().unwrap();
    assert!(Arc::ptr_eq(&owner, &first_handle));
    first.close();

    let owner = drivers.get(&state, REMOTE, ENDPOINT, &task).unwrap().unwrap();
    assert!(Arc::ptr_eq(&owner, &second), "a closed cached consumer cannot keep task ownership");
    assert!(!first_handle.is_live());
}

#[test]
fn a_late_merged_task_association_cannot_replace_its_live_result_owner() {
    let drivers = Drivers::default();
    let first: Arc<dyn RemoteTaskDriver> = Arc::new(Driver::new());
    let second: Arc<dyn RemoteTaskDriver> = Arc::new(Driver::new());
    let task = RemoteTaskRef { context_id: "meeting".into(), task_id: "booking".into() };
    let mut state = SessionSnapshot::new("conversation".into(), "run".into(), "desk".into());
    let mut remote = RemoteAgentState::new("https://booking.test".into());
    remote.requests.insert("first".into(), operation(0, None));
    remote.requests.insert("second".into(), operation(1, Some(task.clone())));
    state.remote_agents.insert("booking".into(), remote);
    drivers.register("booking", "https://booking.test", "first", Arc::downgrade(&first));
    drivers.register("booking", "https://booking.test", "second", Arc::downgrade(&second));

    let owner = drivers.get(&state, "booking", "https://booking.test", &task).unwrap().unwrap();
    assert!(Arc::ptr_eq(&owner, &second));
    drop(owner);
    state.remote_agents.get_mut("booking").unwrap().requests.get_mut("first").unwrap().task =
        Some(task.clone());
    let owner = drivers.get(&state, "booking", "https://booking.test", &task).unwrap().unwrap();
    assert!(Arc::ptr_eq(&owner, &second));
    assert!(drivers.get(&state, "booking", "https://another.test", &task).is_err());

    // A recovered observer can claim the task only once the old runtime resource has ended.
    drop(owner);
    drop(second);
    let owner = drivers.get(&state, "booking", "https://booking.test", &task).unwrap().unwrap();
    assert!(Arc::ptr_eq(&owner, &first));
}
