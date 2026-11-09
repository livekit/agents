use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use parking_lot::Mutex;
use serde_json::json;
use tokio::sync::{mpsc, oneshot};
use tokio_util::sync::{CancellationToken, DropGuard};

use super::fakes::{Callback, FakeHandler, FakeHost, FakeLlm, Fault, RunFakes, tool};
use super::*;
use crate::fakes::storage::text::Store;
use lk_agents_abi::a2a::options::{Policy, PolicySpec, TurnOptions};
use lk_agents_abi::a2a::turn::HandlerPort;
use lk_agents_abi::agent::run::RunUpdate;
use lk_agents_abi::agent::{AgentDefinition, Invocation, OutputSpec, Retries};
use lk_agents_abi::errors::RefRegistry;
use lk_agents_abi::resources::{LlmRef, ResourceId};
use lk_agents_abi::telemetry::spans::common::GenAiOperation;
use lk_agents_abi::telemetry::spans::text::AgentRun;
use lk_agents_abi::text_session::{
    TextSessionAgent, TextSessionDelivery, TextSessionRequest, TextSessionRunner, TextSessionUpdate,
};
use lk_agents_abi::{
    ChatItem, ChatMessage, Clock, HostRef, Llm, Role, StreamOutput, Ts, WallAnchor,
};
use lk_agents_core::tool_outcome::Mode;
use lk_agents_core::{Opening, RunInput};
use lk_agents_runtime::a2a::Turn;
use lk_agents_runtime::agent_run::{self, RunSetup};
use lk_agents_runtime::net::env::{ProcessEnv, ProcessError};
use lk_agents_runtime::storage::StorageError;

const LANE: &str = "front-desk";
/// How long a client waits on its turn before giving up on it.
const TURN_DEADLINE_MS: u64 = 600_000;

/// One seeded situation: what runs, how the clients behave, and which faults may strike.
#[derive(Clone, Debug)]
pub struct Scenario {
    pub seed: u64,
    pub storage_faults: bool,
    /// Every fake may panic or never answer, now and then.
    pub random_faults: bool,
    /// Faults each run's fakes meet on that callback's first call.
    pub forced: Vec<(Callback, Fault)>,
    /// Direct runs, each with the instant its client cancels it.
    pub direct: Vec<Option<u64>>,
    /// A TextSession request, with the instant its client cancels it.
    pub session: Option<Option<u64>>,
    pub policy: Policy,
    pub turns: Vec<TurnPlan>,
}

#[derive(Clone, Copy, Debug)]
pub struct TurnPlan {
    pub start_ms: u64,
    pub client: Client,
    /// When a cancelling or aborting client acts.
    pub after_ms: u64,
    /// Which of two worker processes sends it.
    pub worker: u8,
}

impl Scenario {
    /// One plain direct run: no faults, no turns.
    pub fn direct(seed: u64) -> Self {
        Self {
            seed,
            storage_faults: false,
            random_faults: false,
            forced: Vec::new(),
            direct: vec![None],
            session: None,
            policy: Policy::Queue,
            turns: Vec::new(),
        }
    }

    pub fn draw(seed: u64) -> Self {
        let mut rng = fastrand::Rng::with_seed(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ 0x7478);
        let cancel = |rng: &mut fastrand::Rng| (rng.u32(0..3) == 0).then(|| rng.u64(0..6_000));
        let policies =
            [Policy::Batch, Policy::Interrupt, Policy::Queue, Policy::Concurrent, Policy::Reject];
        let clients =
            [Client::Waits, Client::Waits, Client::Disconnects, Client::Cancels, Client::Aborts];
        let direct = (0..rng.u32(0..=2)).map(|_| cancel(&mut rng)).collect();
        let session = (rng.u32(0..3) == 0).then(|| cancel(&mut rng));
        let turns = (0..rng.u32(0..=4))
            .map(|_| TurnPlan {
                start_ms: rng.u64(0..3_000),
                client: clients[rng.usize(0..clients.len())],
                after_ms: rng.u64(0..4_000),
                worker: rng.u8(0..2),
            })
            .collect();
        Self {
            seed,
            storage_faults: rng.bool(),
            random_faults: rng.bool(),
            forced: Vec::new(),
            direct,
            session,
            policy: policies[rng.usize(0..policies.len())],
            turns,
        }
    }
}

/// A finished run and its verdict.
pub struct Outcome {
    pub scenario: Scenario,
    pub recording: TextRecording,
    pub failures: Vec<Failure>,
    pub expectations: Vec<(&'static str, u32)>,
    pub worker_deaths: u64,
}

/// What every fake and client of a run shares.
pub struct Env {
    pub clock: Arc<dyn Clock>,
    pub refs: Arc<dyn RefRegistry>,
    tracer: lk_agents_telemetry::Tracer,
    process: ProcessEnv,
    scenario: Scenario,
    /// The fakes stop faulting: the lane's next turn after the scenario.
    calm: std::sync::atomic::AtomicBool,
    recording: Mutex<TextRecording>,
}

impl Env {
    pub fn mark(&self, id: u32, f: impl FnOnce(&mut RunRecord)) {
        if let Some(run) = self.recording.lock().runs.iter_mut().find(|r| r.id == id) {
            f(run)
        }
    }

    fn now_ms(&self) -> u64 {
        self.clock.now().as_nanos() / 1_000_000
    }

    pub fn wall(&self) -> SystemTime {
        self.clock.wall()
    }

    /// Run `id`'s fake seed, forced faults and whether random faults strike it.
    pub fn fakes_for(&self, id: u32) -> (u64, Vec<(Callback, Fault)>, bool) {
        let seed = self.scenario.seed.wrapping_mul(31).wrapping_add(u64::from(id));
        if self.calm.load(std::sync::atomic::Ordering::Acquire) {
            return (seed, Vec::new(), false);
        }
        (seed, self.scenario.forced.clone(), self.scenario.random_faults)
    }

    fn add_run(&self, kind: RunKind) -> u32 {
        let mut recording = self.recording.lock();
        let id = recording.runs.len() as u32;
        let mut run = run_record(kind);
        run.id = id;
        (run.terminal_ms, run.exited_ms) = (None, None);
        recording.runs.push(run);
        id
    }

    fn agent() -> AgentDefinition {
        AgentDefinition {
            name: "desk".into(),
            instructions: "Answer bookings.".into(),
            dynamic_instructions: false,
            tools: vec![tool()],
            output: OutputSpec::Text { validated: false },
            retries: Retries::default(),
        }
    }

    fn message(&self, id: &str) -> ChatMessage {
        ChatMessage::text(id, Role::User, "When is my booking?", self.wall())
    }

    /// Starts a real AgentRun over this run's fakes; resolves `true` once it finished, `false`
    /// once it failed. Dropping the guard cancels it.
    pub fn start_run(
        self: &Arc<Self>,
        kind: RunKind,
        port: Option<Arc<dyn HandlerPort>>,
        cancel_after: Option<u64>,
    ) -> (oneshot::Receiver<bool>, DropGuard) {
        let id = self.add_run(kind);
        let fakes = RunFakes::new(self.clone(), id, port);
        let llm: Arc<dyn Llm> = Arc::new(FakeLlm(fakes.clone()));
        let invocation_id = format!("run_{id}");
        let input = RunInput {
            invocation_id: invocation_id.clone(),
            opening: Opening::Message(Box::new(self.message(&format!("m_{id}")))),
            history: Vec::new(),
            answer_tool: None,
            delegated: false,
            mode: Mode::Text,
            llm: Some(LlmRef { id: ResourceId(0), provider: llm.id() }),
            wall: WallAnchor::of(&*self.clock),
            cleanup_timeout: lk_agents_abi::HOST_TEARDOWN_TIMEOUT,
        };
        let setup = RunSetup {
            llm,
            tools_host: None,
            host: Arc::new(FakeHost(fakes)),
            clock: self.clock.clone(),
            conn_options: Default::default(),
            entry: Invocation {
                operation: 0,
                speech: None,
                invocation_id,
                agent_name: "desk".into(),
                execution_id: None,
                frame_id: None,
                history: Vec::new(),
                function_call: None,
                approval: None,
                host_context: Some(HostRef::new(0, &self.refs)),
            },
        };
        let decl = AgentRun { operation: GenAiOperation::InvokeAgent, agent_name: "desk".into() };
        let span = lk_agents_telemetry::start(&self.tracer.root(), decl);
        let spawner = (self.process.spawner)().expect("the DST's runtime");
        let (active, updates) =
            agent_run::start(&spawner, Self::agent(), input, setup, span, 16).expect("a valid run");
        let (done, finished) = oneshot::channel();
        let stop = CancellationToken::new();
        let watch = self.clone().watch_run(id, active, updates, cancel_after, stop.clone(), done);
        self.spawn(watch);
        (finished, stop.drop_guard())
    }

    fn spawn(&self, work: impl Future<Output = ()> + Send + 'static) {
        drop(tokio::spawn(work));
    }

    fn cancel_deadline(&self, start: Ts, cancel_after: Option<u64>) -> (Ts, CancelBy) {
        let deadline = start.saturating_add(Duration::from_millis(CLIENT_DEADLINE_MS));
        match cancel_after.map(|ms| start.saturating_add(Duration::from_millis(ms))) {
            Some(at) if at < deadline => (at, CancelBy::Client),
            _ => (deadline, CancelBy::Deadline),
        }
    }

    fn cancelled(&self, id: u32, by: CancelBy) -> Ts {
        let at = self.now_ms();
        self.mark(id, |r| r.cancelled = Some((at, by)));
        self.clock.now().saturating_add(Duration::from_millis(EXIT_BOUND_MS))
    }

    /// The client's side of a run: its outcome, its cancel, and its driver's exit.
    async fn watch_run(
        self: Arc<Self>,
        id: u32,
        active: agent_run::ActiveRun,
        mut updates: mpsc::Receiver<RunUpdate>,
        cancel_after: Option<u64>,
        stop: CancellationToken,
        done: oneshot::Sender<bool>,
    ) {
        let clock = self.clock.clone();
        let (cancel_at, by) = self.cancel_deadline(clock.now(), cancel_after);
        let mut done = Some(done);
        let mut exit_by: Option<Ts> = None;
        loop {
            tokio::select! {
                biased;
                update = updates.recv() => match update {
                    Some(RunUpdate::Finished(_)) | Some(RunUpdate::Failed { .. }) => {
                        let at = self.now_ms();
                        self.mark(id, |r| r.terminal_ms = r.terminal_ms.or(Some(at)));
                        let finished = matches!(update, Some(RunUpdate::Finished(_)));
                        if let Some(done) = done.take() {
                            let _ = done.send(finished);
                        }
                    }
                    Some(_) => {}
                    None => {
                        let at = self.now_ms();
                        self.mark(id, |r| r.exited_ms = Some(at));
                        return;
                    }
                },
                () = stop.cancelled(), if exit_by.is_none() => {
                    active.cancel();
                    exit_by = Some(self.cancelled(id, CancelBy::Client));
                }
                () = clock.wait_until(cancel_at), if exit_by.is_none() => {
                    active.cancel();
                    exit_by = Some(self.cancelled(id, by));
                }
                () = wait(&*clock, exit_by) => return,
            }
        }
    }

    /// A TextSession request over one agent, released as each invocation settles.
    async fn session(self: Arc<Self>, cancel_after: Option<u64>) {
        let id = self.add_run(RunKind::Session);
        let fakes = RunFakes::new(self.clone(), id, None);
        let agent = TextSessionAgent {
            definition: Self::agent(),
            llm: Arc::new(FakeLlm(fakes.clone())),
            host: Arc::new(FakeHost(fakes)),
            output_type_explicit: false,
            agent_tools: BTreeMap::new(),
            host_context: HostRef::new(0, &self.refs),
            tools_host: None,
        };
        let runner = lk_agents_runtime::text_session::Runner::new(
            &self.process,
            "desk".into(),
            vec![agent],
            4,
        )
        .expect("a valid session");
        let request = TextSessionRequest {
            session_id: format!("session_{id}"),
            execution_id: format!("exec_{id}"),
            task_id: None,
            message: ChatItem::Message(self.message(&format!("m_{id}"))),
            state: None,
            delegated: false,
            conn_options: Default::default(),
        };
        let mut updates = runner.run(request).expect("the session starts");
        let clock = self.clock.clone();
        let (cancel_at, by) = self.cancel_deadline(clock.now(), cancel_after);
        loop {
            let update = tokio::select! {
                biased;
                update = updates.next() => update.ok().flatten(),
                () = clock.wait_until(cancel_at) => {
                    let exit_by = self.cancelled(id, by);
                    let closed = Box::new(updates).close(exit_by).await;
                    if closed == lk_agents_abi::Closed::Clean {
                        let at = self.now_ms();
                        self.mark(id, |r| r.exited_ms = Some(at));
                    }
                    return;
                }
            };
            let at = self.now_ms();
            match update {
                Some(TextSessionUpdate::InvocationSettled(invocation_id))
                | Some(TextSessionUpdate::Parked(invocation_id)) => {
                    let delivery = TextSessionDelivery { invocation_id, messages: BTreeMap::new() };
                    let _ = runner.release(delivery);
                }
                Some(TextSessionUpdate::Finished(_)) | Some(TextSessionUpdate::Failed { .. }) => {
                    self.mark(id, |r| r.terminal_ms = r.terminal_ms.or(Some(at)));
                }
                Some(_) => {}
                None => {
                    self.mark(id, |r| r.exited_ms = Some(at));
                    return;
                }
            }
        }
    }

    /// One client's message to the lane, as its client behaves.
    async fn client(self: Arc<Self>, store: Arc<Store>, n: usize, plan: TurnPlan) -> TurnRecord {
        let clock = self.clock.clone();
        clock.wait_until(clock.now().saturating_add(Duration::from_millis(plan.start_ms))).await;
        let message_id = format!("msg_{n}");
        let message = json!({
            "messageId": message_id,
            "role": "ROLE_USER",
            "parts": [{"text": format!("When is booking {n}?")}],
        });
        let host: Arc<dyn lk_agents_abi::a2a::turn::TurnHost> = Arc::new(FakeHandler(self.clone()));
        let options =
            TurnOptions { policy: PolicySpec::All(self.scenario.policy), ..TurnOptions::default() };
        let record =
            |answered| TurnRecord { message_id: message_id.clone(), client: plan.client, answered };
        let thread = match store.thread().await {
            Ok(thread) => thread,
            Err(error) => return record(Some(Err(error.to_string()))),
        };
        let holder = format!("worker-{}", plan.worker);
        let spawner = self.process.spawner;
        let turn = Turn::new(thread.clone(), &holder, LANE, host, clock.clone(), spawner, options);
        let after = clock.now().saturating_add(Duration::from_millis(plan.after_ms));
        let answered = match plan.client {
            Client::Waits | Client::Disconnects => {
                let resends = if plan.client == Client::Waits { 2 } else { 0 };
                let mut attempt = 0;
                loop {
                    match turn.run(message.clone(), None).await {
                        Ok(result) => break Some(Ok(result.task.status)),
                        Err(error) if attempt < resends && resendable(&error) => {
                            attempt += 1;
                            clock
                                .wait_until(clock.now().saturating_add(Duration::from_secs(1)))
                                .await;
                        }
                        Err(error) => break Some(Err(error.to_string())),
                    }
                }
            }
            Client::Cancels => {
                let running = turn.run(message.clone(), None);
                tokio::pin!(running);
                let mut sent = false;
                loop {
                    tokio::select! {
                        biased;
                        result = &mut running => break Some(result.map(|r| r.task.status).map_err(|e| e.to_string())),
                        () = clock.wait_until(after), if !sent => {
                            sent = true;
                            cancel_task(&thread, &message_id).await;
                        }
                    }
                }
            }
            Client::Aborts => tokio::select! {
                biased;
                result = turn.run(message.clone(), None) => Some(result.map(|r| r.task.status).map_err(|e| e.to_string())),
                () = clock.wait_until(after) => None,
            },
        };
        thread.close().await;
        record(answered)
    }
}

/// A client's A2A cancel, as the door sends it: a terminal task stays as it is.
async fn cancel_task(thread: &lk_agents_runtime::storage::Thread, message_id: &str) {
    let lane = thread.lane(LANE);
    let Ok(Some(row)) = lane.task_for_message(message_id).await else { return };
    let reason = Some("the client cancelled".to_owned());
    if let Some(metadata) = lk_agents_core::a2a::rpc::cancellation(&row, reason) {
        let _ = lane.cancel_task(&row.task_id, metadata).await;
    }
}

fn resendable(error: &StorageError) -> bool {
    matches!(error, StorageError::Uncertain(_) | StorageError::Unavailable(_))
}

async fn wait(clock: &dyn Clock, at: Option<Ts>) {
    match at {
        Some(at) => clock.wait_until(at).await,
        None => std::future::pending().await,
    }
}

struct NoRefs;

impl RefRegistry for NoRefs {
    fn release(&self, _id: u64) {}
}

/// Simulates `scenario` on its own paused runtime and judges it.
pub fn run(scenario: Scenario) -> Outcome {
    quiet_fault_panics();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .start_paused(true)
        .build()
        .expect("a runtime");
    let recording = runtime.block_on(simulate(scenario.clone()));
    let (failures, expectations) = check_all(&recording.0);
    Outcome { scenario, recording: recording.0, failures, expectations, worker_deaths: recording.1 }
}

async fn simulate(scenario: Scenario) -> (TextRecording, u64) {
    let clock: Arc<dyn Clock> = Arc::new(lk_agents_runtime::net::clock::SimClock::new());
    let process = ProcessEnv {
        runtime: || {
            tokio::runtime::Handle::try_current()
                .map_err(|e| ProcessError::NoRuntime(e.to_string()))
        },
        spawner: || {
            let runtime = tokio::runtime::Handle::try_current()
                .map_err(|e| ProcessError::NoRuntime(e.to_string()))?;
            Ok(Arc::new(lk_agents_runtime::net::task::TokioSpawner::new(runtime)))
        },
        clock: clock.clone(),
        http: || Err(ProcessError::Start("the DST has no HTTP".into())),
        tracer: lk_agents_telemetry::Tracer::disabled(clock.clone()),
        tasks: Arc::default(),
    };
    let env = Arc::new(Env {
        clock: clock.clone(),
        refs: Arc::new(NoRefs),
        tracer: lk_agents_telemetry::Tracer::disabled(clock.clone()),
        process,
        scenario: scenario.clone(),
        calm: Default::default(),
        recording: Mutex::default(),
    });
    let mut watched = Vec::new();
    for cancel_after in &scenario.direct {
        let (finished, stop) = env.start_run(RunKind::Direct, None, *cancel_after);
        watched.push(tokio::spawn(async move {
            let _stop = stop;
            let _ = finished.await;
        }));
    }
    if let Some(cancel_after) = scenario.session {
        watched.push(tokio::spawn(env.clone().session(cancel_after)));
    }
    let store = Arc::new(Store::open(scenario.seed, scenario.storage_faults, clock.clone()).await);
    let clients: Vec<_> = (scenario.turns.iter().enumerate())
        .map(|(n, plan)| tokio::spawn(env.clone().client(store.clone(), n, *plan)))
        .collect();
    let mut turns = Vec::new();
    for client in clients {
        let deadline = Duration::from_millis(TURN_DEADLINE_MS);
        match tokio::time::timeout(deadline, client).await {
            Ok(Ok(turn)) => turns.push(turn),
            _ => turns.push(TurnRecord {
                message_id: String::new(),
                client: Client::Waits,
                answered: None,
            }),
        }
    }
    // Every run's client gives up by its deadline and the exit bound past it.
    let settle = Duration::from_millis(CLIENT_DEADLINE_MS + EXIT_BOUND_MS + 1_000);
    for watch in watched {
        let _ = tokio::time::timeout(settle, watch).await;
    }
    // Past the lease's TTL: a lease no live owner renews has lapsed.
    clock.wait_until(clock.now().saturating_add(Duration::from_secs(11))).await;
    if !turns.is_empty() {
        // The lane's next turn, on a healed database with answering fakes: whatever a client
        // left behind is settled by then at the latest.
        store.heal();
        env.calm.store(true, std::sync::atomic::Ordering::Release);
        let next = TurnPlan { start_ms: 0, client: Client::Waits, after_ms: 0, worker: 0 };
        let next = env.clone().client(store.clone(), turns.len(), next);
        if let Ok(turn) = tokio::time::timeout(Duration::from_millis(TURN_DEADLINE_MS), next).await
        {
            turns.push(turn);
        }
        clock.wait_until(clock.now().saturating_add(Duration::from_secs(11))).await;
    }
    let mut recording = std::mem::take(&mut *env.recording.lock());
    if !turns.is_empty() {
        settle_lane(&store, &mut recording, &turns).await;
    }
    recording.turns = turns;
    let deaths = store.worker_deaths();
    store.close().await;
    (recording, deaths)
}

/// What the lane holds once every turn ended.
async fn settle_lane(store: &Store, recording: &mut TextRecording, turns: &[TurnRecord]) {
    let thread = store.clean_thread().await;
    let lane = thread.lane(LANE);
    let runs = lane.runs(usize::MAX).await.unwrap_or_default();
    let unended = runs.iter().filter(|run| run.interrupted()).count() as u32;
    let held = store.held(LANE).await;
    recording.lane = Some(LaneRecord { unended, held, faulted: store.faulted() });
    let tasks = lane.tasks(None, 1_000).await.unwrap_or_default();
    let history = lane.history().await.unwrap_or_default();
    for turn in turns.iter().filter(|t| !t.message_id.is_empty()) {
        let mine = |id: &Option<String>| id.as_deref() == Some(turn.message_id.as_str());
        recording.messages.push(MessageRecord {
            message_id: turn.message_id.clone(),
            items: history.iter().filter(|row| mine(&row.message_id)).count() as u32,
            tasks: tasks.iter().filter(|row| mine(&row.message_id)).count() as u32,
        });
    }
    for task in tasks {
        let client =
            turns.iter().find(|t| task.message_id.as_deref() == Some(t.message_id.as_str()));
        recording.tasks.push(TaskRecord {
            task_id: task.task_id,
            status: task.status,
            left: client.is_some_and(|t| t.client != Client::Waits),
        });
    }
    thread.close().await;
}

/// A fake's deliberate panic is the scenario, not noise: its message is kept off stderr.
fn quiet_fault_panics() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let default = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            let message = info
                .payload()
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| info.payload().downcast_ref::<&str>().copied())
                .unwrap_or_default();
            if !message.starts_with("dst fault") {
                default(info);
            }
        }));
    });
}
