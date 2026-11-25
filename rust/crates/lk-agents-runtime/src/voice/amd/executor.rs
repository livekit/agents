//! The AMD executor: one [`Amd`] machine per run, stepped in its own task on the session's
//! observations, its legs' results and its deadlines, with every [`AmdEffect`] executed.
//!
//! Each direction of the protocol is FIFO: observations arrive on one channel in the order the
//! session emitted them, and answers go onto the session's queue in the order the machine sent
//! them. The two directions interleave freely; the protocol needs nothing more.
//!
//! Teardown: once the machine completes (its `Detach` already sent), every leg is stopped and
//! joined, then the machine hears `LegsClosed`. `amd_completed` comes after both that and the
//! session's `Detached`, in either order.

use super::callee::{self, CalleeSetup};
use super::classifier;
use super::stt;
use super::{Classifier, ModelCloser};
use crate::telemetry;
use crate::voice::legs::OpeningLeg;
use crate::voice::resources::Resources;
use crate::voice::session::{AmdTap, Journal, Timebase, Tx};
use lk_agents_abi::resources::ResourceId;
use lk_agents_abi::{ApiConnectOptions, Spawner, Stt, TimerSet};
use lk_agents_net::owned::Background;
use lk_agents_net::task::{ScopedTask, TaskSet};
use lk_agents_voice::agent_session::VoiceEvent;
use lk_agents_voice::amd::effects::{AmdEffect, AmdTimer, ClassifyRequest, MenuRequest, ReqId};
use lk_agents_voice::amd::events::{AmdEvent, AmdReason, Outcome, Stage};
use lk_agents_voice::amd::machine::{Amd, Callee, In, Lifecycle};
use lk_agents_voice::amd::{AmdAnswer, AmdObservation, AmdRunId};
use lk_agents_voice::audio::AudioChannel;
use lk_agents_voice::data_plane::AudioInputRouterCommand;
use lk_agents_voice::trace::AmdStep;
use std::sync::Arc;
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};
use tokio_util::sync::CancellationToken;

/// What a leg tells the executor.
pub(crate) enum LegReport {
    /// The classifier's answer to `req`: `In::Classified` or `In::LegFailed`.
    Classified {
        req: ReqId,
        input: In,
    },
    /// The menu extraction's answer to `req`: `In::MenuExtracted`.
    Menu {
        req: ReqId,
        input: In,
    },
    /// AMD's STT stream opened; its owner closes it if nobody adopts it.
    SttOpened(Box<OpeningLeg>),
    AmdFinal(String),
    /// AMD's STT will not open, failed or ended.
    SttFailed,
    Callee(Callee),
}

/// What the run executes against, taken from the running session.
#[derive(Clone)]
pub(crate) struct AmdPeer {
    pub api: Tx,
    pub journal: Journal,
    pub time: Timebase,
    pub spawner: Arc<dyn Spawner>,
    pub tasks: Arc<TaskSet>,
    pub resources: Resources,
    pub cx: telemetry::Cx,
    pub route: AmdTap,
}

/// The models and the callee a run uses.
pub(crate) struct Legs {
    pub classifier: Arc<Classifier>,
    pub llm_conn_options: ApiConnectOptions,
    pub stt: Option<Arc<dyn Stt>>,
    pub stt_conn_options: ApiConnectOptions,
    pub callee: CalleeSetup,
    /// The models AMD created, closed once every leg ended and before the outcome.
    pub closers: Vec<ModelCloser>,
}

/// AMD's own STT, from listening to teardown.
enum SttLeg {
    /// No STT of AMD's own, or not listening yet.
    Idle,
    Opening,
    /// Its sink is on the router's AMD channel.
    Installed(ResourceId),
    /// Failed or stopped.
    Done,
}

/// Where the run is in its teardown.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Teardown {
    Running,
    /// The machine completed; the legs are stopping.
    Joining,
    /// The machine heard `LegsClosed`.
    Joined,
}

pub(crate) struct Executor {
    machine: Amd,
    run: AmdRunId,
    peer: AmdPeer,
    legs: Legs,
    timers: TimerSet<AmdTimer>,
    /// Every leg's task: the run joins them before `LegsClosed`.
    leg_tasks: Arc<TaskSet>,
    /// The parent of every leg's cancellation; dropped with the executor, it stops them all.
    _leg_cancel: tokio_util::sync::DropGuard,
    leg_token: CancellationToken,
    reports: UnboundedSender<LegReport>,
    /// The classification in flight.
    classify: Option<(ReqId, CancellationToken)>,
    /// The menu extraction in flight.
    menu: Option<(ReqId, CancellationToken)>,
    stt: SttLeg,
    callee: Option<ScopedTask>,
    teardown: Teardown,
    outcome: Option<Outcome>,
    events: UnboundedSender<AmdEvent>,
}

/// The ends the executor's loop reads.
pub(crate) struct Inputs {
    pub observations: UnboundedReceiver<AmdObservation>,
    pub reports: UnboundedReceiver<LegReport>,
    pub close: UnboundedReceiver<()>,
}

impl Executor {
    pub(crate) fn new(
        machine: Amd,
        peer: AmdPeer,
        legs: Legs,
        reports: UnboundedSender<LegReport>,
        events: UnboundedSender<AmdEvent>,
    ) -> Self {
        let leg_token = CancellationToken::new();
        Executor {
            run: machine.run(),
            machine,
            peer,
            legs,
            timers: TimerSet::new(),
            leg_tasks: Arc::default(),
            _leg_cancel: leg_token.clone().drop_guard(),
            leg_token,
            reports,
            classify: None,
            menu: None,
            stt: SttLeg::Idle,
            callee: None,
            teardown: Teardown::Running,
            outcome: None,
            events,
        }
    }

    /// Runs the run to its outcome. A session that went away first completes it as its close
    /// would.
    pub(crate) async fn run(mut self, mut inputs: Inputs) -> Outcome {
        self.start_callee();
        let mut host_open = true;
        let mut session_open = true;
        loop {
            if let Some(outcome) = self.outcome.take() {
                self.peer.route.detach(self.run);
                return outcome;
            }
            let now = self.peer.time.now();
            let deadline = self.timers.next_deadline();
            if deadline.is_some_and(|at| at <= now) {
                self.fire(now);
                continue;
            }
            let joining = self.teardown == Teardown::Joining;
            let clock = self.peer.time.clock();
            let leg_tasks = Arc::clone(&self.leg_tasks);
            let api = self.peer.api.clone();
            tokio::select! {
                biased;
                () = self.peer.time.wait_until_opt(deadline) => self.fire(self.peer.time.now()),
                observation = inputs.observations.recv() => match observation {
                    Some(observation) => self.step(In::Observed(observation)),
                    // The session dropped the route: it is gone.
                    None => break,
                },
                report = inputs.reports.recv() => {
                    if let Some(report) = report {
                        self.on_report(report);
                    }
                }
                closed = inputs.close.recv(), if host_open => match closed {
                    Some(()) => self.step(In::Close),
                    None => host_open = false,
                },
                _ = leg_tasks.wait_empty(clock.as_ref(), None), if joining => {
                    self.legs_closed().await;
                }
                () = api.closed(), if session_open => {
                    session_open = false;
                    // Observations already queued come first; then the session is gone.
                    while let Ok(observation) = inputs.observations.try_recv() {
                        self.step(In::Observed(observation));
                    }
                    if self.outcome.is_none() {
                        self.step(In::Observed(AmdObservation::Detached { run: self.run }));
                    }
                }
            }
        }
        self.peer.route.detach(self.run);
        // The session is gone before `Detached`: the machine completes as if it had sent it.
        self.step(In::Observed(AmdObservation::Detached { run: self.run }));
        self.join_legs().await;
        // The machine completes once detached and joined; nothing else is left to wait for.
        self.outcome.take().unwrap_or_else(lost)
    }

    /// Joins the legs outside the loop, for a run whose session went away.
    async fn join_legs(&mut self) {
        if self.teardown == Teardown::Joining {
            self.leg_tasks.wait_empty(self.peer.time.clock().as_ref(), None).await;
            self.legs_closed().await;
        }
    }

    /// Every leg ended: the models AMD created close, then the machine hears `LegsClosed`
    /// (`_cleanup` closes them before it sets the completion).
    async fn legs_closed(&mut self) {
        for close in std::mem::take(&mut self.legs.closers) {
            close().await;
        }
        self.teardown = Teardown::Joined;
        self.step(In::LegsClosed);
    }

    /// Fires every deadline due at `now`, in order; one an earlier fire cleared is skipped.
    fn fire(&mut self, now: lk_agents_abi::Ts) {
        while let Some(timer) = self.timers.pop_due(now) {
            self.step(In::Timeout(timer));
        }
    }

    fn step(&mut self, input: In) {
        let now = self.peer.time.now();
        let effects = self.machine.step(now, input);
        for step in self.machine.take_trace() {
            self.peer.journal.record(step);
        }
        self.perform(effects);
        self.after_step();
    }

    fn perform(&mut self, effects: Vec<AmdEffect>) {
        for effect in effects {
            match effect {
                AmdEffect::Answer(answer) => self.answer(answer),
                AmdEffect::Classify(request) => self.start_classify(request),
                AmdEffect::CancelClassify => {
                    if let Some((req, cancel)) = self.classify.take() {
                        cancel.cancel();
                        self.peer.journal.record(AmdStep::RequestCancelled { run: self.run, req });
                    }
                }
                AmdEffect::ExtractMenu(request) => self.start_menu(request),
                AmdEffect::CancelMenu => {
                    if let Some((_, cancel)) = self.menu.take() {
                        cancel.cancel();
                    }
                }
                AmdEffect::SetDeadline(timer, at) => {
                    self.timers.set_deadline(timer, at);
                }
                AmdEffect::ClearDeadline(timer) => {
                    self.timers.clear_deadline(timer);
                }
                AmdEffect::Emit(event) => {
                    if let AmdEvent::Completed(outcome) = &event {
                        let (category, reason) = (outcome.category, outcome.reason);
                        self.peer.journal.record(AmdStep::Completed {
                            run: self.run,
                            category,
                            reason,
                        });
                        self.outcome = Some(outcome.clone());
                    }
                    // The host may have stopped reading its events.
                    let _ = self.events.send(event);
                }
            }
        }
    }

    /// An answer to the session, in sending order. A session gone refuses it; its close
    /// reports `Detached` or the queue's end tells the loop.
    fn answer(&mut self, answer: AmdAnswer) {
        let _ = self.peer.api.send(VoiceEvent::Amd(answer));
    }

    /// Opens AMD's STT once listening starts, and stops every leg once the run completed.
    fn after_step(&mut self) {
        match self.machine.lifecycle() {
            Lifecycle::Active => {
                if matches!(self.stt, SttLeg::Idle)
                    && let Some(model) = self.legs.stt.clone()
                {
                    self.stt = SttLeg::Opening;
                    self.open_stt(model, self.legs.stt_conn_options);
                }
            }
            Lifecycle::Finished(_) if self.teardown == Teardown::Running => self.stop_legs(),
            Lifecycle::Pending | Lifecycle::Finished(_) => {}
        }
    }

    /// Requests every leg's end; the loop joins them.
    fn stop_legs(&mut self) {
        self.teardown = Teardown::Joining;
        self.leg_token.cancel();
        self.classify = None;
        self.menu = None;
        self.callee = None;
        self.remove_stt();
    }

    fn remove_stt(&mut self) {
        if let SttLeg::Installed(sink) = std::mem::replace(&mut self.stt, SttLeg::Done) {
            let _ = self.peer.api.send(AudioInputRouterCommand::Remove(sink));
        } else {
            self.stt = SttLeg::Done;
        }
    }

    /// A leg's sender: `false` once the executor is gone.
    fn reporter(&self) -> impl Fn(LegReport) -> bool + Send + Sync + 'static {
        let reports = self.reports.clone();
        move |report| reports.send(report).is_ok()
    }

    fn start_callee(&mut self) {
        let report = self.reporter();
        let watch = callee::watch(
            self.legs.callee.clone(),
            self.peer.time.clone(),
            Box::new(move |callee| report(LegReport::Callee(callee))),
        );
        let spawner = Arc::clone(&self.peer.spawner);
        self.callee = self.leg_tasks.try_scoped(spawner.as_ref(), "amd callee", watch);
    }

    fn start_classify(&mut self, request: ClassifyRequest) {
        let req = request.req;
        self.peer.journal.record(AmdStep::RequestIssued { run: self.run, req, turn: request.turn });
        let cancel = self.leg_token.child_token();
        self.classify = Some((req, cancel.clone()));
        let report = self.reporter();
        let classifier = Arc::clone(&self.legs.classifier);
        let (conn, clock, cx) =
            (self.legs.llm_conn_options, self.peer.time.clock(), self.peer.cx.clone());
        let spawner = Arc::clone(&self.peer.spawner);
        self.leg_tasks.try_spawn(spawner.as_ref(), "amd classify", async move {
            let classified = classifier::classify(&classifier, &request, conn, clock, &cx, &cancel);
            if let Some(input) = classified.await {
                report(LegReport::Classified { req, input });
            }
        });
    }

    fn start_menu(&mut self, request: MenuRequest) {
        // A decision model never asks for a menu (`AmdSetup::extract_menus`).
        if !self.legs.classifier.extracts_menus() {
            return;
        }
        let req = request.req;
        let cancel = self.leg_token.child_token();
        self.menu = Some((req, cancel.clone()));
        let report = self.reporter();
        let classifier = Arc::clone(&self.legs.classifier);
        let (conn, clock, cx) =
            (self.legs.llm_conn_options, self.peer.time.clock(), self.peer.cx.clone());
        let spawner = Arc::clone(&self.peer.spawner);
        self.leg_tasks.try_spawn(spawner.as_ref(), "amd menu", async move {
            let Classifier::Llm(llm) = classifier.as_ref() else { return };
            let extracted = classifier::extract_menu(llm, &request, conn, clock, &cx, &cancel);
            if let Some(input) = extracted.await {
                report(LegReport::Menu { req, input });
            }
        });
    }

    fn open_stt(&mut self, model: Arc<dyn Stt>, conn_options: ApiConnectOptions) {
        let spawner = Arc::clone(&self.peer.spawner);
        let background =
            Background::new(Arc::clone(&self.leg_tasks), spawner.clone(), self.peer.time.clock());
        let work = stt::open(
            (model, conn_options),
            self.peer.cx.clone(),
            self.peer.time.clone(),
            self.peer.resources.clone(),
            background,
            self.leg_token.child_token(),
            self.reporter(),
        );
        self.leg_tasks.try_spawn(spawner.as_ref(), "amd stt open", work);
    }

    fn on_report(&mut self, report: LegReport) {
        match report {
            LegReport::Classified { req, input } => {
                if self.classify.as_ref().is_none_or(|(current, _)| *current != req) {
                    return;
                }
                self.classify = None;
                self.step(input);
            }
            LegReport::Menu { req, input } => {
                if self.menu.as_ref().is_some_and(|(current, _)| *current == req) {
                    self.menu = None;
                }
                self.step(input);
            }
            LegReport::SttOpened(leg) => self.adopt_stt(*leg),
            LegReport::AmdFinal(text) => self.step(In::AmdFinal(text)),
            LegReport::SttFailed => {
                if matches!(self.stt, SttLeg::Done) {
                    return;
                }
                self.remove_stt();
                self.step(In::SttFailed);
            }
            LegReport::Callee(callee) => self.step(In::Callee(callee)),
        }
    }

    /// The opened stream's sink goes on the router's AMD channel and its drain starts; a stream
    /// opened after the run stopped listening is closed.
    fn adopt_stt(&mut self, leg: OpeningLeg) {
        if !matches!(self.stt, SttLeg::Opening) {
            drop(leg);
            return;
        }
        let sink = leg.resource();
        let install = AudioInputRouterCommand::Install { channel: AudioChannel::Amd, sink };
        if self.peer.api.send(install).is_err() {
            self.stt = SttLeg::Done;
            drop(leg);
            return;
        }
        self.stt = SttLeg::Installed(sink);
        let work = stt::drain(leg, self.leg_token.child_token(), self.reporter());
        let spawner = Arc::clone(&self.peer.spawner);
        self.leg_tasks.try_spawn(spawner.as_ref(), "amd stt", work);
    }
}

/// The outcome of a run whose task ended without one: it failed, as a failed task ends a Python
/// run (`_on_task_done`).
pub(super) fn lost() -> Outcome {
    Outcome {
        category: Stage::Uncertain,
        reason: AmdReason::InternalError,
        voicemail_message_played: false,
        last_prediction: None,
    }
}
