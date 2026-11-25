//! The AMD executor: one run of answering-machine detection beside a voice session.
//!
//! The executor owns the run's pure [`Amd`] machine. It steps the machine on the session's
//! observations (`VoiceEffect::Amd`), its legs' results and its deadlines, and performs the
//! machine's effects: answers to the session's reply gate (`VoiceEvent::Amd`), classifier and menu
//! requests, deadlines and events. The legs are:
//!
//! - the classifier ([`classifier::classify`]): an LLM asked for one `record_result` call, or a
//!   [`DecisionModel`] asked for one choice;
//! - the menu extraction ([`classifier::extract_menu`]), for an LLM classifier only;
//! - AMD's own streaming STT ([`stt`]), fed the participant's real frames, whose finals go to the
//!   machine;
//! - the callee ([`callee`]): the wait for the participant's audio track and, for a SIP
//!   participant, the call answered.
//!
//! [`enter`] attaches a run to a session; [`AmdRun::execute`] returns its [`Outcome`] once the
//! session reported `Detached` and the legs closed. The run answers the session's reply gate
//! through the session's queue and hears the gate's observations on its own channel, each
//! direction FIFO.

pub mod callee;
pub mod classifier;
mod executor;
pub mod prompts;
mod stt;

pub use callee::CalleeSetup;
pub(crate) use executor::AmdPeer;

use crate::voice::VoiceSession;
use crate::voice::session::{Awaited, Call, Timebase, send_command};
use executor::{Executor, Inputs, Legs};
use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::telemetry::Cx;
use lk_agents_abi::voice::session::SessionRefused;
use lk_agents_abi::{
    ApiConnectOptions, BoxFuture, HOST_TEARDOWN_TIMEOUT, Llm, Spawner, Stt, ToolSpec,
};
use lk_agents_net::signal;
use lk_agents_net::task::TaskSet;
use lk_agents_voice::agent_session::{Answer, Command, VoiceCommandOrigin};
use lk_agents_voice::amd::events::{AmdEvent, Outcome};
use lk_agents_voice::amd::machine::{Amd, AmdSetup};
use lk_agents_voice::amd::options::{AmdOptions, AmdOptionsError};
use lk_agents_voice::amd::{AmdAttachRefused, AmdCommand, AmdObservation, AmdRunId};
use std::collections::BTreeMap;
use std::sync::Arc;
use tokio::sync::mpsc::{Receiver, UnboundedSender, unbounded_channel};
use tokio_util::sync::CancellationToken;

/// Closes a model AMD created; run once, when the run's legs close, bounded by the teardown
/// bound.
pub type ModelCloser = Box<dyn FnOnce() -> BoxFuture<'static, ()> + Send + Sync>;

/// One model a run uses, and what AMD knows of it beyond the model itself.
pub struct AmdModel<M: ?Sized> {
    pub model: Arc<M>,
    /// The model reaches only LiveKit Inference (a fallback chain only when every member does):
    /// its requests carry AMD's client feature header (spec §4.1).
    pub inference: bool,
    /// AMD created this model from a model string, so it closes it once its legs close (spec §14
    /// invariant 11). `None` for a model the application or the session supplied.
    pub close: Option<ModelCloser>,
}

impl<M: ?Sized> AmdModel<M> {
    /// A model the application or the session supplied: AMD never closes it.
    pub fn supplied(model: Arc<M>, inference: bool) -> Self {
        AmdModel { model, inference, close: None }
    }
}

/// What AMD classifies with.
pub enum Classifier {
    /// An LLM: one `record_result` call per request. It also extracts IVR menus.
    Llm(AmdModel<dyn Llm>),
    /// A decision model (`llm.DecisionModel`): one `choice` per request, and no menus.
    Decision(AmdModel<dyn DecisionModel>),
}

impl Classifier {
    /// Only an LLM reads menus.
    pub fn extracts_menus(&self) -> bool {
        matches!(self, Classifier::Llm(_))
    }

    /// The closer of the model AMD created, taken once.
    fn take_closer(&mut self) -> Option<ModelCloser> {
        match self {
            Classifier::Llm(model) => model.close.take(),
            Classifier::Decision(model) => model.close.take(),
        }
    }
}

impl std::fmt::Debug for Classifier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Classifier::Llm(m) => f.debug_tuple("Llm").field(&m.model.label()).finish(),
            Classifier::Decision(m) => f.debug_tuple("Decision").field(&m.model.label()).finish(),
        }
    }
}

/// A decision model's port (`llm.DecisionModel.evaluate` with one `Choice`).
pub trait DecisionModel: Send + Sync {
    /// The name logs give this instance.
    fn label(&self) -> String;

    /// Evaluates one choice: the chosen option, or `None` when the response holds no choice
    /// result for the decision.
    fn choose<'a>(
        &'a self,
        cx: &Cx,
        request: &'a ChoiceRequest,
    ) -> BoxFuture<'a, Result<Option<String>, ProviderError>>;
}

/// One `evaluate` call with a single `Choice` decision.
#[derive(Debug, Clone, PartialEq)]
pub struct ChoiceRequest {
    /// `evaluate(input=...)`: the request state and the transcript.
    pub input: String,
    /// The decision's name.
    pub name: String,
    /// The choice's prompt.
    pub prompt: String,
    /// The choice's options, in order. They carry no descriptions: a short definition next to a
    /// choice outweighed the prompt's conditions.
    pub options: Vec<String>,
    pub conn_options: ApiConnectOptions,
    /// `extra_headers`.
    pub extra_headers: BTreeMap<String, String>,
}

/// Everything one run needs beyond its session (`AMD(...)`'s arguments, resolved by the host).
pub struct AmdConfig {
    pub options: AmdOptions,
    pub classifier: Classifier,
    /// AMD's own streaming STT; `None` classifies on the session's transcripts alone.
    pub stt: Option<AmdModel<dyn Stt>>,
    /// The participant to listen to; `None` listens to the one the session's room I/O links.
    pub participant: Option<String>,
    /// AMD's built-in `send_dtmf_events`, as the host declares it.
    pub dtmf_tool: ToolSpec,
    /// The session's `llm_conn_options`: classifier and menu requests use them.
    pub llm_conn_options: ApiConnectOptions,
    /// The session's `stt_conn_options`: AMD's STT stream uses them.
    pub stt_conn_options: ApiConnectOptions,
}

/// Why a run did not attach.
#[derive(Debug, thiserror::Error)]
pub enum AmdEnterError {
    #[error(transparent)]
    Options(#[from] AmdOptionsError),
    #[error("amd requires a streaming STT")]
    NotStreaming,
    /// A realtime model with neither session STT nor AMD's own: AMD would classify nothing.
    #[error("amd requires session STT or AMD STT with realtime models")]
    RealtimeWithoutStt,
    /// The session refused the attach (spec §5.1 entry checks).
    #[error(transparent)]
    Refused(#[from] AmdAttachRefused),
    #[error(transparent)]
    Session(#[from] SessionRefused),
}

/// The run's events the host has not read yet. Beyond this the newest is dropped and logged:
/// a run emits one prediction per turn, a menu at most per turn and one completion, so only a
/// host that never reads them fills it. `execute()` still returns the outcome.
pub const EVENTS_CAPACITY: usize = 128;

/// One run, from its attach to its outcome. Dropping it does not cancel the run.
pub struct AmdRun {
    run: AmdRunId,
    close: CancellationToken,
    /// Ends without a value if the run's task ended without an outcome.
    outcome: signal::Cell<Option<Outcome>>,
    events: parking_lot::Mutex<Option<Receiver<AmdEvent>>>,
}

impl AmdRun {
    /// The run's id, as the session's reply gate knows it.
    pub fn id(&self) -> AmdRunId {
        self.run
    }

    /// The run's events from its attach, in order: each `amd_prediction` and
    /// `amd_menu_observed`, then `amd_completed` once. Taken once; `None` after.
    pub fn events(&self) -> Option<Receiver<AmdEvent>> {
        self.events.lock().take()
    }

    /// `execute()`: the outcome, once the session detached and the legs closed. Every caller gets
    /// the same outcome; a caller that stops waiting does not cancel the run.
    pub fn execute(&self) -> BoxFuture<'static, Outcome> {
        let outcome = self.outcome.clone();
        Box::pin(
            async move { outcome.wait().await.unwrap_or_else(|signal::Ended| executor::lost()) },
        )
    }

    /// `aclose()`: completes the run with `cancelled` if it is still running, and returns its
    /// outcome once completed.
    pub fn close(&self) -> BoxFuture<'static, Outcome> {
        // A run that already ended has nothing to close.
        self.close.cancel();
        self.execute()
    }
}

impl std::fmt::Debug for AmdRun {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AmdRun").field("run", &self.run).finish_non_exhaustive()
    }
}

/// The models AMD created, closed once: by the run after its legs closed, by a refused entry
/// before it reports the refusal, or, if dropped first (an entry the session would not run),
/// as a cleanup on the session's AMD task set, which the session's close joins.
pub(crate) struct OwnedModels {
    closers: Vec<ModelCloser>,
    time: Timebase,
    cleanup: Option<(Arc<TaskSet>, Arc<dyn Spawner>)>,
}

impl OwnedModels {
    /// Closes every model now, each bounded by the teardown bound.
    pub(crate) async fn close(&mut self) {
        close_models(std::mem::take(&mut self.closers), &self.time).await;
    }
}

impl Drop for OwnedModels {
    fn drop(&mut self) {
        let closers = std::mem::take(&mut self.closers);
        if closers.is_empty() {
            return;
        }
        let Some((tasks, spawner)) = &self.cleanup else { return };
        let time = self.time.clone();
        let close = async move { close_models(closers, &time).await };
        tasks.cleanup(spawner.as_ref(), "amd models close", close);
    }
}

/// Each close is bounded by [`HOST_TEARDOWN_TIMEOUT`]; one that overruns is logged and dropped.
async fn close_models(closers: Vec<ModelCloser>, time: &Timebase) {
    for close in closers {
        if time.timeout(HOST_TEARDOWN_TIMEOUT, close()).await.is_err() {
            log::warn!(target: lk_agents_abi::LOG_TARGET, timeout_ms = HOST_TEARDOWN_TIMEOUT.as_millis() as u64; "AMD model close timed out");
        }
    }
}

/// `async with AMD(session, ...)`: attaches a new run to `session` and starts its executor.
/// Returns once the session accepted the attach; the run then listens as its callee leg allows.
/// A refused run closes the models AMD created before this returns, as `__aenter__` does.
///
/// One task on the session's AMD task set owns the run from here: it routes the run, sends the
/// attach, reads the answer and runs the executor. A caller that stops waiting therefore leaves
/// nothing behind: an attach the session accepts after that is closed with `cancelled`, as
/// `__aenter__` closes a run it could not finish entering.
pub async fn enter(session: &VoiceSession, config: AmdConfig) -> Result<AmdRun, AmdEnterError> {
    enter_with(session, config, None).await
}

/// [`enter`], with `callee` in place of the session's callee setup when given.
async fn enter_with(
    session: &VoiceSession,
    mut config: AmdConfig,
    callee: Option<CalleeSetup>,
) -> Result<AmdRun, AmdEnterError> {
    let mut closers: Vec<ModelCloser> = config.classifier.take_closer().into_iter().collect();
    closers.extend(config.stt.as_mut().and_then(|stt| stt.close.take()));
    let prepared = match prepare(session, config, callee) {
        Ok(prepared) => prepared,
        Err(refused) => {
            close_models(closers, session.time()).await;
            return Err(refused);
        }
    };
    let Prepared { peer, run, machine, legs, dtmf_tool } = prepared;
    let tasks = Arc::clone(peer.route.tasks());
    let mut models = OwnedModels {
        closers,
        time: peer.time.clone(),
        cleanup: Some((Arc::clone(&tasks), Arc::clone(&peer.spawner))),
    };
    let (call, attached) = match session.call(pick_attach) {
        Ok(call) => call,
        Err(refused) => {
            models.close().await;
            return Err(refused.into());
        }
    };
    let (observed, observations) = unbounded_channel();
    let (reports_tx, reports) = unbounded_channel();
    let (events_tx, events) = tokio::sync::mpsc::channel(EVENTS_CAPACITY);
    let close = CancellationToken::new();
    let (producer, outcome) = signal::once();
    let (entered_tx, entered) = tokio::sync::oneshot::channel();
    let executor =
        Executor::new(machine, peer.clone(), Legs { models, ..legs }, reports_tx, events_tx);
    let entry = Entry {
        run,
        peer: peer.clone(),
        observed,
        dtmf_tool,
        call,
        attached,
        executor,
        inputs: Inputs { observations, reports, close: close.clone() },
        entered: entered_tx,
        outcome: producer,
    };
    if !tasks.try_spawn(peer.spawner.as_ref(), "amd", entry.run()) {
        // Refused: the entry was dropped unsent, and its models close on the task set.
        return Err(SessionRefused::Closed.into());
    }
    // A caller that stops waiting from here closes the run: before the answer, the run's task
    // closes it once the session accepts; after, nobody would hold it.
    let abandoned = close.clone().drop_guard();
    match entered.await {
        Ok(Ok(())) => {
            abandoned.disarm();
            let events = parking_lot::Mutex::new(Some(events));
            Ok(AmdRun { run, close, outcome, events })
        }
        Ok(Err(refused)) => Err(refused),
        // The run's task ended before the session answered: the session is gone.
        Err(_) => Err(SessionRefused::Closed.into()),
    }
}

/// What `enter` checked and built before the run's task owns it.
struct Prepared {
    peer: AmdPeer,
    run: AmdRunId,
    machine: Amd,
    legs: Legs,
    dtmf_tool: ToolSpec,
}

fn prepare(
    session: &VoiceSession,
    config: AmdConfig,
    callee: Option<CalleeSetup>,
) -> Result<Prepared, AmdEnterError> {
    let AmdConfig {
        options,
        classifier,
        stt,
        participant,
        dtmf_tool,
        llm_conn_options,
        stt_conn_options,
    } = config;
    // `AMD.__init__` checks the options, then the STT.
    options.validate()?;
    if stt.as_ref().is_some_and(|stt| !stt.model.capabilities().streaming) {
        return Err(AmdEnterError::NotStreaming);
    }
    let (session_stt, realtime) = (session.legs().stt, session.amd_realtime());
    if realtime.is_some() && !session_stt && stt.is_none() {
        return Err(AmdEnterError::RealtimeWithoutStt);
    }
    let (peer, agent) = session.amd_peer()?;
    let run = peer.route.next_run();
    let setup = AmdSetup {
        agent,
        session_id: format!("amd_{}", run.0),
        entered_at: peer.time.wall(),
        session_stt,
        amd_stt: stt.is_some(),
        extract_menus: classifier.extracts_menus(),
        // The session's record has no words for the turn otherwise (spec §7.2).
        insert_user_items: realtime.is_some_and(|caps| !caps.user_transcription) && !session_stt,
    };
    let callee = callee.unwrap_or_else(|| {
        let room = session.amd_room();
        // The participant the session's room I/O links, waited for when none is given
        // (`_wait_for_linked_participant`).
        let linked = room.as_ref().map(|(_, room_io)| {
            let room_io = Arc::clone(room_io);
            Arc::new(move || {
                let room_io = Arc::clone(&room_io);
                Box::pin(async move { room_io.wait_linked_participant().await.map(|id| id.0) })
                    as BoxFuture<'static, Option<String>>
            }) as callee::LinkedWait
        });
        CalleeSetup {
            room: room.map(|(room, _)| room),
            participant,
            linked,
            wait_until_answered: options.wait_until_answered,
        }
    });
    let machine = Amd::new(run, options, setup)?;
    // AMD's client feature header reaches an Inference STT's every connection (`_stt.py` passes
    // `client_feature_kwargs`); another STT takes none.
    let stt = stt.map(|stt| match stt.inference {
        true => {
            stt.model.with_extra_headers(&classifier::feature_headers(true)).unwrap_or(stt.model)
        }
        false => stt.model,
    });
    let legs = Legs {
        classifier: Arc::new(classifier),
        llm_conn_options,
        stt,
        stt_conn_options,
        callee,
        models: OwnedModels { closers: Vec::new(), time: peer.time.clone(), cleanup: None },
    };
    Ok(Prepared { peer, run, machine, legs, dtmf_tool })
}

/// The run's task: everything one run owns, from its attach to its outcome.
struct Entry {
    run: AmdRunId,
    peer: AmdPeer,
    observed: UnboundedSender<AmdObservation>,
    dtmf_tool: ToolSpec,
    call: Call,
    attached: Awaited<Result<(), AmdAttachRefused>>,
    executor: Executor,
    inputs: Inputs,
    entered: tokio::sync::oneshot::Sender<Result<(), AmdEnterError>>,
    outcome: signal::Producer<Option<Outcome>>,
}

impl Entry {
    async fn run(self) {
        let Entry {
            run,
            peer,
            observed,
            dtmf_tool,
            call,
            attached,
            executor,
            inputs,
            entered,
            outcome,
        } = self;
        // Routed before the attach, so the observations the gate sends at attach are heard.
        // Routes are keyed by run: a refused run never displaces the one attached.
        peer.route.attach(run, observed);
        let attach = Command::Amd(AmdCommand::Attach { run, dtmf_tool, call: Some(call.id) });
        let answer = match send_command(&peer.api, VoiceCommandOrigin::Host, attach, Some(call)) {
            Ok(()) => attached.await.ok(),
            Err(_) => None,
        };
        let refused = match answer {
            Some(Ok(())) => None,
            Some(Err(why)) => Some(AmdEnterError::from(why)),
            None => Some(SessionRefused::Closed.into()),
        };
        if let Some(refused) = refused {
            peer.route.detach(run);
            // `__aenter__` closes the models it created before it raises.
            executor.close_models().await;
            let _ = entered.send(Err(refused));
            return;
        }
        // A caller that stopped waiting has cancelled `inputs.close`: the run then completes
        // with `cancelled`, as `__aenter__` closes a run it could not finish entering.
        let _ = entered.send(Ok(()));
        outcome.resolve(executor.run(inputs).await);
    }
}

fn pick_attach(answer: Answer) -> Option<Result<(), AmdAttachRefused>> {
    match answer {
        Answer::AmdAttach(result) => Some(result),
        _ => None,
    }
}

#[cfg(test)]
#[path = "amd_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "executor_tests.rs"]
mod executor_tests;
