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
use executor::{Executor, Inputs, Legs};
use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::telemetry::Cx;
use lk_agents_abi::voice::session::SessionRefused;
use lk_agents_abi::{ApiConnectOptions, BoxFuture, Llm, Stt, ToolSpec};
use lk_agents_net::signal;
use lk_agents_voice::agent_session::{Answer, Command, VoiceCommandOrigin};
use lk_agents_voice::amd::events::{AmdEvent, Outcome};
use lk_agents_voice::amd::machine::{Amd, AmdSetup};
use lk_agents_voice::amd::options::{AmdOptions, AmdOptionsError};
use lk_agents_voice::amd::{AmdAttachRefused, AmdCommand, AmdRunId};
use std::collections::BTreeMap;
use std::sync::Arc;
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender, unbounded_channel};

/// Closes a model AMD created; run once, when the run's legs close.
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
    /// The session refused the attach (spec §5.1 entry checks).
    #[error(transparent)]
    Refused(#[from] AmdAttachRefused),
    #[error(transparent)]
    Session(#[from] SessionRefused),
}

/// One run, from its attach to its outcome. Dropping it does not cancel the run.
pub struct AmdRun {
    run: AmdRunId,
    close: UnboundedSender<()>,
    /// Ends without a value if the run's task ended without an outcome.
    outcome: signal::Cell<Option<Outcome>>,
    events: parking_lot::Mutex<Option<UnboundedReceiver<AmdEvent>>>,
}

impl AmdRun {
    /// The run's id, as the session's reply gate knows it.
    pub fn id(&self) -> AmdRunId {
        self.run
    }

    /// The run's events from its attach, in order: each `amd_prediction` and
    /// `amd_menu_observed`, then `amd_completed` once. Taken once; `None` after.
    pub fn events(&self) -> Option<UnboundedReceiver<AmdEvent>> {
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
        let _ = self.close.send(());
        self.execute()
    }
}

impl std::fmt::Debug for AmdRun {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AmdRun").field("run", &self.run).finish_non_exhaustive()
    }
}

/// `async with AMD(session, ...)`: attaches a new run to `session` and starts its executor on the
/// session's scheduler. Returns once the session accepted the attach; the run then listens as its
/// callee leg allows. A refused run closes the models AMD created, as `__aenter__` does.
pub async fn enter(session: &VoiceSession, mut config: AmdConfig) -> Result<AmdRun, AmdEnterError> {
    let mut closers: Vec<ModelCloser> = config.classifier.take_closer().into_iter().collect();
    closers.extend(config.stt.as_mut().and_then(|stt| stt.close.take()));
    match attach(session, config, &mut closers).await {
        Ok(run) => Ok(run),
        Err(refused) => {
            for close in closers {
                close().await;
            }
            Err(refused)
        }
    }
}

async fn attach(
    session: &VoiceSession,
    config: AmdConfig,
    closers: &mut Vec<ModelCloser>,
) -> Result<AmdRun, AmdEnterError> {
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
    let (peer, agent) = session.amd_peer()?;
    let run = peer.route.next_run();
    let setup = AmdSetup {
        agent,
        session_id: format!("amd_{}", run.0),
        entered_at: peer.time.wall(),
        session_stt: session.legs().stt,
        amd_stt: stt.is_some(),
        extract_menus: classifier.extracts_menus(),
        // Realtime models are refused at attach.
        insert_user_items: false,
    };
    let room = session.amd_room();
    let callee = CalleeSetup {
        participant: participant.or_else(|| room.as_ref().and_then(|(_, linked)| linked.clone())),
        room: room.map(|(room, _)| room),
        wait_until_answered: options.wait_until_answered,
    };
    let machine = Amd::new(run, options, setup)?;
    // Routed before the attach, so the observations the gate sends at attach are heard.
    let (observed, observations) = unbounded_channel();
    peer.route.attach(run, observed);
    let attached = session.ask(VoiceCommandOrigin::Host, pick_attach, |call| {
        Command::Amd(AmdCommand::Attach { run, dtmf_tool, call: Some(call) })
    });
    let answer = match attached {
        Ok(attached) => attached.await,
        Err(refused) => {
            peer.route.detach(run);
            return Err(refused.into());
        }
    };
    match answer {
        Ok(Ok(())) => {}
        Ok(Err(why)) => {
            peer.route.detach(run);
            return Err(why.into());
        }
        Err(_) => {
            peer.route.detach(run);
            return Err(SessionRefused::Closed.into());
        }
    }
    let (reports_tx, reports) = unbounded_channel();
    let (events_tx, events) = unbounded_channel();
    let (close, close_rx) = unbounded_channel();
    let (producer, outcome) = signal::once();
    let legs = Legs {
        classifier: Arc::new(classifier),
        llm_conn_options,
        stt: stt.map(|stt| stt.model),
        stt_conn_options,
        callee,
        closers: std::mem::take(closers),
    };
    let executor = Executor::new(machine, peer.clone(), legs, reports_tx, events_tx);
    let inputs = Inputs { observations, reports, close: close_rx };
    if !peer.tasks.spawn_settling(peer.spawner.as_ref(), "amd", producer, executor.run(inputs)) {
        // The session is closing: its close detaches the run it just attached.
        peer.route.detach(run);
        return Err(SessionRefused::Closed.into());
    }
    Ok(AmdRun { run, close, outcome, events: parking_lot::Mutex::new(Some(events)) })
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
