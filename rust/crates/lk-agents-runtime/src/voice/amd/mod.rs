//! Answering-machine detection runs on a voice session.
//!
//! [`VoiceSession::start_amd`] attaches one run to a running session (`AmdCommand::Attach`) and
//! starts its executor, which steps the pure [`lk_agents_voice::amd::machine::Amd`] and runs its
//! legs: the classifier ([`classifier`]), menu extraction, AMD's own STT ([`stt`]) and the wait
//! for the callee ([`callee`]). The run answers the session's reply gate through the session's
//! queue and hears the gate's observations on its own channel, each direction FIFO.
//!
//! [`AmdRun::outcome`] resolves with `amd_completed`'s outcome once the session detached the run
//! and every leg closed; [`AmdRun::close`] is `aclose()`.

pub mod callee;
pub mod classifier;
mod executor;
pub mod prompts;
mod stt;

pub use callee::CalleeSetup;
pub use classifier::{Choice, Classifier, DecisionModel};
pub(crate) use executor::AmdPeer;

use crate::voice::session::VoiceSession;
use executor::{Executor, Inputs, Legs};
use lk_agents_abi::voice::session::SessionRefused;
use lk_agents_abi::{ApiConnectOptions, Stt, ToolSpec};
use lk_agents_net::signal;
use lk_agents_voice::agent_session::{Answer, Command, VoiceCommandOrigin};
use lk_agents_voice::amd::events::{AmdEvent, Outcome};
use lk_agents_voice::amd::machine::{Amd, AmdSetup};
use lk_agents_voice::amd::options::{AmdOptions, AmdOptionsError};
use lk_agents_voice::amd::{AmdAttachRefused, AmdCommand, AmdRunId};
use std::sync::Arc;
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender, unbounded_channel};

/// One run to start: its options, its models and the participant it listens to.
pub struct AmdStart {
    /// The run's id, unique on the session: the host numbers its runs.
    pub run: AmdRunId,
    pub options: AmdOptions,
    pub classifier: Classifier,
    /// `llm_conn_options`: the classifier's and the menu's connection budget.
    pub llm_conn_options: ApiConnectOptions,
    /// AMD's own streaming STT, if any.
    pub stt: Option<Arc<dyn Stt>>,
    /// The room and participant to wait for; without a room the run listens at once.
    pub room: Option<Arc<dyn lk_agents_abi::voice::rtc::RtcRoom>>,
    pub participant: Option<String>,
    /// AMD's built-in `send_dtmf_events`, as the host declares it.
    pub dtmf_tool: ToolSpec,
    /// `AMDMenuObservedEvent.session_id`.
    pub session_id: String,
}

/// Why a run did not start.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum AmdStartRefused {
    #[error(transparent)]
    Options(#[from] AmdOptionsError),
    #[error(transparent)]
    Session(#[from] SessionRefused),
}

/// Why a run gave no outcome.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum AmdFailed {
    /// The session refused the attach (spec §5.1 entry checks).
    #[error(transparent)]
    Refused(#[from] AmdAttachRefused),
    /// The session or the run's task ended before the run completed.
    #[error("the session ended before AMD completed")]
    Ended,
}

/// A started run: `execute()` and `aclose()`, and its public events in order.
pub struct AmdRun {
    run: AmdRunId,
    close: UnboundedSender<()>,
    outcome: signal::Cell<Option<Result<Outcome, AmdFailed>>>,
    events: parking_lot::Mutex<Option<UnboundedReceiver<AmdEvent>>>,
}

impl AmdRun {
    pub fn run(&self) -> AmdRunId {
        self.run
    }

    /// `aclose()`: completes the run with `cancelled` if it still runs. Its outcome still waits
    /// for the session's `Detached` and the legs.
    pub fn close(&self) {
        // A run that already ended has nothing to close.
        let _ = self.close.send(());
    }

    /// `execute()`: the outcome, once the session detached the run and every leg closed.
    pub async fn outcome(&self) -> Result<Outcome, AmdFailed> {
        match self.outcome.wait().await {
            Ok(outcome) => outcome,
            Err(signal::Ended) => Err(AmdFailed::Ended),
        }
    }

    /// The outcome if the run ended.
    pub fn outcome_now(&self) -> Option<Result<Outcome, AmdFailed>> {
        self.outcome.get()
    }

    /// `amd_prediction`, `amd_menu_observed` and `amd_completed`, in order. Taken once.
    pub fn take_events(&self) -> Option<UnboundedReceiver<AmdEvent>> {
        self.events.lock().take()
    }
}

impl std::fmt::Debug for AmdRun {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AmdRun").field("run", &self.run).finish_non_exhaustive()
    }
}

impl VoiceSession {
    /// Attaches `start.run` to this running session and starts its executor beside the loop.
    /// The attach is a session command: the gate refuses it (a run attached, a handoff or task in
    /// flight, a realtime model, the close) and the outcome says so.
    pub fn start_amd(&self, start: AmdStart) -> Result<AmdRun, AmdStartRefused> {
        let AmdStart {
            run,
            options,
            classifier,
            llm_conn_options,
            stt,
            room,
            participant,
            dtmf_tool,
            session_id,
        } = start;
        let (peer, agent) = self.amd_peer()?;
        let setup = AmdSetup {
            agent,
            session_id,
            entered_at: peer.time.wall(),
            session_stt: self.legs().stt,
            amd_stt: stt.is_some(),
            extract_menus: classifier.extracts_menus(),
            // Realtime models are refused at attach.
            insert_user_items: false,
        };
        let callee =
            CalleeSetup { room, participant, wait_until_answered: options.wait_until_answered };
        let machine = Amd::new(run, options, setup)?;
        // Routed before the attach, so the observations the gate sends at attach are heard.
        let (observed, observations) = unbounded_channel();
        peer.route.attach(run, observed);
        let attached = self.ask(VoiceCommandOrigin::Host, pick_attach, |call| {
            Command::Amd(AmdCommand::Attach { run, dtmf_tool, call: Some(call) })
        });
        let attached = match attached {
            Ok(attached) => attached,
            Err(refused) => {
                peer.route.detach(run);
                return Err(refused.into());
            }
        };
        let (reports_tx, reports) = unbounded_channel();
        let (events_tx, events) = unbounded_channel();
        let (close, close_rx) = unbounded_channel();
        let (producer, outcome) = signal::once();
        let legs = Legs { classifier, llm_conn_options, stt, callee };
        let executor = Executor::new(machine, peer.clone(), legs, reports_tx, events_tx);
        let inputs = Inputs { observations, reports, close: close_rx };
        let route = peer.route.clone();
        let work = async move {
            match attached.await {
                Ok(Ok(())) => executor.run(inputs).await.ok_or(AmdFailed::Ended),
                Ok(Err(why)) => {
                    route.detach(run);
                    Err(AmdFailed::Refused(why))
                }
                Err(_) => {
                    route.detach(run);
                    Err(AmdFailed::Ended)
                }
            }
        };
        if !peer.tasks.spawn_settling(peer.spawner.as_ref(), "amd", producer, work) {
            peer.route.detach(run);
            return Err(SessionRefused::Closed.into());
        }
        Ok(AmdRun { run, close, outcome, events: parking_lot::Mutex::new(Some(events)) })
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
