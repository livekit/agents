//! Each live option takes its value from the topmost `override_options` layer that sets it, else
//! from the active agent's `turn_handling`, else from the session's `update_options`, else from
//! what the session started with.
//!
//! voice-session.md "Temporary overrides". The oracle rebuilds the layers from the host's own
//! calls, read off the inputs and never off the core's layers: each accepted `Configure`,
//! `Override` and `ReleaseOverride` applies when its command enters the loop, and each install of
//! an agent (`AgentUpdated`) brings the `turn_handling` its handoff asked for, in order; a
//! `VoiceTask`'s agent sets none, and a task's end brings its parent's back. The bottom layer is
//! the first `ConfigApplied`, the session as it started. Every later `ConfigApplied` is judged on
//! the endpointing delays and the turn detection, the options a host layers; one an agent install
//! brings is judged at that install.
use std::time::Duration;

use lk_agents_voice::agent_activity::options::{AgentSessionOptions, ConfigLayer, TurnDetection};
use lk_agents_voice::agent_session::Origin;
use lk_agents_voice::trace::{CoreStep, HandoffStep, Step};

use crate::{
    authored::Authored,
    check::{Check, Contract, Rule, Subject},
    drive::{Case, Setup},
    input::{AgentText, AgentTurn, Input, Knob, Millis, TurnMode},
    record::{Answer, Observed, Recording},
};

pub struct OptionComesFromItsLayer;

impl Rule for OptionComesFromItsLayer {
    const CONTRACT: Contract = Contract {
        id: "option_comes_from_its_layer",
        guarantee: "A live option is the topmost override's that sets it, else the active agent's, else the session's.",
        examples: &[override_over_agent_over_session],
        counterexamples: &[agent_over_override, released_override_still_set],
        probe: Some(layered_endpointing),
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        let inputs = r.inputs_view();
        let mut taken = inputs.accepted().filter_map(|sent| match &sent.input {
            Input::Configure { knob, mode, ms, .. } => {
                Some(Taken::Configure(said(*knob, *mode, *ms)))
            }
            Input::Override { knob, mode, ms } => Some(Taken::Override(said(*knob, *mode, *ms))),
            Input::ReleaseOverride { nth } => Some(Taken::Release(nth.0 as usize)),
            _ => None,
        });
        // The layer each handoff installs, in the order the loop takes them; a refused one
        // installs nothing.
        let mut handoffs = (r.host_view().api_calls().into_iter()).filter_map(|call| {
            match (call.input(), call.returned()) {
                (_, Some((_, Answer::HandoffRefused))) => None,
                (Input::Handoff { turn, .. }, _) => {
                    Some(turn.map_or_else(Leaves::default, |t| said(t.knob, t.mode, t.ms)))
                }
                _ => None,
            }
        });
        let (mut started, mut session, mut agent) = (None, Leaves::default(), Leaves::default());
        let mut overrides: Vec<Leaves> = Vec::new();
        // The agents a `VoiceTask` suspended, innermost last, and the layer a task's own install
        // brings: its agent's (none), or on its end its parent's again.
        let (mut parents, mut task_install) = (Vec::new(), None);
        // A configuration an agent install brings, judged once the install says which agent.
        let mut installing: Option<(Observed, Leaves)> = None;
        let mut judge = |at: Observed,
                         actual: Leaves,
                         layers: [&Leaves; 3],
                         overrides: &[Leaves]| {
            let expected = (overrides.iter().rev())
                .chain(layers)
                .fold(Leaves::default(), |acc, below| acc.over(*below));
            check.expect(
                Subject::Session,
                expected == actual,
                format!("at {at:?} the layers set {expected:?}, the session resolved {actual:?}"),
            );
        };
        for (at, step) in r.journal() {
            match step {
                Step::Handoff(HandoffStep::TaskEntered { .. }) => {
                    parents.push(agent);
                    task_install = Some(Leaves::default());
                }
                Step::Handoff(HandoffStep::TaskEnded { .. }) => task_install = parents.pop(),
                Step::Handoff(HandoffStep::AgentUpdated { .. }) => {
                    agent = task_install.take().or_else(|| handoffs.next()).unwrap_or_default();
                    if let (Some((at, actual)), Some(started)) = (installing.take(), &started) {
                        judge(at, actual, [&agent, &session, started], &overrides);
                    }
                }
                Step::Core(CoreStep::CommandReceived {
                    origin: Origin::Host,
                    kind:
                        "TurnCommand::Configure"
                        | "TurnCommand::Override"
                        | "TurnCommand::ReleaseOverride",
                }) => match taken.next() {
                    Some(Taken::Configure(layer)) => session = layer.over(session),
                    Some(Taken::Override(layer)) => overrides.push(layer),
                    Some(Taken::Release(nth)) if !overrides.is_empty() => {
                        overrides.remove(nth % overrides.len());
                    }
                    _ => {}
                },
                Step::Core(CoreStep::ConfigApplied { layer, effective }) => {
                    let actual = Leaves::of(effective);
                    match &started {
                        None => started = Some(actual),
                        Some(_) if *layer == ConfigLayer::Agent => installing = Some((*at, actual)),
                        Some(started) => {
                            judge(*at, actual, [&agent, &session, started], &overrides)
                        }
                    }
                }
                _ => {}
            }
        }
    }
}

/// What a host call says for `knob`, as voice-session.md and the reference define it: a turn
/// detection mode (`None` reverting to automatic selection) or an endpointing delay. The other
/// options are not judged.
fn said(knob: Knob, mode: TurnMode, ms: Millis) -> Leaves {
    let delay = Some(Duration::from_millis(ms.0));
    match knob {
        Knob::Detection => Leaves {
            detection: Some(match mode {
                TurnMode::Default => TurnDetection::Auto,
                TurnMode::Stt => TurnDetection::Stt,
                TurnMode::Manual => TurnDetection::Manual,
            }),
            ..Leaves::default()
        },
        Knob::MinEndpointing => Leaves { min_delay: delay, ..Leaves::default() },
        Knob::MaxEndpointing => Leaves { max_delay: delay, ..Leaves::default() },
        _ => Leaves::default(),
    }
}

/// A host call that moves a layer, in the order the loop takes them.
enum Taken {
    Configure(Leaves),
    Override(Leaves),
    Release(usize),
}

/// The options a host layers, each `None` where the layer leaves it unsaid.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Leaves {
    detection: Option<TurnDetection>,
    min_delay: Option<Duration>,
    max_delay: Option<Duration>,
}

impl Leaves {
    fn of(options: &AgentSessionOptions) -> Self {
        let turn = &options.turn;
        Leaves {
            detection: turn.detection,
            min_delay: turn.endpointing.min_delay,
            max_delay: turn.endpointing.max_delay,
        }
    }

    /// `self` laid on `below`: each said option wins.
    fn over(self, below: Leaves) -> Leaves {
        Leaves {
            detection: self.detection.or(below.detection),
            min_delay: self.min_delay.or(below.min_delay),
            max_delay: self.max_delay.or(below.max_delay),
        }
    }
}

fn min_delay(ms: u64) -> Input {
    Input::Configure {
        knob: Knob::MinEndpointing,
        mode: TurnMode::Default,
        ms: Millis(ms),
        on: false,
        words: crate::input::Count(0),
    }
}

fn override_min_delay(ms: u64) -> Input {
    Input::Override { knob: Knob::MinEndpointing, mode: TurnMode::Default, ms: Millis(ms) }
}

fn handoff_min_delay(ms: u64) -> Input {
    Input::Handoff {
        agent: "billing".into(),
        instructions: AgentText("Take the payment.".into()),
        tools: Vec::new(),
        turn: Some(AgentTurn {
            knob: Knob::MinEndpointing,
            mode: TurnMode::Default,
            ms: Millis(ms),
        }),
    }
}

fn received(kind: &'static str) -> Step {
    Step::Core(CoreStep::CommandReceived { origin: Origin::Host, kind })
}

fn applied(layer: ConfigLayer, min_ms: u64) -> Step {
    let mut effective = AgentSessionOptions::default();
    effective.turn.endpointing.min_delay = Some(Duration::from_millis(min_ms));
    Step::Core(CoreStep::ConfigApplied { layer, effective: Box::new(effective) })
}

fn installed(agent: &str) -> Step {
    Step::Handoff(HandoffStep::AgentUpdated {
        agent: agent.into(),
        tools: 0,
        chat_ctx: false,
        reopened: "",
        legs: Default::default(),
    })
}

/// The session says 0.9 s, the agent 0.7 s, then an override 0.3 s: each new layer shows on top,
/// and releasing the override shows the agent's again.
fn layered(agent_min_ms: u64, override_shows: u64) -> Recording {
    Authored::new()
        .journal(applied(ConfigLayer::Session, 500))
        .sent(min_delay(900))
        .journal(received("TurnCommand::Configure"))
        .journal(applied(ConfigLayer::Session, 900))
        .api_call(handoff_min_delay(agent_min_ms))
        .journal(received("ActivityCommand::Update"))
        .journal(applied(ConfigLayer::Agent, agent_min_ms))
        .journal(installed("billing"))
        .returns(Answer::HandoffOk)
        .sent(override_min_delay(300))
        .journal(received("TurnCommand::Override"))
        .journal(applied(ConfigLayer::Override, override_shows))
        .sent(Input::ReleaseOverride { nth: crate::input::Count(0) })
        .journal(received("TurnCommand::ReleaseOverride"))
        .journal(applied(ConfigLayer::Override, agent_min_ms))
        .done()
}

fn override_over_agent_over_session() -> Recording {
    layered(700, 300)
}

/// The agent's layer resolved above the override.
fn agent_over_override() -> Recording {
    layered(700, 700)
}

/// An override released with no other layer setting its option: the option stayed the override's
/// instead of going back to what the session started with.
fn released_override_still_set() -> Recording {
    Authored::new()
        .journal(applied(ConfigLayer::Session, 500))
        .sent(override_min_delay(300))
        .journal(received("TurnCommand::Override"))
        .journal(applied(ConfigLayer::Override, 300))
        .sent(Input::ReleaseOverride { nth: crate::input::Count(0) })
        .journal(received("TurnCommand::ReleaseOverride"))
        .journal(applied(ConfigLayer::Override, 300))
        .done()
}

fn layered_endpointing() -> Case {
    Case {
        setup: Setup { stt: true, vad: true, ..Setup::default() },
        inputs: vec![
            min_delay(900),
            handoff_min_delay(700),
            Input::Advance { ms: Millis(500) },
            override_min_delay(300),
            Input::Advance { ms: Millis(500) },
            Input::ReleaseOverride { nth: crate::input::Count(0) },
            Input::Advance { ms: Millis(500) },
        ],
        ..Case::default()
    }
}
