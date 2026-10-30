//! Each live option takes its value from the topmost `override_options` layer that sets it, else
//! from the active agent's `turn_handling`, else from the session's `update_options`.
//!
//! voice-session.md "Temporary overrides". The layers are rebuilt from the host's own calls: each
//! accepted `Configure`, `Override` and `ReleaseOverride` is applied when its command enters the
//! loop. The agent layer is the `turn_handling` of each handoff the loop installs, in order; a
//! `VoiceTask`'s agent and the opening agent set none, and a task's end brings its parent's back.
//! Every `ConfigApplied` after the first is
//! judged on the endpointing delays and the turn detection, the options a host layers; an option no
//! layer set is the session's construction or a default, and is not judged.
use std::time::Duration;

use lk_agents_voice::agent_activity::options::{
    AgentSessionOptions, ConfigLayer, TurnDetection, update_layer,
};
use lk_agents_voice::agent_session::Origin;
use lk_agents_voice::trace::{CoreStep, HandoffStep, Step};

use crate::{
    authored::Authored,
    check::{Check, Contract, Rule, Subject},
    drive::{Case, Setup, agent_turn_handling, turn_update},
    input::{AgentText, AgentTurn, Input, Knob, Millis, TurnMode},
    record::{Answer, Recording},
};

pub struct OptionComesFromItsLayer;

impl Rule for OptionComesFromItsLayer {
    const CONTRACT: Contract = Contract {
        id: "option_comes_from_its_layer",
        guarantee: "A live option is the topmost override's that sets it, else the active agent's, else the session's.",
        examples: &[override_over_agent_over_session],
        counterexamples: &[agent_over_override],
        probe: Some(layered_endpointing),
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        let inputs = r.inputs_view();
        let mut taken = inputs.accepted().filter_map(|sent| match &sent.input {
            Input::Configure { knob, mode, ms, .. } => {
                Some(Taken::Configure(turn_update(*knob, *mode, *ms).map(|u| update_layer(&u))))
            }
            Input::Override { knob, mode, ms } => {
                turn_update(*knob, *mode, *ms).map(|u| Taken::Override(update_layer(&u)))
            }
            Input::ReleaseOverride { nth } => Some(Taken::Release(nth.0 as usize)),
            _ => None,
        });
        // The layer each handoff installs, in the order the loop takes them; a refused one
        // installs nothing.
        let mut handoffs = (r.host_view().api_calls().into_iter()).filter_map(|call| {
            match (call.input(), call.returned()) {
                (_, Some((_, Answer::HandoffRefused))) => None,
                (Input::Handoff { turn, .. }, _) => Some(layer_of(*turn)),
                _ => None,
            }
        });
        let (mut session, mut agent) = (Leaves::default(), Leaves::default());
        let mut overrides: Vec<Leaves> = Vec::new();
        // The agents a `VoiceTask` suspended, innermost last, and the layer a task's own install
        // brings: its agent's (none), or on its end its parent's again.
        let (mut parents, mut task_install) = (Vec::new(), None);
        let mut first = true;
        for (at, step) in r.journal() {
            match step {
                Step::Handoff(HandoffStep::TaskEntered { .. }) => {
                    parents.push(agent);
                    task_install = Some(Leaves::default());
                }
                Step::Handoff(HandoffStep::TaskEnded { .. }) => task_install = parents.pop(),
                Step::Core(CoreStep::CommandReceived {
                    origin: Origin::Host,
                    kind:
                        "TurnCommand::Configure"
                        | "TurnCommand::Override"
                        | "TurnCommand::ReleaseOverride",
                }) => match taken.next() {
                    Some(Taken::Configure(Some(layer))) => {
                        session = Leaves::of(&layer).over(session);
                    }
                    Some(Taken::Override(layer)) => overrides.push(Leaves::of(&layer)),
                    Some(Taken::Release(nth)) if !overrides.is_empty() => {
                        overrides.remove(nth % overrides.len());
                    }
                    _ => {}
                },
                Step::Core(CoreStep::ConfigApplied { layer, effective }) => {
                    if *layer == ConfigLayer::Agent {
                        agent = task_install.take().or_else(|| handoffs.next()).unwrap_or_default();
                    }
                    if std::mem::take(&mut first) {
                        continue;
                    }
                    let expected = (overrides.iter().rev())
                        .chain([&agent, &session])
                        .fold(Leaves::default(), |acc, below| acc.over(*below));
                    let actual = Leaves::of(effective);
                    check.expect(
                        Subject::Session,
                        expected.agrees(&actual),
                        format!(
                            "at {at:?} the layers set {expected:?}, the session resolved {actual:?}"
                        ),
                    );
                }
                _ => {}
            }
        }
    }
}

/// A host call that moves a layer, in the order the loop takes them.
enum Taken {
    /// `None`: an option the oracle does not judge.
    Configure(Option<AgentSessionOptions>),
    Override(AgentSessionOptions),
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

    /// Every option `self` says, `resolved` says too.
    fn agrees(&self, resolved: &Leaves) -> bool {
        fn leaf<T: PartialEq>(said: Option<T>, resolved: Option<T>) -> bool {
            said.is_none() || said == resolved
        }
        leaf(self.detection, resolved.detection)
            && leaf(self.min_delay, resolved.min_delay)
            && leaf(self.max_delay, resolved.max_delay)
    }
}

fn layer_of(turn: Option<AgentTurn>) -> Leaves {
    let kwargs = turn.and_then(agent_turn_handling);
    Leaves::of(&lk_agents_voice::turn::agent_layer(kwargs.as_ref(), None, None))
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
