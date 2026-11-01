//! Every node the session opens is the active agent's.
//!
//! An agent's `llm_node`, `tts_node`, `stt_node` and `transcription_node`
//! are its own: the reference calls each on the agent of the activity
//! running the step, and a handoff drains the old activity before the new one speaks. So once the
//! driver installed an agent (`AgentUpdated`, which names it: a handoff's, a `VoiceTask`'s, or a
//! parent a task gives the floor back to), no node of the agent it replaced opens again past the
//! instant of that install (several handoffs landing in one cycle each ask for a recognition
//! stream, opened after the cycle and discarded once superseded). The agent a handoff or a task
//! asked for may open its nodes from the call on: the journal is read after the fact, so its
//! stamps can trail what the fakes saw. A parent a task suspends finishes the step whose tool
//! started the task: its nodes may open until the task ends.
use lk_agents_voice::trace::{HandoffStep, Step};

use crate::{
    authored::Authored,
    check::{Check, Contract, Rule, Subject},
    drive::{Case, NodeSetup, OPENING_AGENT, Setup},
    input::{AgentText, Input, Millis, SayInput},
    record::{NodeUse, Observed, Recording},
};

pub struct NodesAreTheActiveAgents;

impl Rule for NodesAreTheActiveAgents {
    const CONTRACT: Contract = Contract {
        id: "nodes_are_the_active_agents",
        guarantee: "Every node the session opens is an override of the agent active then.",
        examples: &[opened_by_each_agent_in_turn],
        counterexamples: &[opened_by_the_replaced_agent],
        probe: Some(handoff_between_overrides),
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        // Each install, and the agent it installed.
        let installs: Vec<(Observed, &str)> = (r.internal_view().steps().iter())
            .filter_map(|(at, step)| match step {
                Step::Handoff(HandoffStep::AgentUpdated { agent, .. }) => {
                    Some((*at, agent.as_str()))
                }
                _ => None,
            })
            .collect();
        // Each task's parent, from its entry to its end.
        let mut suspended: Vec<(Observed, Observed, &str)> = Vec::new();
        let mut open: Vec<(
            lk_agents_voice::agent_activity::voice_task::VoiceTaskId,
            Observed,
            &str,
        )> = Vec::new();
        for (at, step) in r.internal_view().steps() {
            match step {
                Step::Handoff(HandoffStep::TaskEntered { task, parent, .. }) => {
                    open.push((*task, *at, parent.as_str()));
                }
                Step::Handoff(HandoffStep::TaskEnded { task, .. }) => {
                    if let Some(i) = open.iter().position(|(t, ..)| t == task) {
                        let (_, from, parent) = open.remove(i);
                        suspended.push((from, *at, parent));
                    }
                }
                _ => {}
            }
        }
        suspended.extend(open.into_iter().map(|(_, from, parent)| (from, Observed::END, parent)));
        let asked: Vec<(Observed, &str)> = r
            .host_view()
            .api_calls()
            .into_iter()
            .filter_map(|call| match call.input() {
                Input::Handoff { agent, .. } | Input::VoiceTask { agent, .. } => {
                    Some((call.called_at(), agent.as_str()))
                }
                _ => None,
            })
            .collect();
        for (at, agent, node) in r.inference_view().nodes() {
            let before = &installs[..installs.partition_point(|(when, _)| *when < at)];
            let active = before.last().map_or(OPENING_AGENT, |(_, agent)| *agent);
            let since = before.last().map(|(when, _)| *when);
            // What one cycle decided runs after it: an agent replaced in the same instant
            // (several handoffs landing at once) may still open what its install asked for.
            let same_instant = before.iter().enumerate().any(|(i, (when, _))| {
                let replaced = if i == 0 { OPENING_AGENT } else { before[i - 1].1 };
                when.at_ms == at.at_ms && replaced == agent
            });
            // Asked for after the last install, and not installed yet as far as the journal says.
            let arriving = asked
                .iter()
                .any(|(called, to)| *called < at && *to == agent && since < Some(*called));
            let finishing = suspended
                .iter()
                .any(|(from, to, parent)| *from < at && at <= *to && *parent == agent);
            check.expect(
                Subject::Session,
                agent == active || arriving || same_instant || finishing,
                format!("{node:?} of {agent} opened at seq {} while {active} is active", at.seq),
            );
        }
    }
}

fn swapped(agent: &str) -> Step {
    Step::Handoff(HandoffStep::AgentUpdated {
        agent: agent.into(),
        tools: 0,
        chat_ctx: false,
        reopened: "",
        legs: Default::default(),
    })
}

fn handoff(agent: &str) -> Input {
    Input::Handoff {
        agent: agent.into(),
        instructions: AgentText("You are next.".into()),
        tools: Vec::new(),
        turn: None,
    }
}

/// The opening agent's nodes, its successor's from the handoff's call and after its install.
fn opened_by_each_agent_in_turn() -> Recording {
    Authored::new()
        .node(OPENING_AGENT, NodeUse::Llm)
        .api_call(handoff("next"))
        .node("next", NodeUse::Stt)
        .node(OPENING_AGENT, NodeUse::Tts)
        .journal(swapped("next"))
        .node("next", NodeUse::Transcription)
        .done()
}

/// The replaced agent's transcription node, still opened after the new agent was installed.
fn opened_by_the_replaced_agent() -> Recording {
    Authored::new()
        .node(OPENING_AGENT, NodeUse::Transcription)
        .api_call(handoff("next"))
        .journal(swapped("next"))
        .node(OPENING_AGENT, NodeUse::Transcription)
        .done()
}

/// Prerequisite: an opening agent overriding every node hands off to an agent
/// with overrides of its own, which then speaks.
fn handoff_between_overrides() -> Case {
    let say = || Input::Say {
        text: AgentText("Let me check that for you.".into()),
        input: SayInput::Words,
        allow: None,
        to_ctx: true,
    };
    let nodes = NodeSetup { llm: true, tts: true, stt: true, llm_text: None, transcription: true };
    Case {
        setup: Setup { stt: true, vad: true, nodes: Some(nodes), ..Setup::default() },
        inputs: vec![
            handoff("next"),
            Input::Advance { ms: Millis(2_000) },
            say(),
            Input::Advance { ms: Millis(2_000) },
        ],
        ..Case::default()
    }
}
