//! Every node the session opens is the active agent's.
//!
//! An agent's `llm_node`, `tts_node`, `stt_node`, `transcription_node` and
//! `realtime_audio_output_node` are its own: the reference calls each on the agent of the activity
//! running the step, and a handoff drains the old activity before the new one speaks. So once the
//! driver installed an agent (`AgentUpdated`, named by the `on_enter` it issues), no node of the
//! agent it replaced opens again past the instant of that install (several handoffs landing in one
//! cycle each ask for a recognition stream, opened after the cycle and discarded once superseded).
//! The agent a handoff asked for may open its nodes from the call on: the journal is read after
//! the fact, so its stamps can trail what the fakes saw.
use lk_agents_voice::{
    agent_session::{HookId, HookKind},
    trace::{HandoffStep, Step},
};

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
        // Each install, and the agent its `on_enter` names.
        let mut installs: Vec<(Observed, &str)> = Vec::new();
        let mut installing = None;
        for (at, step) in r.internal_view().steps() {
            match step {
                Step::Handoff(HandoffStep::AgentUpdated { .. }) => installing = Some(at),
                Step::Handoff(HandoffStep::HookIssued {
                    kind: HookKind::OnEnter,
                    agent: Some(agent),
                    ..
                }) => {
                    if let Some(at) = installing.take() {
                        installs.push((*at, agent.as_str()));
                    }
                }
                _ => {}
            }
        }
        let asked: Vec<(Observed, &str)> = r
            .host_view()
            .api_calls()
            .into_iter()
            .filter_map(|call| match call.input() {
                Input::Handoff { agent, .. } => Some((call.called_at(), agent.as_str())),
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
            check.expect(
                Subject::Session,
                agent == active || arriving || same_instant,
                format!("{node:?} of {agent} opened at seq {} while {active} is active", at.seq),
            );
        }
    }
}

fn swapped() -> Step {
    Step::Handoff(HandoffStep::AgentUpdated {
        tools: 0,
        chat_ctx: false,
        reopened: "",
        legs: Default::default(),
    })
}

fn entered(hook: u64, agent: &str) -> Step {
    Step::Handoff(HandoffStep::HookIssued {
        hook: HookId(hook),
        kind: HookKind::OnEnter,
        agent: Some(agent.into()),
        speech: None,
        turn: None,
    })
}

fn handoff(agent: &str) -> Input {
    Input::Handoff {
        agent: agent.into(),
        instructions: AgentText("You are next.".into()),
        tools: Vec::new(),
    }
}

/// The opening agent's nodes, its successor's from the handoff's call and after its install.
fn opened_by_each_agent_in_turn() -> Recording {
    Authored::new()
        .node(OPENING_AGENT, NodeUse::Llm)
        .api_call(handoff("next"))
        .node("next", NodeUse::Stt)
        .node(OPENING_AGENT, NodeUse::Tts)
        .journal(swapped())
        .journal(entered(2, "next"))
        .node("next", NodeUse::Transcription)
        .done()
}

/// The replaced agent's transcription node, still opened after the new agent was installed.
fn opened_by_the_replaced_agent() -> Recording {
    Authored::new()
        .node(OPENING_AGENT, NodeUse::Transcription)
        .api_call(handoff("next"))
        .journal(swapped())
        .journal(entered(2, "next"))
        .node(OPENING_AGENT, NodeUse::Transcription)
        .done()
}

/// Prerequisite: an opening agent overriding every node but the realtime one hands off to an agent
/// with overrides of its own, which then speaks.
fn handoff_between_overrides() -> Case {
    let say = || Input::Say {
        text: AgentText("Let me check that for you.".into()),
        input: SayInput::Words,
        allow: None,
        to_ctx: true,
    };
    let nodes = NodeSetup {
        llm: true,
        tts: true,
        stt: true,
        llm_text: None,
        transcription: true,
        realtime_audio: false,
    };
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
