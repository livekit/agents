//! A drain asked during an inline task's entry waits for the paused agent's `on_exit`, not the
//! task agent's.
use lk_agents_fuzz::drive::{Case, Setup, run};
use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref, SayInput};

// Catches: the contract reading the last `DrainStarted` as the drain's exit (campaign seed 5026).
// The first drain binds to the agent being paused and is answered once that agent's `on_exit`
// ends, as the reference's `drain()` on the paused activity is; the second drain, on the task
// agent, had meanwhile started that agent's own exit, which the first drain does not wait for.
#[test]
fn a_drain_of_a_paused_agent_waits_for_its_own_exit() {
    let inputs = vec![
        // A speech the handoff's pause waits on, so the drain arrives mid-switch.
        Input::Say {
            text: AgentText("Your balance is due on the fourteenth.".into()),
            input: SayInput::Words,
            allow: None,
            to_ctx: true,
        },
        Input::Handoff {
            agent: "billing_agent".into(),
            instructions: AgentText("You handle billing.".into()),
            tools: vec!["lookup".into()],
            pause_previous: true,
            resume: false,
        },
        Input::Drain { wait: true },
        Input::Advance { ms: Millis(1_000) },
        Input::PlayoutFinished { segment: Ref::new(0) },
        Input::Advance { ms: Millis(10) },
        Input::Drain { wait: true },
    ];
    let setup = Setup { hook_bound_ms: Some(15_000), ..Setup::default() };
    let outcome = run(Case { setup, inputs, ..Case::default() });
    assert!(outcome.ok(), "{:#?}", outcome.failures);
}
