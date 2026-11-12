//! The text driver: its checkers, its regressions and its seeded campaign.

use lk_agents_abi::a2a::options::Policy;
use lk_agents_fuzz::text::{self, Callback, Fault, Scenario};

fn report(outcome: &text::Outcome) -> String {
    let failures: Vec<String> =
        outcome.failures.iter().map(|f| format!("{}: {}", f.contract, f.detail)).collect();
    let r = &outcome.recording;
    format!(
        "{:?}\n{}\n  turns: {:?}\n  tasks: {:?}\n  partition_key: {:?}",
        outcome.scenario,
        failures.join("\n"),
        r.turns,
        r.tasks,
        r.partition_key
    )
}

/// Every text contract accepts its examples while expecting something, and rejects each of its
/// counterexamples.
#[test]
fn every_text_contract_rejects_its_counterexamples() {
    for rule in text::rules::TEXT_RULES {
        let id = rule.contract.id;
        for example in rule.contract.examples {
            let check = rule.run(&example());
            assert!(check.failures.is_empty(), "{id} rejected an example: {:?}", check.failures);
            assert!(check.expectations > 0, "{id} made no expectation on an example");
        }
        assert!(!rule.contract.counterexamples.is_empty(), "{id} has no counterexample");
        for counterexample in rule.contract.counterexamples {
            let check = rule.run(&counterexample());
            assert!(
                check.failures.iter().any(|f| f.contract == id),
                "{id} accepted a counterexample"
            );
        }
        assert!(rule.run(&text::TextRecording::default()).failures.is_empty(), "{id} fails empty");
    }
}

/// A run whose fakes all answer finishes, and its driver exits.
#[test]
fn a_plain_run_finishes() {
    let outcome = text::run(Scenario::direct(1));
    assert!(outcome.failures.is_empty(), "{}", report(&outcome));
    assert!(outcome.recording.runs[0].terminal_ms.is_some());
}

/// A panicked host callback is a lost input to the run, which settles on it rather than waiting
/// forever for the callback's report.
#[test]
fn a_panicking_host_callback_settles_the_run() {
    for callback in
        [Callback::Checkpoint, Callback::ListTools, Callback::Prepare, Callback::RunTool]
    {
        let scenario = Scenario { forced: vec![(callback, Fault::Panic)], ..Scenario::direct(7) };
        let outcome = text::run(scenario);
        assert!(outcome.failures.is_empty(), "{callback:?}: {}", report(&outcome));
    }
}

/// A model stream whose task panicked reports the request failed: the run finishes, its driver
/// exits, and an A2A turn releases its partition.
#[test]
fn a_panicking_model_stream_ends_the_run() {
    for callback in [Callback::LlmOpen, Callback::LlmChunk] {
        let scenario = Scenario {
            forced: vec![(callback, Fault::Panic)],
            turns: vec![text::TurnPlan {
                start_ms: 0,
                client: text::Client::Waits,
                after_ms: 0,
                worker: 0,
            }],
            ..Scenario::direct(9)
        };
        let outcome = text::run(scenario);
        assert!(outcome.failures.is_empty(), "{callback:?}: {}", report(&outcome));
    }
}

/// A queued turn whose client drops it while the partition is busy, on a healthy database.
fn a_queued_turn_aborted() -> Scenario {
    let busy = text::TurnPlan { start_ms: 0, client: text::Client::Waits, after_ms: 0, worker: 0 };
    let queued =
        text::TurnPlan { start_ms: 10, client: text::Client::Aborts, after_ms: 500, worker: 1 };
    // The busy turn's model thinks until its client gives up on it.
    let forced = vec![(Callback::LlmOpen, Fault::Hang)];
    Scenario { direct: Vec::new(), forced, turns: vec![busy, queued], ..Scenario::direct(11) }
}

/// A turn dropped while its task waits in the queue cancels the task it submitted, which no
/// later turn would run (`a2a_task_not_orphaned`; `TEXT_SEED=24` ends a turn by an outage).
#[test]
fn a_queued_turn_aborted_settles_its_task() {
    let outcome = text::run(a_queued_turn_aborted());
    assert!(outcome.failures.is_empty(), "{}", report(&outcome));
}

/// Two turns whose clients drop them under `queue`: the first while its model thinks, the
/// second while it waits for its turn.
fn two_queued_turns_aborted() -> Scenario {
    let first =
        text::TurnPlan { start_ms: 0, client: text::Client::Aborts, after_ms: 800, worker: 0 };
    let second =
        text::TurnPlan { start_ms: 100, client: text::Client::Aborts, after_ms: 300, worker: 1 };
    Scenario {
        direct: Vec::new(),
        forced: vec![(Callback::LlmOpen, Fault::Hang)],
        policy: Policy::Queue,
        turns: vec![first, second],
        ..Scenario::direct(13)
    }
}

/// Every run a turn started ends, however its turn ended: a run whose turn was dropped records
/// `run_ended` `canceled` (`TEXT_SEED=449`).
#[test]
fn an_aborted_turn_ends_its_run() {
    let outcome = text::run(two_queued_turns_aborted());
    assert!(outcome.failures.is_empty(), "{}", report(&outcome));
}

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

/// `TEXT_SEEDS` seeded scenarios from `TEXT_FIRST_SEED`; `TEXT_SEED` reruns one.
#[test]
#[ignore = "a campaign: run with --ignored"]
fn survey() {
    let seeds: Vec<u64> = match std::env::var("TEXT_SEED").ok().and_then(|v| v.parse().ok()) {
        Some(seed) => vec![seed],
        None => {
            let first = env_u64("TEXT_FIRST_SEED", 0);
            (first..first + env_u64("TEXT_SEEDS", 64)).collect()
        }
    };
    let (mut failed, mut deaths) = (Vec::new(), 0);
    let mut expectations = std::collections::BTreeMap::<&str, u32>::new();
    for seed in seeds {
        let outcome = text::run(Scenario::draw(seed));
        deaths += outcome.worker_deaths;
        for (id, n) in &outcome.expectations {
            *expectations.entry(id).or_default() += n;
        }
        if !outcome.failures.is_empty() {
            println!("FINDING seed {seed}: {}", report(&outcome));
            println!(
                "  rerun: TEXT_SEED={seed} cargo test --release -p lk-agents-fuzz --test text survey -- --ignored --nocapture"
            );
            failed.push(seed);
        }
    }
    println!("expectations: {expectations:?}; database workers died: {deaths}");
    assert!(failed.is_empty(), "seeds with findings: {failed:?}");
}
