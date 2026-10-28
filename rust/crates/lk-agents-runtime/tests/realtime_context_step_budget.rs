//! Round-robin realtime core steps at 1/8/32/64 sessions and 1k/10k-item histories.
//! LK_REALTIME_STEP_QUALIFICATION=1 enforces K-31 CPU gates with optimized core packages.
#![cfg(unix)]

mod realtime_support;
use realtime_support::{RetirementStats, thread_cpu};

use std::time::{Duration, Instant, UNIX_EPOCH};

use lk_agents_abi::llm::realtime::UpdateChatCtx;
use lk_agents_abi::{ChatContext, ChatMessage, RealtimeCommand, Role, Ts};
use lk_agents_inference::realtime::{OPENAI, Outgoing, RealtimeMachine, SessionOptions, Step};

fn context(count: usize) -> ChatContext {
    ChatContext::new(
        (0..count)
            .map(|id| {
                let id = id.to_string();
                ChatMessage::text(&id, Role::User, &id, UNIX_EPOCH).into()
            })
            .collect(),
    )
}

#[derive(Default)]
struct Samples {
    cpu: Vec<Duration>,
    wall: Vec<Duration>,
    phase: &'static str,
    maximum: Option<Maximum>,
    cleanup: RetirementStats,
}

#[derive(Debug)]
struct Maximum {
    cpu: Duration,
    phase: &'static str,
    operation: &'static str,
    frames: usize,
    first_item: Option<String>,
    last_item: Option<String>,
}

impl Samples {
    fn step(&mut self, operation: &'static str, call: impl FnOnce() -> Step) -> Step {
        let wall = Instant::now();
        let cpu = thread_cpu();
        let result = call();
        let cpu = thread_cpu().saturating_sub(cpu);
        self.cpu.push(cpu);
        self.wall.push(wall.elapsed());
        if self.maximum.as_ref().is_none_or(|previous| cpu > previous.cpu) {
            let mut ids = result.frames.iter().filter_map(|frame| match frame {
                Outgoing::ItemCreate { item, .. } => Some(item.id().to_owned()),
                _ => None,
            });
            let first_item = ids.next();
            let last_item = ids.last().or_else(|| first_item.clone());
            self.maximum = Some(Maximum {
                cpu,
                phase: self.phase,
                operation,
                frames: result.frames.len(),
                first_item,
                last_item,
            });
        }
        result
    }
}

fn observe(step: Step, ids: &mut Vec<String>) {
    assert!(step.frames.len() <= 64, "one context page exceeded the operation bound");
    for frame in step.frames {
        if let Outgoing::ItemCreate { item, .. } = frame {
            ids.push(item.id().to_owned());
        }
    }
}

fn execute_retirements(machine: &mut RealtimeMachine, samples: &mut Samples) {
    let stats = realtime_support::execute_retirements(machine, |_| {});
    samples.cleanup.jobs += stats.jobs;
    samples.cleanup.cpu += stats.cpu;
    samples.cleanup.maximum = samples.cleanup.maximum.max(stats.maximum);
}

fn drain(
    machines: &mut [RealtimeMachine],
    samples: &mut Samples,
    first: Vec<Step>,
) -> Vec<Vec<String>> {
    let mut ids: Vec<Vec<String>> = (0..machines.len()).map(|_| Vec::new()).collect();
    for (step, observed) in first.into_iter().zip(&mut ids) {
        observe(step, observed);
    }
    for machine in &mut *machines {
        execute_retirements(machine, samples);
    }
    while machines.iter().any(RealtimeMachine::context_work_pending) {
        for (machine, observed) in machines.iter_mut().zip(&mut ids) {
            if machine.context_work_pending() {
                observe(
                    samples.step("continue_context", || machine.continue_context(Ts::ZERO)),
                    observed,
                );
                execute_retirements(machine, samples);
            }
            assert!(samples.step("tick", || machine.tick(Ts::ZERO)).frames.is_empty());
        }
    }
    ids
}

fn check_history(ids: Vec<Vec<String>>, count: usize) {
    for history in ids {
        assert_eq!(history.len(), count);
        for (expected, id) in history.into_iter().enumerate() {
            assert_eq!(id.parse::<usize>().unwrap(), expected, "history order or identity changed");
        }
    }
}

#[test]
fn realtime_context_step_budget() {
    let mut failures = Vec::new();
    for sessions in [1, 8, 32, 64] {
        for count in [1_000, 10_000] {
            let mut machines: Vec<_> = (0..sessions)
                .map(|session| {
                    let opts = SessionOptions::from_dialect(&OPENAI, false);
                    let caps = OPENAI.capabilities(opts.turn_detection.as_ref(), false, true, true);
                    let mut machine =
                        RealtimeMachine::new(OPENAI, opts, caps, format!("s{session}_"));
                    machine.open(Ts::ZERO);
                    machine
                })
                .collect();
            let mut samples = Samples { phase: "initial", ..Default::default() };
            let targets: Vec<_> = (0..sessions).map(|_| context(count)).collect();
            let mut first = Vec::new();
            for (machine, target) in machines.iter_mut().zip(&targets) {
                let request =
                    RealtimeCommand::UpdateChatCtx(UpdateChatCtx { chat_ctx: target.clone() });
                first.push(samples.step("command", || machine.command(Ts::ZERO, request)));
            }
            check_history(drain(&mut machines, &mut samples, first), count);
            samples.phase = "identical";
            for (machine, target) in machines.iter_mut().zip(targets) {
                let request = RealtimeCommand::UpdateChatCtx(UpdateChatCtx { chat_ctx: target });
                assert!(
                    samples
                        .step("command", || machine.command(Ts::ZERO, request))
                        .frames
                        .is_empty()
                );
            }
            samples.phase = "append";
            let mut first = Vec::new();
            for machine in &mut machines {
                let request =
                    RealtimeCommand::UpdateChatCtx(UpdateChatCtx { chat_ctx: context(count + 1) });
                first.push(samples.step("command", || machine.command(Ts::ZERO, request)));
            }
            for ids in drain(&mut machines, &mut samples, first) {
                assert_eq!(ids, [count.to_string()]);
            }
            samples.phase = "reconnect";
            let first = machines
                .iter_mut()
                .map(|machine| samples.step("reconnect", || machine.reconnect(Ts::ZERO)))
                .collect();
            check_history(drain(&mut machines, &mut samples, first), count + 1);
            while samples.cpu.len() < 1_000 {
                samples.phase = "repeated_reconnect";
                let first = machines
                    .iter_mut()
                    .map(|machine| samples.step("reconnect", || machine.reconnect(Ts::ZERO)))
                    .collect();
                check_history(drain(&mut machines, &mut samples, first), count + 1);
            }
            samples.cpu.sort_unstable();
            samples.wall.sort_unstable();
            let index = (samples.cpu.len() * 999 / 1000).min(samples.cpu.len() - 1);
            let cpu_p999 = samples.cpu[index];
            let cpu_max = *samples.cpu.last().unwrap();
            println!(
                "{sessions} sessions / {count} items / {} steps: CPU p99.9={cpu_p999:?} max={cpu_max:?}; wall p99.9={:?} max={:?}",
                samples.cpu.len(),
                samples.wall[index],
                samples.wall.last().unwrap()
            );
            println!("largest step: {:?}", samples.maximum);
            println!(
                "retirement effects: jobs={} worker_cpu_total_ns={} worker_cpu_max_job_ns={}",
                samples.cleanup.jobs,
                samples.cleanup.cpu.as_nanos(),
                samples.cleanup.maximum.as_nanos()
            );
            if std::env::var_os("LK_REALTIME_STEP_QUALIFICATION").is_some()
                && (cpu_p999 > Duration::from_millis(1) || cpu_max > Duration::from_millis(2))
            {
                failures.push(format!("{sessions} sessions/{count} items: CPU p99.9={cpu_p999:?} max={cpu_max:?}; {:?}", samples.maximum));
            }
        }
    }
    assert!(failures.is_empty(), "K-31 core CPU budget exceeded:\n{}", failures.join("\n"));
}
