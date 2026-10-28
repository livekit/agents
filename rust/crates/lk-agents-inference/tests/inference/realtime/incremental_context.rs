//! Core context preparation, acknowledgements and retirement under long histories.

use super::support::thread_cpu as cpu_time;

use std::collections::BTreeMap;
use std::time::{Duration, Instant, UNIX_EPOCH};

use lk_agents_abi::llm::realtime::UpdateChatCtx;
use lk_agents_abi::{ChatContext, ChatMessage, RealtimeCommand, Role, Ts};
use lk_agents_inference::realtime::{
    Incoming, OPENAI, Outgoing, RealtimeMachine, SessionOptions, Step,
};

fn context(count: usize) -> ChatContext {
    ChatContext::new(
        (0..count)
            .map(|id| {
                ChatMessage::text(&id.to_string(), Role::User, &id.to_string(), UNIX_EPOCH).into()
            })
            .collect(),
    )
}

#[derive(Default)]
struct Samples {
    cpu: Vec<Duration>,
    wall: Vec<Duration>,
    max_frames: usize,
    maximum: Option<(&'static str, Duration)>,
    cleanup_cpu: Duration,
    cleanup_max: Duration,
    cleanup_jobs: usize,
}

impl Samples {
    fn step(&mut self, name: &'static str, f: impl FnOnce() -> Step) -> Step {
        let wall = Instant::now();
        let cpu = cpu_time();
        let step = f();
        let cpu = cpu_time().saturating_sub(cpu);
        self.cpu.push(cpu);
        self.wall.push(wall.elapsed());
        self.max_frames = self.max_frames.max(step.frames.len());
        if self.maximum.is_none_or(|(_, maximum)| cpu > maximum) {
            self.maximum = Some((name, cpu));
        }
        step
    }

    fn report(&mut self, sessions: usize, count: usize, phase: &str) -> Option<String> {
        assert!(self.cpu.len() >= 1_000, "qualification needs >=1000 observations");
        self.cpu.sort_unstable();
        self.wall.sort_unstable();
        let index = self.cpu.len() * 999 / 1000;
        println!(
            "incremental_context sessions={sessions} items={count} phase={phase} observations={} cpu_p999_ns={} cpu_max_ns={} wall_p999_ns={} wall_max_ns={} max_frames={} maximum={:?}",
            self.cpu.len(),
            self.cpu[index].as_nanos(),
            self.cpu.last().unwrap().as_nanos(),
            self.wall[index].as_nanos(),
            self.wall.last().unwrap().as_nanos(),
            self.max_frames,
            self.maximum
        );
        println!(
            "retirement effects: jobs={} worker_cpu_total_ns={} worker_cpu_max_job_ns={}",
            self.cleanup_jobs,
            self.cleanup_cpu.as_nanos(),
            self.cleanup_max.as_nanos()
        );
        let strict = std::env::var_os("LK_REALTIME_STEP_QUALIFICATION").is_some();
        (self.max_frames > 64 || (strict && (self.cpu[index] > Duration::from_millis(1) || *self.cpu.last().unwrap() > Duration::from_millis(2))))
            .then(|| format!("{sessions} sessions/{count}/{phase}: max frames={}, CPU p99.9={:?} max={:?}, operation={:?}", self.max_frames, self.cpu[index], self.cpu.last().unwrap(), self.maximum))
    }
}

fn execute_retirements(machine: &mut RealtimeMachine, samples: &mut Samples, controls: bool) {
    let stats = super::support::execute_retirements(machine, |machine| {
        if controls {
            assert!(
                samples.step("tick_during_cleanup", || machine.tick(Ts::ZERO)).frames.is_empty()
            );
            assert!(
                samples
                    .step("continue_during_cleanup", || machine.continue_context(Ts::ZERO))
                    .frames
                    .is_empty()
            );
        }
    });
    samples.cleanup_cpu += stats.cpu;
    samples.cleanup_max = samples.cleanup_max.max(stats.maximum);
    samples.cleanup_jobs += stats.jobs;
}

fn acknowledge(machine: &mut RealtimeMachine, samples: &mut Samples, step: Step) {
    for frame in step.frames {
        let incoming = match frame {
            Outgoing::ItemCreate { item, previous_item_id, .. } => Incoming::ItemAdded {
                previous_item_id: previous_item_id.filter(|id| id != "root"),
                item,
            },
            Outgoing::ItemDelete { item_id, .. } => Incoming::ItemDeleted { item_id },
            _ => continue,
        };
        let generation = machine.connection_generation();
        let result =
            samples.step("acknowledge", || machine.incoming_on(generation, Ts::ZERO, incoming));
        assert!(result.frames.is_empty());
        execute_retirements(machine, samples, false);
    }
}

fn finish(machines: &mut [RealtimeMachine], samples: &mut Samples, first: Vec<Step>) {
    for (machine, step) in machines.iter_mut().zip(first) {
        acknowledge(machine, samples, step);
        execute_retirements(machine, samples, false);
    }
    while machines.iter().any(RealtimeMachine::context_work_pending) {
        for machine in &mut *machines {
            if machine.context_work_pending() {
                let step = samples.step("continue_context", || machine.continue_context(Ts::ZERO));
                acknowledge(machine, samples, step);
                execute_retirements(machine, samples, false);
            }
            assert!(samples.step("tick", || machine.tick(Ts::ZERO)).frames.is_empty());
        }
    }
}

#[test]
fn realtime_acknowledgements_and_context_edits_stay_within_one_page() {
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
            let mut observations = BTreeMap::<&str, Samples>::new();
            for phase in ["initial_ack", "reorder", "edit", "reconnect_ack"] {
                let samples = observations.entry(phase).or_default();
                let mut first = Vec::new();
                for machine in &mut machines {
                    if phase == "reconnect_ack" {
                        first.push(samples.step("reconnect", || machine.reconnect(Ts::ZERO)));
                        continue;
                    }
                    let mut target = context(count);
                    if phase != "initial_ack" {
                        target.items.rotate_left(count / 2);
                    }
                    if phase == "edit" {
                        for item in &mut target.items {
                            if let Some(message) = item.as_message_mut() {
                                message.content =
                                    vec![lk_agents_abi::ChatContent::Text("edited".into())];
                            }
                        }
                    }
                    let command =
                        RealtimeCommand::UpdateChatCtx(UpdateChatCtx { chat_ctx: target });
                    first.push(samples.step("command", || machine.command(Ts::ZERO, command)));
                }
                finish(&mut machines, samples, first);
                for machine in &machines {
                    let actual = machine.chat_ctx();
                    assert_eq!(actual.len(), count);
                    for (at, item) in actual.items.iter().enumerate() {
                        let expected =
                            if phase == "initial_ack" { at } else { (at + count / 2) % count };
                        assert_eq!(item.id(), expected.to_string());
                        if phase == "edit" || phase == "reconnect_ack" {
                            assert_eq!(
                                item.as_message().unwrap().raw_text_content().as_deref(),
                                Some("edited")
                            );
                        }
                    }
                }
                if let Some(failure) = samples.report(sessions, count, phase) {
                    failures.push(failure);
                }
            }
        }
    }
    assert!(failures.is_empty(), "context dispatch budget exceeded:\n{}", failures.join("\n"));
}

#[test]
fn realtime_context_churn_fails_explicitly_and_close_retires_off_step() {
    let mut failures = Vec::new();
    for sessions in [1, 8, 32, 64] {
        let mut machines: Vec<_> = (0..sessions)
            .map(|session| {
                let opts = SessionOptions::from_dialect(&OPENAI, false);
                let caps = OPENAI.capabilities(opts.turn_detection.as_ref(), false, true, true);
                let mut machine =
                    RealtimeMachine::new(OPENAI, opts, caps, format!("churn{session}_"));
                machine.open(Ts::ZERO);
                machine
            })
            .collect();
        let mut samples = Samples::default();
        let first = machines
            .iter_mut()
            .map(|machine| {
                machine.command(
                    Ts::ZERO,
                    RealtimeCommand::UpdateChatCtx(UpdateChatCtx { chat_ctx: context(10_000) }),
                )
            })
            .collect();
        finish(&mut machines, &mut samples, first);
        let mut failed = vec![false; sessions];
        for revision in 0..8 {
            for (machine, failed) in machines.iter_mut().zip(&mut failed) {
                if *failed {
                    continue;
                }
                let mut target = context(10_000);
                target.items[0].as_message_mut().unwrap().content =
                    vec![lk_agents_abi::ChatContent::Text(format!("revision {revision}"))];
                let command = RealtimeCommand::UpdateChatCtx(UpdateChatCtx { chat_ctx: target });
                let step = samples.step("supersede", || machine.command(Ts::ZERO, command));
                if let Some(error) = step.fatal {
                    assert_eq!(error.message, "realtime context retirement capacity exceeded");
                    *failed = true;
                }
            }
            if failed.iter().all(|failed| *failed) {
                break;
            }
        }
        assert!(failed.iter().all(|failed| *failed), "context churn must reach a declared bound");
        for machine in &mut machines {
            assert!(samples.step("close", || machine.close(Ts::ZERO)).fatal.is_none());
        }
        for machine in &mut machines {
            execute_retirements(machine, &mut samples, true);
        }
        assert!(machines.iter().all(|machine| !machine.context_work_pending()));
        if let Some(failure) = samples.report(sessions, 10_000, "churn_close_lifecycle") {
            failures.push(failure);
        }
    }
    assert!(failures.is_empty(), "retirement dispatch budget exceeded:\n{}", failures.join("\n"));
}
