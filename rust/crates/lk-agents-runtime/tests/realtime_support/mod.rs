//! Execute context disposal through the same process-owned admission pool as live sessions.

use std::sync::OnceLock;
use std::time::Duration;

use lk_agents_inference::realtime::RealtimeMachine;
use lk_agents_media::workers::{MediaWorkerClass, MediaWorkers};

pub fn thread_cpu() -> Duration {
    let mut time = libc::timespec { tv_sec: 0, tv_nsec: 0 };
    // SAFETY: the OS writes one initialized timespec through a valid exclusive pointer.
    assert_eq!(unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut time) }, 0);
    Duration::new(time.tv_sec.try_into().unwrap(), time.tv_nsec.try_into().unwrap())
}

#[derive(Default)]
pub struct RetirementStats {
    pub jobs: usize,
    pub cpu: Duration,
    pub maximum: Duration,
}

struct Executor {
    handle: tokio::runtime::Handle,
    workers: MediaWorkers,
}

fn executor() -> &'static Executor {
    static EXECUTOR: OnceLock<Executor> = OnceLock::new();
    EXECUTOR.get_or_init(|| Executor {
        handle: lk_agents_runtime::process::handle().expect("process runtime"),
        workers: lk_agents_runtime::process::workers().expect("process Formatting admission"),
    })
}

pub fn execute_retirements(
    machine: &mut RealtimeMachine,
    mut controls: impl FnMut(&mut RealtimeMachine),
) -> RetirementStats {
    let executor = executor();
    let mut stats = RetirementStats::default();
    while machine.has_context_retirement() {
        // Capacity refusal leaves every destructor-bearing owner in the machine.
        let permit = loop {
            match executor.workers.try_admit(MediaWorkerClass::Formatting) {
                Ok(permit) => break permit,
                Err(_) => executor
                    .handle
                    .block_on(executor.workers.wait_for_capacity(MediaWorkerClass::Formatting)),
            }
        };
        let retirement = machine.take_context_retirement().expect("admitted retired owner");
        let core_thread = std::thread::current().id();
        let receipt = executor.workers.spawn_admitted(permit, move || {
            assert_ne!(std::thread::current().id(), core_thread, "cleanup ran on the core thread");
            let start = thread_cpu();
            drop(retirement);
            thread_cpu().saturating_sub(start)
        });
        controls(machine);
        // Admission, dispatch, controls and awaiting settlement are outside timed core calls.
        let cpu = executor.handle.block_on(receipt).expect("retirement worker settled");
        stats.jobs += 1;
        stats.cpu += cpu;
        stats.maximum = stats.maximum.max(cpu);
    }
    stats
}
