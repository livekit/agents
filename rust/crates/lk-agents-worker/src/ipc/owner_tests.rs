//! The state a `JobSupervisor` exposes, over a job thread a scripted child answers for.

use super::*;
use std::any::Any;
use std::time::Duration;

use lk_agents_abi::BoxFuture;
use lk_agents_abi::worker::ipc_host::JobSupervisor as _;
use serde_json::{Map, Value};

use crate::ipc::Channel;
use crate::ipc::pool::tests::job;
use crate::ipc::wire::{ToChild, ToParent};

/// How the scripted child ends.
#[derive(Clone, Copy, PartialEq)]
enum Child {
    /// It shuts down when asked.
    Obedient,
    /// It vanishes once it took its job, without saying it is shutting down.
    Vanishes,
}

/// Starts a job thread the script answers for, and records its completions.
struct Launcher {
    child: Child,
    completions: Mutex<Vec<JobCompletion>>,
}

impl JobLauncher for Launcher {
    fn launch<'a>(
        &'a self,
        _: &telemetry::Cx,
        _: &'a CallContext,
        endpoint: i64,
        launch: Arc<dyn LaunchReport>,
    ) -> BoxFuture<'a, Result<(), HostError>> {
        let child = self.child;
        Box::pin(async move {
            tokio::spawn(serve(endpoint, launch, child));
            Ok(())
        })
    }

    fn completed<'a>(
        &'a self,
        _: &telemetry::Cx,
        _: &'a CallContext,
        completion: JobCompletion,
    ) -> BoxFuture<'a, Result<(), HostError>> {
        self.completions.lock().push(completion);
        Box::pin(async { Ok(()) })
    }

    fn log(
        &self,
        _: i32,
        _: String,
        _: Map<String, Value>,
        _: Option<HostError>,
    ) -> Result<(), HostError> {
        Ok(())
    }
}

async fn serve(endpoint: i64, launch: Arc<dyn LaunchReport>, child: Child) {
    let channel = Channel::<ToParent, ToChild>::from_endpoint(endpoint).unwrap();
    while let Ok(message) = channel.recv().await {
        let reply = match message {
            ToChild::Hello(_) => ToParent::Ready { error: None },
            ToChild::Ping { sent_ms } => ToParent::Pong { sent_ms },
            ToChild::StartJob { request_id, .. } => {
                channel.send(ToParent::JobAccepted { request_id, error: None }).await.unwrap();
                if child == Child::Vanishes {
                    break;
                }
                continue;
            }
            ToChild::Shutdown { .. } => {
                channel.send(ToParent::ShutdownAck).await.unwrap();
                channel.send(ToParent::ShuttingDown).await.unwrap();
                channel.send(ToParent::Exiting { reason: "asked".into() }).await.unwrap();
                break;
            }
        };
        channel.send(reply).await.unwrap();
    }
    channel.close().await;
    let launch: Arc<dyn Any + Send + Sync> = launch;
    let launch = launch.downcast::<JobLaunch>().unwrap();
    ipc_host::JobLaunch::thread_exited(&*launch).unwrap();
}

fn thread(child: Child) -> (JobSupervisor, Arc<Launcher>) {
    let launcher = Arc::new(Launcher { child, completions: Mutex::new(Vec::new()) });
    let options = JobRunnerOptions {
        thread: true,
        initialize_timeout: 5.0,
        close_timeout: 5.0,
        memory_warn_mb: 0.0,
        memory_limit_mb: 0.0,
        ping_interval: 2.5,
        ping_timeout: 60.0,
        high_ping_threshold: 0.5,
        http_proxy: None,
        asyncio_debug: false,
    };
    let env = ProcessEnv::current_runtime();
    let sup = JobSupervisor::new(&env, launcher.clone(), options, CallContext::default()).unwrap();
    (sup, launcher)
}

async fn within<T>(work: impl std::future::Future<Output = T>) -> T {
    tokio::time::timeout(Duration::from_secs(10), work).await.expect("the step did not settle")
}

// Catches: status, running job and exit code not following the job (None, then RUNNING, then
// SUCCESS once the supervision is over), a second job accepted, or the completion reported twice.
#[tokio::test(flavor = "multi_thread")]
async fn a_job_thread_reports_its_job_from_launch_to_a_clean_end() {
    let (sup, launcher) = thread(Child::Obedient);
    assert!(sup.thread() && sup.id().starts_with("THEXEC_"));
    assert!(!sup.started() && sup.status().is_none() && sup.running_job().is_none());
    within(sup.start()).await.unwrap();
    assert!(within(sup.start()).await.is_err(), "a second start launched a second runner");
    within(sup.initialize()).await.unwrap();
    assert!(sup.initialized() && sup.initialize_time().is_some());
    within(sup.launch_job(job("j1"), CallContext::default(), telemetry::Cx::NONE)).await.unwrap();
    assert_eq!(sup.status(), Some(JobStatus::Running));
    assert_eq!(sup.running_job().map(|info| info.job.id), Some("j1".into()));
    let second = within(sup.launch_job(job("j2"), CallContext::default(), telemetry::Cx::NONE));
    assert!(second.await.is_err(), "a runner took a second job");
    within(sup.aclose()).await;
    assert!(within(sup.join()).await);
    assert!(sup.closing() && !sup.killed(), "a thread is never killed");
    assert_eq!(sup.status(), Some(JobStatus::Success));
    assert_eq!(sup.exitcode(), Some(0));
    assert_eq!(sup.running_job().map(|info| info.job.id), Some("j1".into()), "kept after the end");
    assert_eq!(launcher.completions.lock().len(), 1);
}

// Catches: a runner whose control channel ended without a shutdown counted a success.
#[tokio::test(flavor = "multi_thread")]
async fn a_runner_that_vanishes_with_its_job_failed() {
    let (sup, _launcher) = thread(Child::Vanishes);
    within(sup.start()).await.unwrap();
    within(sup.initialize()).await.unwrap();
    within(sup.launch_job(job("j1"), CallContext::default(), telemetry::Cx::NONE)).await.unwrap();
    assert!(within(sup.join()).await);
    assert_eq!(sup.status(), Some(JobStatus::Failed));
}

// Catches: a closed supervisor still launching a runner, or join waiting on a runner never
// started.
#[tokio::test(flavor = "multi_thread")]
async fn a_closed_supervisor_starts_nothing() {
    let (sup, launcher) = thread(Child::Obedient);
    within(sup.aclose()).await;
    assert!(sup.closing());
    assert!(within(sup.start()).await.is_err());
    assert!(!sup.started());
    assert!(!within(sup.join()).await);
    assert!(sup.status().is_none() && sup.exitcode().is_none());
    assert!(launcher.completions.lock().is_empty());
}

// Catches: kill() returning before a thread it cannot stop returned.
#[tokio::test(flavor = "multi_thread")]
async fn killing_a_thread_waits_for_its_return() {
    let (sup, _launcher) = thread(Child::Obedient);
    within(sup.start()).await.unwrap();
    within(sup.initialize()).await.unwrap();
    within(sup.kill()).await;
    assert!(within(sup.join()).await);
    assert!(!sup.killed(), "a thread cannot be killed");
    assert!(sup.status().is_none(), "no job ran");
}
