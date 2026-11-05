//! Lifetime regressions use resource ports with independently controlled completion.

use super::*;
use lk_agents_abi::AudioFrame;
use std::sync::atomic::{AtomicUsize, Ordering};
use tokio::sync::Notify;

fn owner() -> SessionOwner {
    SessionOwner::new(
        &ProcessEnv::current_runtime(),
        SessionOptions::new(crate::voice::session::tests::option_kwargs()),
    )
}

#[derive(Default)]
struct Recorder {
    finishing: Notify,
    release: Notify,
    calls: AtomicUsize,
    fail: bool,
}

impl AudioRecorder for Recorder {
    fn finish(&self) -> BoxFuture<'_, Result<(), RecordingFailure>> {
        Box::pin(async move {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.finishing.notify_one();
            self.release.notified().await;
            if self.fail { Err(RecordingFailure::Overflow("staging bytes")) } else { Ok(()) }
        })
    }
    fn on_input_frame(&self, _: f64, _: &AudioFrame) {}
    fn capture_frame(&self, _: f64, _: &AudioFrame) {}
    fn on_playback_started(&self, _: f64) {}
    fn on_playback_progressed(&self, _: f64, _: f64, _: f64) {}
    fn on_playback_finished(&self, _: f64, _: f64) {}
    fn flush(&self, _: f64) -> Result<(), RecordingFailure> {
        Ok(())
    }
}

#[tokio::test]
async fn cancelled_close_waiter_does_not_abandon_recording_and_late_close_joins() {
    let owner = owner();
    // A reporter probing before recording is attached must not cache an empty outcome.
    owner.finish_recording().await.unwrap();
    let recorder = Arc::new(Recorder::default());
    assert!(owner.0.recorder.set(recorder.clone()).is_ok());
    let waiting = owner.aclose("user_initiated".to_owned(), false).unwrap();
    recorder.finishing.notified().await;
    drop(waiting);
    owner.shutdown("user_initiated".to_owned(), false).unwrap();
    let mut rejoined = owner.aclose("job_shutdown".to_owned(), false).unwrap();
    assert!(
        std::future::poll_fn(|cx| std::task::Poll::Ready(rejoined.as_mut().poll(cx).is_pending()))
            .await
    );
    recorder.release.notify_one();
    rejoined.await;
    assert_eq!(recorder.calls.load(Ordering::SeqCst), 1);
    owner.finish_recording().await.unwrap();
    assert_eq!(recorder.calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn recording_failure_is_retained_for_reporter_without_failing_close() {
    let owner = owner();
    let recorder = Arc::new(Recorder { fail: true, ..Default::default() });
    assert!(owner.0.recorder.set(recorder.clone()).is_ok());
    let closing = owner.aclose("user_initiated".to_owned(), false).unwrap();
    recorder.finishing.notified().await;
    recorder.release.notify_one();
    closing.await;
    assert!(matches!(owner.finish_recording().await, Err(RecordingFailure::Overflow(_))));
    assert_eq!(recorder.calls.load(Ordering::SeqCst), 1);
}

#[derive(Default)]
struct FailingHost {
    preparing: Notify,
    release: Notify,
    prepared: AtomicUsize,
    closed: AtomicUsize,
}

impl SessionHost for FailingHost {
    fn session(&self) -> Result<HostRef, HostError> {
        panic!("a session that never prepared joins no job")
    }
    fn prepare(
        &self,
        _cx: &lk_agents_abi::telemetry::Cx,
    ) -> BoxFuture<'_, Result<Arc<dyn SessionBindingsPort>, HostError>> {
        Box::pin(async move {
            self.prepared.fetch_add(1, Ordering::SeqCst);
            self.preparing.notify_one();
            self.release.notified().await;
            Err(HostError::new("toolset initialization failed"))
        })
    }
    fn started(&self, _cx: &lk_agents_abi::telemetry::Cx) -> BoxFuture<'_, Result<(), HostError>> {
        Box::pin(async { panic!("failed startup must not announce success") })
    }
    fn closed(&self, _cx: &lk_agents_abi::telemetry::Cx) -> BoxFuture<'_, ()> {
        Box::pin(async move {
            self.closed.fetch_add(1, Ordering::SeqCst);
        })
    }
    fn bind_providers(
        &self,
        _providers: std::collections::BTreeMap<String, Option<HostRef>>,
    ) -> Result<lk_agents_abi::voice::session_owner::ProviderBindings, HostError> {
        panic!("a session that is not running binds no provider")
    }
}

// Catches: a provider update binding host legs while the session is still opening, before the
// voice loop it would install them on exists.
#[tokio::test]
async fn option_updates_while_the_session_opens_stay_on_the_options() {
    let owner = owner();
    let host = Arc::new(FailingHost::default());
    let storage = Arc::new(SessionStorage::new(
        &ProcessEnv::current_runtime(),
        None,
        None,
        "frontdesk".into(),
        "frontdesk".into(),
        None,
    ));
    let opening =
        owner.start(storage, host.clone(), None, lk_agents_abi::telemetry::Cx::NONE).unwrap();
    host.preparing.notified().await;
    use Maybe::NotGiven as N;
    owner.update_options(Maybe::Given(None), N, N, N, N, N, N, N, N, N).unwrap();
    let layer = owner.override_options(N, Maybe::Given(None), N, N, N, N, N, N).unwrap();
    owner.release_override(layer).unwrap();
    host.release.notify_one();
    assert!(opening.await.is_err());
}

#[tokio::test]
async fn cancelled_start_waiter_and_concurrent_close_settle_partial_acquisition_once() {
    let owner = owner();
    let host = Arc::new(FailingHost::default());
    let storage = Arc::new(SessionStorage::new(
        &ProcessEnv::current_runtime(),
        None,
        None,
        "frontdesk".into(),
        "frontdesk".into(),
        None,
    ));
    let waiting = owner
        .start(storage.clone(), host.clone(), None, lk_agents_abi::telemetry::Cx::NONE)
        .unwrap();
    host.preparing.notified().await;
    drop(waiting);
    let other = owner
        .start(storage.clone(), host.clone(), None, lk_agents_abi::telemetry::Cx::NONE)
        .unwrap();
    let mut closing = owner.aclose("job_shutdown".to_owned(), false).unwrap();
    assert!(
        std::future::poll_fn(|cx| std::task::Poll::Ready(closing.as_mut().poll(cx).is_pending()))
            .await
    );
    host.release.notify_one();
    assert_eq!(other.await.unwrap_err().message, "toolset initialization failed");
    closing.await;
    assert_eq!(host.prepared.load(Ordering::SeqCst), 1);
    assert_eq!(host.closed.load(Ordering::SeqCst), 1);
    assert!(owner.start(storage, host, None, lk_agents_abi::telemetry::Cx::NONE).is_err());
}

#[tokio::test]
async fn close_before_start_refuses_resource_acquisition() {
    let owner = owner();
    owner.aclose("user_initiated".to_owned(), false).unwrap().await;
    let host = Arc::new(FailingHost::default());
    let storage = Arc::new(SessionStorage::new(
        &ProcessEnv::current_runtime(),
        None,
        None,
        "frontdesk".into(),
        "frontdesk".into(),
        None,
    ));
    assert!(owner.start(storage, host.clone(), None, lk_agents_abi::telemetry::Cx::NONE).is_err());
    assert_eq!(host.prepared.load(Ordering::SeqCst), 0);
}

/// Holds no host payloads.
struct Registry;

impl lk_agents_abi::errors::RefRegistry for Registry {
    fn release(&self, _id: u64) {}
}

fn storage() -> Arc<SessionStorage> {
    Arc::new(SessionStorage::new(
        &ProcessEnv::current_runtime(),
        None,
        None,
        "frontdesk".into(),
        "frontdesk".into(),
        None,
    ))
}

/// A job recording what its sessions claim, record and register, in order.
#[derive(Default)]
struct Job {
    record: bool,
    dir: String,
    calls: Mutex<Vec<String>>,
    claimed: Mutex<Vec<u64>>,
}

impl lk_agents_abi::worker::job::SessionJob for Job {
    fn connect(&self, _cx: &lk_agents_abi::telemetry::Cx) -> BoxFuture<'_, Result<(), HostError>> {
        panic!("joining the job connects no room")
    }
    fn enable_recording(&self) -> Result<bool, HostError> {
        Ok(self.record)
    }
    fn claim_primary(&self, session: HostRef) -> Result<bool, HostError> {
        self.calls.lock().push(format!("claim {}", session.id()));
        let mut claimed = self.claimed.lock();
        claimed.push(session.id());
        Ok(claimed[0] == session.id())
    }
    fn recording_dir(&self) -> Result<String, HostError> {
        self.calls.lock().push("recording_dir".into());
        Ok(self.dir.clone())
    }
    fn init_recording(
        &self,
        options: std::collections::BTreeMap<String, bool>,
    ) -> Result<(), HostError> {
        self.calls.lock().push(format!("init_recording audio={}", options["audio"]));
        Ok(())
    }
    fn on_shutdown(&self, session: HostRef) -> Result<(), HostError> {
        self.calls.lock().push(format!("on_shutdown {}", session.id()));
        Ok(())
    }
}

struct Run;

impl lk_agents_abi::worker::ipc_host::JobExecution for Run {
    fn session_closed(&self, _reason: String) {}
}

/// The host of session `id`, already prepared.
struct JobHost(HostRef);

impl SessionHost for JobHost {
    fn session(&self) -> Result<HostRef, HostError> {
        Ok(self.0.clone())
    }
    fn prepare(
        &self,
        _cx: &lk_agents_abi::telemetry::Cx,
    ) -> BoxFuture<'_, Result<Arc<dyn SessionBindingsPort>, HostError>> {
        panic!("already prepared")
    }
    fn started(&self, _cx: &lk_agents_abi::telemetry::Cx) -> BoxFuture<'_, Result<(), HostError>> {
        panic!("not started")
    }
    fn closed(&self, _cx: &lk_agents_abi::telemetry::Cx) -> BoxFuture<'_, ()> {
        panic!("not started")
    }
    fn bind_providers(
        &self,
        _providers: std::collections::BTreeMap<String, Option<HostRef>>,
    ) -> Result<lk_agents_abi::voice::session_owner::ProviderBindings, HostError> {
        panic!("not started")
    }
}

/// Session `id` joining `job` as it opens: its owner and the arguments it opens on.
fn join(
    job: &Arc<Job>,
    id: u64,
) -> (SessionOwner, Result<(), HostError>, lk_agents_abi::voice::session::VoiceSessionArgs) {
    let registry: Arc<dyn lk_agents_abi::errors::RefRegistry> = Arc::new(Registry);
    let owner = owner();
    let mut args = lk_agents_abi::voice::session::VoiceSessionArgs {
        job: Some(job.clone()),
        job_run: Some(Arc::new(Run)),
        ..Default::default()
    };
    let joined = owner.0.join_job(&JobHost(HostRef::new(id, &registry)), &mut args, false);
    (owner, joined, args)
}

// Catches: a second session of the job taking the job's recording or ending the job as it closes,
// or a session of the job left open at its shutdown.
#[tokio::test]
async fn the_first_session_claimed_is_the_jobs_primary_and_every_one_closes_with_the_job() {
    let job = Arc::new(Job::default());
    let (first, joined, args) = join(&job, 1);
    joined.unwrap();
    assert!(args.job_run.is_some(), "the primary session's close ends the job");
    assert!(args.job.is_some(), "the session connects the job's room");
    assert!(first.started_at().is_some());
    let (second, joined, args) = join(&job, 2);
    joined.unwrap();
    assert!(args.job_run.is_none(), "a later session does not end the job");
    assert_eq!(second.recording_options(), RECORDING_ALL_OFF);
    assert_eq!(second.recording_path(), None);
    assert_eq!(
        *job.calls.lock(),
        [
            "claim 1",
            "init_recording audio=false",
            "on_shutdown 1",
            "claim 2",
            "init_recording audio=false",
            "on_shutdown 2"
        ]
    );
}

// Catches: a recording job's audio recorded for a session other than its primary one, or opened
// anywhere but the job's session directory.
#[tokio::test]
async fn only_the_primary_session_records_what_the_job_records() {
    let dir = std::env::temp_dir().join(format!("owner-recording-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let job = Arc::new(Job { record: true, dir: dir.display().to_string(), ..Default::default() });
    let (first, joined, _) = join(&job, 1);
    assert!(first.recording_options().audio);
    let (second, second_joined, _) = join(&job, 2);
    second_joined.unwrap();
    assert!(!second.recording_options().audio);
    let path = dir.join("audio.ogg").display().to_string();
    #[cfg(feature = "media-ffmpeg")]
    {
        joined.unwrap();
        assert_eq!(first.recording_path(), Some(path));
        first.aclose("user_initiated".into(), false).unwrap().await;
    }
    #[cfg(not(feature = "media-ffmpeg"))]
    {
        let _ = path;
        assert!(joined.is_err(), "recording needs the ffmpeg build");
    }
    let calls = job.calls.lock().clone();
    let recorded: Vec<_> = calls.iter().filter(|call| *call == "recording_dir").collect();
    assert_eq!(recorded.len(), 1, "only the primary session opens a recording: {calls:?}");
    let _ = std::fs::remove_dir_all(dir);
}

// Catches: a session's report read by a job other than the one it first started in, or a redacted
// session's report keeping the content of its events.
#[tokio::test]
async fn the_report_belongs_to_the_starting_job_and_a_redacted_one_keeps_no_content() {
    let owner = owner();
    let info = |id: &str| lk_agents_abi::voice::session_owner::SessionJobInfo {
        id: id.into(),
        dispatch_id: None,
        worker_id: None,
        agent_id: None,
    };
    let host = Arc::new(FailingHost::default());
    let opening = owner
        .start(storage(), host.clone(), Some(info("owner")), lk_agents_abi::telemetry::Cx::NONE)
        .unwrap();
    assert!(owner.allows_report("owner".into()) && !owner.allows_report("other".into()));
    host.release.notify_one();
    assert!(opening.await.is_err());

    assert!(owner.admit_event("user_input_transcribed".into()));
    assert!(!owner.accepts_event("not_an_event".into()));
    let redacted = RecordingOptions { redaction: true, ..RECORDING_ALL_OFF };
    assert!(owner.0.recording_options.set(redacted).is_ok());
    assert!(!owner.accepts_event("user_input_transcribed".into()));
    assert!(owner.accepts_event("error".into()) && owner.accepts_event("close".into()));
}
