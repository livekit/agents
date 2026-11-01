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
    fn prepare(&self) -> BoxFuture<'_, Result<Arc<dyn SessionBindingsPort>, HostError>> {
        Box::pin(async move {
            self.prepared.fetch_add(1, Ordering::SeqCst);
            self.preparing.notify_one();
            self.release.notified().await;
            Err(HostError::new("toolset initialization failed"))
        })
    }
    fn started(&self) -> BoxFuture<'_, Result<(), HostError>> {
        Box::pin(async { panic!("failed startup must not announce success") })
    }
    fn closed(&self) -> BoxFuture<'_, ()> {
        Box::pin(async move {
            self.closed.fetch_add(1, Ordering::SeqCst);
        })
    }
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
    let waiting = owner.start(storage.clone(), host.clone()).unwrap();
    host.preparing.notified().await;
    drop(waiting);
    let other = owner.start(storage.clone(), host.clone()).unwrap();
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
    assert!(owner.start(storage, host).is_err());
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
    assert!(owner.start(storage, host.clone()).is_err());
    assert_eq!(host.prepared.load(Ordering::SeqCst), 0);
}
