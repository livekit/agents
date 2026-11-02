//! The production job runner bounds the whole report callback and still finishes shutdown.

use super::*;
use lk_agents_abi::BoxFuture;
use lk_agents_abi::errors::RefRegistry;
use lk_agents_abi::telemetry::policy::UPLOAD_TIMEOUT_SECONDS;
use lk_agents_abi::worker::ipc_host::JobRun as _;
use lk_agents_abi::worker::job::StartJobFields;
use serde_json::{Map, Value};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

#[derive(Default)]
struct Objects {
    next: AtomicU64,
    values: Mutex<BTreeMap<u64, String>>,
}
impl RefRegistry for Objects {
    fn release(&self, id: u64) {
        self.values.lock().remove(&id);
    }
}
impl Objects {
    fn insert(self: &Arc<Self>, value: String) -> HostRef {
        let id = self.next.fetch_add(1, Ordering::Relaxed) + 1;
        self.values.lock().insert(id, value);
        let registry: Arc<dyn RefRegistry> = self.clone();
        HostRef::new(id, &registry)
    }
}

struct Host {
    objects: Arc<Objects>,
    segment: Duration,
    reporting: tokio::sync::Notify,
    cancelled: Arc<AtomicBool>,
    events: Mutex<Vec<&'static str>>,
    spans: Mutex<Vec<(String, bool)>>,
    logs: Mutex<Vec<String>>,
}
struct ReportCleanup(Arc<AtomicBool>, bool);
impl Drop for ReportCleanup {
    fn drop(&mut self) {
        if self.1 {
            self.0.store(true, Ordering::Release);
        }
    }
}
impl JobProcess for Host {
    fn ping(&self) -> BoxFuture<'_, Result<(), HostError>> {
        Box::pin(async { Ok(()) })
    }
    fn start_job(
        &self,
        _: StartJobFields,
    ) -> BoxFuture<'_, Result<Arc<dyn JobExecution>, HostError>> {
        unreachable!("this test starts the admitted job directly")
    }
    fn start_entrypoint<'a>(
        &'a self,
        _: &'a CallContext,
    ) -> BoxFuture<'a, Result<HostRef, HostError>> {
        Box::pin(async { Ok(self.objects.insert("entrypoint".into())) })
    }
    fn wait<'a>(&'a self, _: &'a CallContext, _: HostRef) -> BoxFuture<'a, Result<(), HostError>> {
        Box::pin(async { Ok(()) })
    }
    fn cancel<'a>(
        &'a self,
        _: &'a CallContext,
        _: HostRef,
    ) -> BoxFuture<'a, Result<(), HostError>> {
        Box::pin(async { Ok(()) })
    }
    fn shutdown_began<'a>(
        &'a self,
        _: &'a CallContext,
        _: String,
        _: bool,
    ) -> BoxFuture<'a, Result<(), HostError>> {
        Box::pin(async { Ok(()) })
    }
    fn close_session<'a>(&'a self, _: &'a CallContext) -> BoxFuture<'a, Result<(), HostError>> {
        Box::pin(async { Ok(()) })
    }
    fn session_end<'a>(
        &'a self,
        _: &'a CallContext,
        _: HostRef,
    ) -> BoxFuture<'a, Result<(), HostError>> {
        Box::pin(async { Ok(()) })
    }
    fn upload<'a>(
        &'a self,
        _: &'a CallContext,
        _: HostRef,
    ) -> BoxFuture<'a, Result<(), HostError>> {
        Box::pin(async {
            let mut cleanup = ReportCleanup(self.cancelled.clone(), true);
            self.reporting.notify_one();
            tokio::time::sleep(self.segment).await;
            self.events.lock().push("prepared");
            tokio::time::sleep(self.segment).await;
            self.events.lock().push("uploaded");
            cleanup.1 = false;
            Ok(())
        })
    }
    fn disconnect<'a>(
        &'a self,
        _: &'a CallContext,
        _: HostRef,
    ) -> BoxFuture<'a, Result<(), HostError>> {
        Box::pin(async {
            self.events.lock().push("disconnected");
            Ok(())
        })
    }
    fn shutdown_callback<'a>(
        &'a self,
        _: &'a CallContext,
        _: HostRef,
        _: Option<String>,
        _: Option<HostRef>,
    ) -> BoxFuture<'a, Result<(), HostError>> {
        Box::pin(async {
            self.events.lock().push("callback");
            Ok(())
        })
    }
    fn span(&self, name: String, _: Option<String>) -> Result<HostRef, HostError> {
        Ok(self.objects.insert(name))
    }
    fn span_end(&self, span: HostRef, _: Option<HostError>, late: bool) -> Result<(), HostError> {
        let name = self.objects.values.lock()[&span.id()].clone();
        self.spans.lock().push((name, late));
        Ok(())
    }
    fn log(
        &self,
        _: i32,
        message: String,
        _: Map<String, Value>,
        _: Option<HostError>,
    ) -> Result<(), HostError> {
        self.logs.lock().push(message);
        Ok(())
    }
}

#[tokio::test(start_paused = true)]
async fn report_preparation_and_upload_share_one_budget_and_shutdown_always_continues() {
    for (segment, late) in
        [(UPLOAD_TIMEOUT_SECONDS * 2 / 3, true), (UPLOAD_TIMEOUT_SECONDS / 4, false)]
    {
        let host = Arc::new(Host {
            objects: Arc::default(),
            segment: Duration::from_secs(segment),
            reporting: tokio::sync::Notify::new(),
            cancelled: Arc::default(),
            events: Mutex::new(Vec::new()),
            spans: Mutex::new(Vec::new()),
            logs: Mutex::new(Vec::new()),
        });
        let run = JobRun::new(
            &ProcessEnv::current_runtime(),
            true,
            5.0,
            false,
            "ws://localhost".into(),
            false,
            None,
            5.0,
            None,
        )
        .unwrap();
        run.add_shutdown_callback(
            host.objects.insert("callback".into()),
            false,
            false,
            "done".into(),
        );
        run.shutdown("test".into(), true);
        let running = tokio::spawn({
            let host = host.clone();
            async move { run.run(host, CallContext::default()).await }
        });
        host.reporting.notified().await;
        tokio::time::timeout(Duration::from_secs(UPLOAD_TIMEOUT_SECONDS + 1), running)
            .await
            .expect("report callback restarted the job's budget")
            .unwrap()
            .unwrap();
        assert_eq!(host.cancelled.load(Ordering::Acquire), late);
        let events = host.events.lock().clone();
        assert_eq!(
            events,
            if late {
                vec!["prepared", "disconnected", "callback"]
            } else {
                vec!["prepared", "uploaded", "disconnected", "callback"]
            }
        );
        assert!(host.spans.lock().contains(&("session_end_upload".into(), late)));
        if late {
            assert!(
                host.logs.lock().iter().any(|line| line.contains("session_end_upload timed out"))
            );
        }
        assert!(host.objects.values.lock().is_empty(), "shutdown retained a host object");
    }
}
