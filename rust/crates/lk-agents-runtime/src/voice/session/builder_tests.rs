use super::super::tests::{UnusedRemotePort, close_session, no_room, settle, unobserved};
use super::super::*;
use crate::voice::{Executor, remote_session};
use lk_agents_abi::errors::{HostError, ProviderError};
use lk_agents_abi::telemetry::Cx;
use lk_agents_abi::voice::session::VoiceSession as _;
use std::future::Future as _;
use std::sync::atomic::{AtomicBool, Ordering};

/// A remote host whose transport records whether opening was attempted after session close.
struct RemoteTransport {
    started: AtomicBool,
    closed: AtomicBool,
}

impl lk_agents_abi::voice::console::SessionTransport for RemoteTransport {
    fn start(&self) -> lk_agents_abi::BoxFuture<'_, Result<(), HostError>> {
        Box::pin(async move {
            self.started.store(true, Ordering::SeqCst);
            Ok(())
        })
    }

    fn send_message<'a>(
        &'a self,
        _message: &'a livekit_protocol::agent::AgentSessionMessage,
    ) -> lk_agents_abi::BoxFuture<'a, Result<(), HostError>> {
        Box::pin(async { Ok(()) })
    }

    fn recv(
        &self,
    ) -> lk_agents_abi::BoxFuture<
        '_,
        Result<Option<livekit_protocol::agent::AgentSessionMessage>, HostError>,
    > {
        Box::pin(std::future::pending())
    }

    fn close(&self) -> lk_agents_abi::BoxFuture<'_, ()> {
        Box::pin(async move { self.closed.store(true, Ordering::SeqCst) })
    }
}

#[tokio::test(start_paused = true)]
async fn close_during_opening_retires_only_the_accepted_remote_host() {
    let api = unobserved(no_room(), None);
    let make_host = || {
        let transport = Arc::new(RemoteTransport {
            started: AtomicBool::new(false),
            closed: AtomicBool::new(false),
        });
        let host = Arc::new(remote_session::driver::RemoteSessionLink::new(
            Arc::new(UnusedRemotePort),
            transport.clone(),
            None,
            api.time().clone(),
            Arc::new(lk_agents_net::task::TokioSpawner::current()),
        ));
        (host, transport)
    };
    let (accepted, accepted_transport) = make_host();
    let (refused, refused_transport) = make_host();
    let opening = api
        .open(
            Box::new(Executor::new(api.time().clone(), api.spawner())),
            Some(accepted.clone()),
            |_| Box::pin(async {}),
        )
        .expect("the first opening is accepted");
    assert!(matches!(
        api.open(
            Box::new(Executor::new(api.time().clone(), api.spawner())),
            Some(refused.clone()),
            |_| Box::pin(async {})
        ),
        Err(OpenRefused::AlreadyStarted),
    ));
    api.aclose("user_initiated".into(), false).unwrap().await;
    // The still-pending startup must not reopen this host after close has returned.
    accepted.start().await.unwrap();
    refused.start().await.unwrap();
    let accepted_started = accepted_transport.started.load(Ordering::SeqCst);
    let refused_started = refused_transport.started.load(Ordering::SeqCst);
    accepted.aclose().await;
    refused.aclose().await;
    drop(opening);
    assert!(!accepted_started, "closing during opening lost the accepted remote host");
    assert!(refused_started, "a refused opening closed a host the session never owned");
    assert!(refused_transport.closed.load(Ordering::SeqCst));
}

#[tokio::test(start_paused = true)]
async fn an_opening_polled_after_close_cannot_reinstall_running_resources() {
    let api = unobserved(no_room(), None);
    let opening = api
        .open(Box::new(Executor::new(api.time().clone(), api.spawner())), None, |_| {
            Box::pin(async {})
        })
        .unwrap();
    api.aclose("user_initiated".into(), false).unwrap().await;
    let result = opening.await;
    assert!(result.is_err(), "startup succeeded after close had completed");
    assert!(matches!(api.api(), Err(lk_agents_abi::voice::session::SessionRefused::Closed)));
}

#[tokio::test(start_paused = true)]
async fn closing_between_task_launch_and_installation_retires_the_opening_execution() {
    let api = Arc::new(unobserved(no_room(), None));
    let closing = api.clone();
    let opening = api
        .open(Box::new(Executor::new(api.time().clone(), api.spawner())), None, move |_| {
            // Synchronous close admission wins before the prepared execution is installed.
            drop(closing.close(CloseReason::UserInitiated, Default::default(), None));
            Box::pin(async {})
        })
        .unwrap();
    assert!(matches!(opening.await, Err(DriverError::StartupClosed)));
    settle().await;
    assert!(matches!(api.api(), Err(lk_agents_abi::voice::session::SessionRefused::Closed)));
    assert!(api.tasks.names().is_empty(), "uninstalled execution left a loop task alive");
    assert!(api.resources.live().is_empty());
}

#[tokio::test(start_paused = true)]
async fn refused_task_admission_and_dropped_startup_retire_the_opening_phase() {
    for refuse_tasks in [false, true] {
        let api = unobserved(no_room(), None);
        let opening = api
            .open(Box::new(Executor::new(api.time().clone(), api.spawner())), None, |_| {
                Box::pin(async {})
            })
            .unwrap();
        if refuse_tasks {
            api.tasks.close();
            assert!(matches!(opening.await, Err(DriverError::StartupClosed)));
        } else {
            drop(opening);
        }
        settle().await;
        assert!(matches!(api.api(), Err(lk_agents_abi::voice::session::SessionRefused::Closed)));
        assert!(api.tasks.names().is_empty());
        assert!(api.resources.live().is_empty());
    }
}

struct WaitingVad;
impl lk_agents_abi::Vad for WaitingVad {
    fn label(&self) -> String {
        "waiting-vad".into()
    }
    fn capabilities(&self) -> lk_agents_abi::VadCapabilities {
        Default::default()
    }
    fn stream(
        &self,
        _cx: &Cx,
    ) -> lk_agents_abi::BoxFuture<
        '_,
        Result<
            lk_agents_abi::DuplexStream<lk_agents_abi::VadSink, lk_agents_abi::VadSource>,
            ProviderError,
        >,
    > {
        Box::pin(std::future::pending())
    }
}

#[tokio::test(start_paused = true)]
async fn close_cancels_pending_provider_preparation_without_waiting_for_the_provider() {
    let env = lk_agents_net::env::ProcessEnv::current_runtime();
    let args = lk_agents_abi::voice::session::VoiceSessionArgs {
        agent: "waiting-provider".into(),
        vad: Some(Arc::new(WaitingVad)),
        ..Default::default()
    };
    let api = VoiceSession::new(&env, Arc::new(no_room()), args, None).unwrap();
    let mut opening = Box::pin(
        api.open(Box::new(Executor::new(api.time().clone(), api.spawner())), None, |_| {
            Box::pin(async {})
        })
        .unwrap(),
    );
    std::future::poll_fn(|cx| {
        assert!(opening.as_mut().poll(cx).is_pending());
        std::task::Poll::Ready(())
    })
    .await;
    close_session(&api).await;
    assert!(matches!(opening.await, Err(DriverError::StartupClosed)));
    assert!(api.resources.live().is_empty());
    api.close_io().await;
}

#[tokio::test(start_paused = true)]
async fn closing_before_start_retires_ready_resources_and_permanently_refuses_open() {
    let api = unobserved(no_room(), None);
    api.aclose("user_initiated".into(), false).unwrap().await;
    assert!(matches!(
        api.open(Box::new(Executor::new(api.time().clone(), api.spawner())), None, |_| Box::pin(
            async {}
        ),),
        Err(OpenRefused::Closed)
    ));
}
