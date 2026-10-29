//! The supervisor's real writer task reporting a failed in-flight health ping.

use std::pin::Pin;
use std::task::{Context, Poll};

use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};
use tokio::sync::oneshot;

use super::*;

struct FailPing {
    socket: DuplexStream,
    started: Option<oneshot::Sender<()>>,
    release: oneshot::Receiver<()>,
    panic_writer: bool,
}

impl AsyncRead for FailPing {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buffer: &mut ReadBuf<'_>,
    ) -> Poll<std::io::Result<()>> {
        Pin::new(&mut self.socket).poll_read(cx, buffer)
    }
}

impl AsyncWrite for FailPing {
    fn poll_write(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        bytes: &[u8],
    ) -> Poll<std::io::Result<usize>> {
        if bytes.len() >= 4 && matches!(decode::<ToChild>(&bytes[4..]), Ok(ToChild::Ping { .. })) {
            if let Some(started) = self.started.take() {
                let _ = started.send(());
            }
            if Pin::new(&mut self.release).poll(cx).is_pending() {
                return Poll::Pending;
            }
            assert!(!self.panic_writer, "fixture IPC writer panic");
            return Poll::Ready(Err(std::io::Error::new(
                std::io::ErrorKind::BrokenPipe,
                "peer finished its control connection",
            )));
        }
        Pin::new(&mut self.socket).poll_write(cx, bytes)
    }

    fn poll_flush(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        Pin::new(&mut self.socket).poll_flush(cx)
    }

    fn poll_shutdown(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        Pin::new(&mut self.socket).poll_shutdown(cx)
    }
}

async fn finish_with_failed_ping(
    panic_writer: bool,
) -> (lk_agents_abi::worker::ipc::ProcessCompletion, usize) {
    let (socket, mut child) = tokio::io::duplex(1 << 16);
    let (started, ping_started) = oneshot::channel();
    let (release, fail_write) = oneshot::channel();
    let stream = FailPing { socket, started: Some(started), release: fail_write, panic_writer };
    let host = Arc::new(FakeHost::default());
    host.alive.store(true, Ordering::SeqCst);
    let sup =
        ProcSupervisor::new(stream, host.clone(), opts(), crate::test_support::blocking_io());
    let init = waits(&sup).await;
    assert!(matches!(recv(&mut child).await, ToChild::Hello(_)));
    write(&mut child, &ToParent::Ready { version: PROTOCOL_VERSION, error: None }).await.unwrap();
    init.await.unwrap().unwrap().unwrap();

    sup.supervise().await;
    ping_started.await.unwrap();
    write(&mut child, &ToParent::ShuttingDown).await.unwrap();
    write(&mut child, &ToParent::Exiting { reason: "writer close regression".into() })
        .await
        .unwrap();
    // The following log proves the driver handled ShuttingDown before the held write fails.
    while host.find("process exiting").is_none() {
        host.logged.notified().await;
    }
    release.send(()).unwrap();
    tokio::time::sleep(Duration::from_millis(10)).await;
    let kills = host.kills.load(Ordering::SeqCst);

    // The OS and start-method receipts remain withheld until the writer outcome was observed.
    drop(child);
    sup.exited(Some(0)).await;
    sup.reaped(0).await;
    sup.wait_cleanup().await;
    (sup.completion(Some(0)), kills)
}

#[tokio::test(start_paused = true)]
async fn reported_ping_write_failure_after_shutdown_does_not_become_a_second_protocol_failure() {
    let (completion, kills) = finish_with_failed_ping(false).await;
    assert_eq!(kills, 0, "a correlated expected write failure must not kill a finalizing child");
    assert_eq!(completion.exit_code, Some(0));
    assert!(completion.failure.is_none(), "{:?}", completion.failure);
}

#[tokio::test(start_paused = true)]
async fn unexplained_writer_termination_is_still_a_terminal_protocol_failure() {
    let (completion, kills) = finish_with_failed_ping(true).await;
    assert_eq!(kills, 1, "a writer that vanishes without an outcome must start cleanup");
    assert!(matches!(
        completion.failure,
        Some(lk_agents_abi::worker::ipc::IpcFailure::Unexpected { message })
            if message == "IPC writer closed"
    ));
}
