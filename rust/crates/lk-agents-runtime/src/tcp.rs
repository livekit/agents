//! Raw TCP connections on the hosting process's reactor, exposed as generic byte IO ports.

use lk_agents_net::task::TokioSpawner;
use lk_agents_abi::BoxFuture;
use lk_agents_abi::SpawnerExt;
use std::io;
use tokio::io::{AsyncRead, AsyncWrite};

/// Independent halves, so a consumer can read while another task writes.
pub struct Connection {
    pub read: Box<dyn AsyncRead + Unpin + Send>,
    pub write: Box<dyn AsyncWrite + Unpin + Send>,
}

pub trait Connector: Send + Sync {
    fn connect<'a>(&'a self, addr: &'a str) -> BoxFuture<'a, io::Result<Connection>>;
}

/// Connects on the reactor `runtime` returns, called at connection so constructing a client
/// starts nothing.
pub struct Tcp {
    runtime: fn() -> Result<tokio::runtime::Handle, String>,
}

impl Tcp {
    pub fn new(runtime: fn() -> Result<tokio::runtime::Handle, String>) -> Self {
        Self { runtime }
    }
}

impl Connector for Tcp {
    fn connect<'a>(&'a self, addr: &'a str) -> BoxFuture<'a, io::Result<Connection>> {
        Box::pin(async move {
            let runtime = (self.runtime)().map_err(io::Error::other)?;
            connect_on(&runtime, addr.to_owned()).await
        })
    }
}

async fn connect_on(runtime: &tokio::runtime::Handle, addr: String) -> io::Result<Connection> {
    // Opened on the owning reactor, wherever the caller polls; dropping the wait cancels it.
    let connecting = TokioSpawner::new(runtime.clone()).spawn(async move {
        let stream = lk_agents_net::dns::connect(addr).await?;
        let (read, write) = stream.into_split();
        Ok(Connection { read: Box::new(read), write: Box::new(write) })
    });
    connecting.await.map_err(|_| io::Error::other("TCP connector stopped"))?
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::future::Future;
    use std::sync::Arc;
    use std::task::{Context, Poll, Wake, Waker};
    use std::time::Duration;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    /// Poll without entering Tokio: the byte ports must keep using their owning reactor.
    fn outside_reactor<T>(future: impl Future<Output = T>) -> T {
        struct Thread(std::thread::Thread);
        impl Wake for Thread {
            fn wake(self: Arc<Self>) {
                self.0.unpark();
            }
            fn wake_by_ref(self: &Arc<Self>) {
                self.0.unpark();
            }
        }
        assert!(tokio::runtime::Handle::try_current().is_err());
        let waker = Waker::from(Arc::new(Thread(std::thread::current())));
        let mut context = Context::from_waker(&waker);
        let mut future = std::pin::pin!(future);
        loop {
            match future.as_mut().poll(&mut context) {
                Poll::Ready(value) => return value,
                Poll::Pending => std::thread::park(),
            }
        }
    }

    #[test]
    fn byte_ports_connect_and_exchange_without_a_caller_reactor() {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .enable_all()
            .build()
            .unwrap();
        let listener = runtime.block_on(tokio::net::TcpListener::bind("127.0.0.1:0")).unwrap();
        let addr = listener.local_addr().unwrap().to_string();
        let server = runtime.spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut bytes = [0; 4];
            socket.read_exact(&mut bytes).await.unwrap();
            assert_eq!(&bytes, b"ping");
            socket.write_all(b"pong").await.unwrap();
        });
        outside_reactor(async {
            let mut connection = connect_on(runtime.handle(), addr).await.unwrap();
            connection.write.write_all(b"ping").await.unwrap();
            let mut bytes = [0; 4];
            connection.read.read_exact(&mut bytes).await.unwrap();
            assert_eq!(&bytes, b"pong");
            connection.write.shutdown().await.unwrap();
        });
        runtime.block_on(server).unwrap();
    }

    #[test]
    fn dropping_a_pending_connector_cancels_its_dispatched_socket_open() {
        let runtime = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        let listener = runtime.block_on(tokio::net::TcpListener::bind("127.0.0.1:0")).unwrap();
        let addr = listener.local_addr().unwrap().to_string();
        let mut opening = Box::pin(connect_on(runtime.handle(), addr));
        let mut context = Context::from_waker(Waker::noop());
        assert!(opening.as_mut().poll(&mut context).is_pending());
        drop(opening);
        runtime.block_on(async {
            tokio::task::yield_now().await;
            assert!(
                tokio::time::timeout(Duration::from_millis(10), listener.accept()).await.is_err(),
                "a cancelled connector opened a detached connection"
            );
        });
    }
}
