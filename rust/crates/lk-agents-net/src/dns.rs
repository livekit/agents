//! Name resolution: the system resolver (`getaddrinfo`, so hosts files and the platform's resolver
//! configuration apply) on the blocking I/O pool, never on Tokio's blocking pool. Every HTTP
//! client and socket connect of the framework resolves here.

use std::io;
use std::net::{SocketAddr, ToSocketAddrs};

/// The addresses `addr` (`(host, port)` or `"host:port"`) resolves to.
pub async fn lookup(addr: impl ToSocketAddrs + Send + 'static) -> io::Result<Vec<SocketAddr>> {
    crate::blocking_io::blocking_io()
        .run((), move || addr.to_socket_addrs().map(Iterator::collect))
        .await
        .map_err(io::Error::other)?
}

/// A TCP connection to the first address of `addr` that accepts, with Nagle off: a small request
/// after a small frame must not wait for the peer's delayed ack.
pub async fn connect(addr: impl ToSocketAddrs + Send + 'static) -> io::Result<tokio::net::TcpStream> {
    let addrs = lookup(addr).await?;
    let stream = tokio::net::TcpStream::connect(addrs.as_slice()).await?;
    stream.set_nodelay(true)?;
    Ok(stream)
}

/// [`lookup`] as reqwest's resolver.
pub struct Resolver;

impl reqwest::dns::Resolve for Resolver {
    fn resolve(&self, name: reqwest::dns::Name) -> reqwest::dns::Resolving {
        let host = name.as_str().to_owned();
        Box::pin(async move {
            let addrs = lookup((host, 0)).await?;
            Ok(Box::new(addrs.into_iter()) as reqwest::dns::Addrs)
        })
    }
}
