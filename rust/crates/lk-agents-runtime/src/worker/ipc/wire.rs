//! Runtime framing over the worker core's typed IPC protocol.

pub use lk_agents_worker::ipc::wire::*;
use serde::{Serialize, de::DeserializeOwned};
use std::io;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

/// Capture the transport cause before crossing into the decision core.
pub(super) fn io_failure(error: io::Error) -> WireError {
    WireError::Io { kind: error.kind(), os_error: error.raw_os_error(), message: error.to_string() }
}

/// The next message.
pub async fn read<T: DeserializeOwned, R: AsyncRead + Unpin + ?Sized>(
    r: &mut R,
) -> Result<T, WireError> {
    let mut len = [0u8; 4];
    let first = r.read(&mut len[..1]).await.map_err(io_failure)?;
    if first == 0 {
        return Err(WireError::Eof);
    }
    r.read_exact(&mut len[1..]).await.map_err(io_failure)?;
    let len = u32::from_be_bytes(len) as usize;
    if len > MAX_FRAME {
        return Err(WireError::TooLarge(len));
    }
    let mut buf = vec![0; len];
    r.read_exact(&mut buf).await.map_err(io_failure)?;
    decode(&buf)
}

/// Writes one message.
pub async fn write<T: Serialize, W: AsyncWrite + Unpin + ?Sized>(
    w: &mut W,
    msg: &T,
) -> io::Result<()> {
    // Measure with postcard's non-allocating flavor before constructing the encoded payload.
    let size = postcard::serialize_with_flavor(msg, postcard::ser_flavors::Size::default())
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, error))?;
    if size > MAX_FRAME {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, WireError::TooLarge(size)));
    }
    let data = encode(msg);
    let mut frame = Vec::with_capacity(4 + data.len());
    frame.extend_from_slice(&(data.len() as u32).to_be_bytes());
    frame.extend_from_slice(&data);
    w.write_all(&frame).await?;
    w.flush().await
}
