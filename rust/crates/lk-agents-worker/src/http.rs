//! Incoming HTTP streams: preamble, metadata and opaque bidirectional bytes for the host.

use crate::transport::{AgentEndpointKind, EndpointStream as TransportStream};
use crate::{StreamPreamble, WorkerHost};
use lk_agents_abi::errors::{HostError, HostResult};
use lk_agents_abi::worker::http::{
    ByteStream, ByteTransport, EndpointHandler, EndpointKind, EndpointRequest as RequestMetadata,
    HttpScheme,
};
use lk_agents_abi::worker::ipc_host::WorkerError;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;

use crate::transport::{
    EndpointControl, EndpointReader, EndpointWriter, HttpStreamResetCode, TransportError,
};

/// The largest single `read`; a body streams through many.
const MAX_READ: usize = 1 << 20;

/// One exchange stream after its preamble. The two directions are independent: a `read` and a
/// `write` may be in flight at once. Dropping it without `finish` resets with `HSR_ABORT`.
pub struct HttpStream {
    reader: Mutex<EndpointReader>,
    writer: Mutex<EndpointWriter>,
    control: EndpointControl,
}

impl HttpStream {
    pub fn new(reader: EndpointReader, writer: EndpointWriter) -> Self {
        let control = writer.control();
        Self { reader: Mutex::new(reader), writer: Mutex::new(writer), control }
    }

    /// The host never held the stream (`refused`), or failed with it (`internal`).
    pub fn fail(&self, code: HttpStreamResetCode) {
        self.reset_with(code);
    }

    async fn read_some(&self, max_bytes: usize) -> Result<Vec<u8>, TransportError> {
        let mut reader = self.reader.lock().await;
        let mut buf = vec![0u8; max_bytes.clamp(1, MAX_READ)];
        match reader.read(&mut buf).await? {
            Some(n) => {
                buf.truncate(n);
                Ok(buf)
            }
            None => Ok(Vec::new()),
        }
    }

    async fn write_all(&self, data: &[u8]) -> Result<(), TransportError> {
        self.writer.lock().await.write_all(data).await
    }

    /// Control is admitted independently of pending data I/O: applied now when the half is
    /// free, else by the I/O holding it.
    fn apply_writer(&self) -> Result<(), TransportError> {
        match self.writer.try_lock() {
            Ok(mut writer) => writer.apply_control(),
            Err(_) => Ok(()),
        }
    }

    fn apply_reader(&self) {
        if let Ok(mut reader) = self.reader.try_lock() {
            reader.apply_control();
        }
    }

    fn reset_with(&self, code: HttpStreamResetCode) {
        self.control.reset(code);
        let _ = self.apply_writer();
        self.stop_with(code);
    }

    fn stop_with(&self, code: HttpStreamResetCode) {
        self.control.stop(code);
        self.apply_reader();
    }
}

fn reset_code(code: i32) -> Result<HttpStreamResetCode, WorkerError> {
    HttpStreamResetCode::try_from(code)
        .map_err(|_| WorkerError::Invalid(format!("unknown HttpStreamResetCode {code}")))
}

impl ByteStream for HttpStream {
    async fn read(&self, max_bytes: usize) -> Result<Vec<u8>, WorkerError> {
        self.read_some(max_bytes).await.map_err(|error| WorkerError::Failed(error.to_string()))
    }

    async fn write(&self, data: Vec<u8>) -> Result<(), WorkerError> {
        self.write_all(&data).await.map_err(|error| WorkerError::Failed(error.to_string()))
    }

    fn finish(&self) -> Result<(), WorkerError> {
        self.control.finish();
        self.apply_writer().map_err(|error| WorkerError::Failed(error.to_string()))
    }

    fn reset(&self, code: i32) -> Result<(), WorkerError> {
        self.reset_with(reset_code(code)?);
        Ok(())
    }

    fn stop(&self, code: i32) -> Result<(), WorkerError> {
        self.stop_with(reset_code(code)?);
        Ok(())
    }
}

/// The native stream wherever the host passes an exchange stream.
impl ByteTransport for HttpStream {
    fn read(&self, max_bytes: usize) -> lk_agents_abi::BoxFuture<'_, HostResult<Vec<u8>>> {
        Box::pin(async move {
            self.read_some(max_bytes).await.map_err(|error| HostError::new(error.to_string()))
        })
    }

    fn write(&self, data: Vec<u8>) -> lk_agents_abi::BoxFuture<'_, HostResult> {
        Box::pin(async move {
            self.write_all(&data).await.map_err(|error| HostError::new(error.to_string()))
        })
    }

    fn finish(&self) {
        self.control.finish();
        let _ = self.apply_writer();
    }

    fn reset(&self, code: HttpStreamResetCode) {
        self.reset_with(code);
    }

    fn stop(&self, code: HttpStreamResetCode) {
        self.stop_with(code);
    }
}

/// `wire.MaxPreambleSize` is small, so a peer that opens a stream and sends nothing is the only
/// way to wait here; the node writes the preamble in the same call as the request head.
const PREAMBLE_TIMEOUT: Duration = Duration::from_secs(30);

pub(crate) async fn accept<P: WorkerHost + ?Sized>(host: Arc<P>, mut stream: TransportStream) {
    let preamble = match tokio::time::timeout(PREAMBLE_TIMEOUT, stream.read_preamble()).await {
        Ok(Ok(preamble)) => preamble,
        Ok(Err(error)) => {
            host.logger().emit(
                log::Level::Warn,
                format!("endpoint stream without a valid preamble error={error}"),
            );
            stream.reset(HttpStreamResetCode::HsrProtocol);
            return;
        }
        Err(_) => {
            host.logger().emit(log::Level::Warn, "endpoint stream sent no preamble in time".into());
            stream.reset(HttpStreamResetCode::HsrProtocol);
            return;
        }
    };
    host.serve_endpoint(IncomingRequest { preamble, stream }).await;
}

/// The node's metadata and the stream positioned at the first application byte.
#[derive(Debug)]
pub struct IncomingRequest {
    pub preamble: StreamPreamble,
    pub stream: TransportStream,
}

/// Pass the request's bytes and metadata to the binding's application server.
pub(crate) async fn dispatch(request: IncomingRequest, handler: Option<&dyn EndpointHandler>) {
    let IncomingRequest { preamble, stream } = request;
    let Some(endpoints) = handler else {
        return stream.refuse();
    };
    let info = match metadata(preamble) {
        Ok(info) => info,
        Err(error) => {
            log::warn!(target: crate::LOG, "invalid HTTP stream metadata: {error}");
            stream.reset(HttpStreamResetCode::HsrProtocol);
            return;
        }
    };
    let (writer, reader) = stream.split();
    let stream = Arc::new(HttpStream::new(reader, writer));
    if let Err(error) = endpoints.handle(stream.clone(), info).await {
        log::error!(target: crate::LOG, "endpoint host failed: {error}");
        stream.fail(HttpStreamResetCode::HsrInternal);
    }
}

fn metadata(preamble: StreamPreamble) -> Result<RequestMetadata, TransportError> {
    let kind = match AgentEndpointKind::try_from(preamble.kind) {
        Ok(AgentEndpointKind::AekHttp) => EndpointKind::Http,
        Ok(AgentEndpointKind::AekText) => EndpointKind::Text,
        _ => {
            return Err(TransportError::Protocol(format!(
                "unknown endpoint kind {}",
                preamble.kind
            )));
        }
    };
    let scheme = match preamble.scheme.as_str() {
        "http" => HttpScheme::Http,
        "https" => HttpScheme::Https,
        scheme => {
            return Err(TransportError::Protocol(format!("unknown client scheme {scheme:?}")));
        }
    };
    Ok(RequestMetadata {
        request_id: preamble.request_id,
        authorized: preamble.authorized,
        route: preamble.route,
        kind,
        client_addr: preamble.client_addr,
        scheme,
        timeout_ms: u64::from(preamble.timeout_ms),
        unauthenticated: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metadata_preserves_the_nodes_values_and_uses_typed_kinds_and_schemes() {
        for (wire, kind) in [
            (AgentEndpointKind::AekHttp, EndpointKind::Http),
            (AgentEndpointKind::AekText, EndpointKind::Text),
        ] {
            let info = metadata(StreamPreamble {
                request_id: "AER_test".into(),
                route: "/items/{id}".into(),
                kind: wire as i32,
                scheme: "https".into(),
                client_addr: "2001:db8::1".into(),
                authorized: true,
                timeout_ms: 1234,
            })
            .unwrap();
            assert_eq!(info.request_id, "AER_test");
            assert_eq!(info.route, "/items/{id}");
            assert_eq!(info.kind, kind);
            assert_eq!(info.scheme, HttpScheme::Https);
            assert_eq!(info.client_addr, "2001:db8::1");
            assert!(info.authorized);
            assert_eq!(info.timeout_ms, 1234);
        }
    }

    #[test]
    fn unknown_metadata_is_rejected_instead_of_becoming_http() {
        for (kind, scheme) in [
            (i32::MAX, "https"),
            (AgentEndpointKind::AekHttp as i32, ""),
            (AgentEndpointKind::AekHttp as i32, "ftp"),
        ] {
            assert!(matches!(
                metadata(StreamPreamble { kind, scheme: scheme.into(), ..Default::default() }),
                Err(TransportError::Protocol(_))
            ));
        }
        assert!(reset_code(i32::MAX).is_err());
        assert_eq!(
            reset_code(HttpStreamResetCode::HsrAbort as i32).unwrap(),
            HttpStreamResetCode::HsrAbort
        );
    }
}
