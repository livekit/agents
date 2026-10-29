//! The endpoint exchange as the host drives it: `_core.EndpointStream` (one stream after its
//! preamble), `_core.EndpointServer` (an exchange served with the host's ASGI application, and
//! the open WebSockets it drains) and the native `receive`/`send` the application calls.

use std::sync::Arc;

use async_trait::async_trait;
use lk_agents_abi::errors::{HostError, HostResult};
use lk_agents_abi::worker::endpoint::EndpointRequest;
use lk_agents_abi::worker::endpoint_host::{
    AsgiApplication, AsgiReceive, AsgiReceived, AsgiReceiver, AsgiSend, AsgiSender, AsgiSent,
    EndpointServer, EndpointService, EndpointStream, ExchangeStream,
};
use lk_agents_abi::worker::ipc_host::WorkerError;
use tokio::sync::Mutex;

use crate::endpoint::{self, AppError, AsgiApp, Conn, Registry, Scope, StreamIo};
use crate::transport::{EndpointControl, EndpointReader, EndpointWriter, HttpStreamResetCode};

/// The largest single `read`; a body streams through many.
const MAX_READ: usize = 1 << 20;

/// One exchange stream after its preamble. The two directions are independent: a `read` and a
/// `write` may be in flight at once. Dropping it without `finish` resets with `HSR_ABORT`.
pub struct NativeStream {
    reader: Mutex<Option<EndpointReader>>,
    writer: Mutex<Option<EndpointWriter>>,
    control: EndpointControl,
}

impl NativeStream {
    pub fn new(reader: EndpointReader, writer: EndpointWriter) -> Self {
        let control = writer.control();
        Self { reader: Mutex::new(Some(reader)), writer: Mutex::new(Some(writer)), control }
    }

    /// The host never held the stream (`refused`), or failed with it (`internal`).
    pub async fn fail(&self, code: HttpStreamResetCode) {
        if let Some(writer) = self.writer.lock().await.as_mut() {
            writer.reset(code);
        }
        if let Some(reader) = self.reader.lock().await.as_mut() {
            reader.stop(code);
        }
    }

    async fn read_some(&self, max_bytes: usize) -> Result<Vec<u8>, String> {
        let mut guard = self.reader.lock().await;
        let reader = guard.as_mut().ok_or("endpoint stream is closed")?;
        let mut buf = vec![0u8; max_bytes.clamp(1, MAX_READ)];
        match reader.read(&mut buf).await.map_err(|e| e.to_string())? {
            Some(n) => {
                buf.truncate(n);
                Ok(buf)
            }
            None => Ok(Vec::new()),
        }
    }

    async fn write_all(&self, data: &[u8]) -> Result<(), String> {
        let mut guard = self.writer.lock().await;
        let writer = guard.as_mut().ok_or("endpoint stream is closed")?;
        writer.write_all(data).await.map_err(|e| e.to_string())
    }

    /// Control is admitted independently of pending data I/O: applied now when the half is
    /// free, else by the I/O holding it.
    fn apply_writer(&self) -> Result<(), String> {
        match self.writer.try_lock() {
            Ok(mut writer) => match writer.as_mut() {
                Some(writer) => writer.apply_control().map_err(|e| e.to_string()),
                None => Ok(()),
            },
            Err(_) => Ok(()),
        }
    }

    fn apply_reader(&self) {
        if let Ok(mut reader) = self.reader.try_lock()
            && let Some(reader) = reader.as_mut()
        {
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

impl EndpointStream for NativeStream {
    async fn read(&self, max_bytes: usize) -> Result<Vec<u8>, WorkerError> {
        self.read_some(max_bytes).await.map_err(WorkerError::Failed)
    }

    async fn write(&self, data: Vec<u8>) -> Result<(), WorkerError> {
        self.write_all(&data).await.map_err(WorkerError::Failed)
    }

    fn finish(&self) -> Result<(), WorkerError> {
        self.control.finish();
        self.apply_writer().map_err(WorkerError::Failed)
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
impl ExchangeStream for NativeStream {
    fn read(&self, max_bytes: usize) -> lk_agents_abi::BoxFuture<'_, HostResult<Vec<u8>>> {
        Box::pin(async move { self.read_some(max_bytes).await.map_err(HostError::new) })
    }

    fn write(&self, data: Vec<u8>) -> lk_agents_abi::BoxFuture<'_, HostResult> {
        Box::pin(async move { self.write_all(&data).await.map_err(HostError::new) })
    }

    fn finish(&self) {
        self.control.finish();
        let _ = self.apply_writer();
    }

    fn reset(&self, code: i32) {
        if let Ok(code) = HttpStreamResetCode::try_from(code) {
            self.reset_with(code);
        }
    }

    fn stop(&self, code: i32) {
        if let Ok(code) = HttpStreamResetCode::try_from(code) {
            self.stop_with(code);
        }
    }
}

/// An exchange stream as the endpoint exchange reads and writes it.
struct Exchange(Arc<dyn ExchangeStream>);

#[async_trait]
impl StreamIo for Exchange {
    async fn read(&self, max: usize) -> Result<Vec<u8>, String> {
        self.0.read(max).await.map_err(|e| e.to_string())
    }

    async fn write(&self, data: &[u8]) -> Result<(), String> {
        self.0.write(data.to_vec()).await.map_err(|e| e.to_string())
    }

    async fn finish(&self) {
        self.0.finish();
    }

    async fn reset(&self, code: HttpStreamResetCode) {
        self.0.reset(code as i32);
    }

    async fn stop(&self, code: HttpStreamResetCode) {
        self.0.stop(code as i32);
    }
}

/// The host's application, called with this exchange's preamble.
struct Application {
    app: Arc<dyn AsgiApplication>,
    request: EndpointRequest,
}

#[async_trait]
impl AsgiApp for Application {
    async fn call(&self, scope: Scope, conn: Conn) -> Result<(), AppError> {
        let receive = Arc::new(ReceiveCall(conn.clone()));
        let send = Arc::new(SendCall(conn));
        match self.app.call(scope, self.request.clone(), receive, send).await {
            Ok(None) => Ok(()),
            Ok(Some(failure)) => {
                Err(AppError { summary: failure.summary, traceback: failure.traceback })
            }
            Err(error) => Err(AppError { summary: error.to_string(), traceback: String::new() }),
        }
    }
}

#[derive(Default)]
pub struct Endpoints {
    websockets: Arc<Registry>,
}

impl EndpointService for Endpoints {}

impl EndpointServer for Endpoints {
    fn new() -> Self {
        Self::default()
    }

    async fn serve(
        &self,
        stream: Arc<dyn ExchangeStream>,
        app: Arc<dyn AsgiApplication>,
        request: EndpointRequest,
    ) {
        let app = Application { app, request: request.clone() };
        endpoint::serve(Arc::new(Exchange(stream)), request, &app, &self.websockets).await;
    }

    async fn close_websockets(&self, code: u16, reason: String) {
        self.websockets.close_all(code, &reason).await;
    }
}

/// One exchange's `receive()`.
pub struct ReceiveCall(Conn);

impl AsgiReceiver for ReceiveCall {}

impl AsgiReceive for ReceiveCall {
    async fn __call__(&self) -> AsgiReceived {
        self.0.receive().await
    }
}

/// One exchange's `send()`.
pub struct SendCall(Conn);

impl AsgiSender for SendCall {}

impl AsgiSend for SendCall {
    async fn __call__(&self, message: AsgiSent) -> Result<(), WorkerError> {
        self.0.send(message).await.map_err(WorkerError::Failed)
    }
}
