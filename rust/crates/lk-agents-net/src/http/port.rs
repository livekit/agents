//! HTTP socket boundary. Request deadlines, retries and protocol decoding remain in executors.

use std::pin::Pin;

use bytes::Bytes;
use futures_util::Stream;
use lk_agents_abi::BoxFuture;
use lk_agents_core::request::AttemptFailure;

use super::exchange::ResponsePolicy;
use super::{GatewayHttp, Reporter, Sent};
use crate::transport::HttpRequest;

pub type ResponseBody = Pin<Box<dyn Stream<Item = Result<Bytes, AttemptFailure>> + Send>>;

pub struct HttpResponse {
    pub status: reqwest::StatusCode,
    pub headers: reqwest::header::HeaderMap,
    pub body: ResponseBody,
}

pub trait HttpExchangePort: Send + Sync {
    fn send<'a>(
        &'a self,
        method: reqwest::Method,
        request: HttpRequest,
        policy: ResponsePolicy,
        reporter: Reporter<(u32, Sent)>,
        attempt: u32,
    ) -> BoxFuture<'a, Result<HttpResponse, AttemptFailure>>;
}

impl HttpExchangePort for GatewayHttp {
    fn send<'a>(
        &'a self,
        method: reqwest::Method,
        request: HttpRequest,
        policy: ResponsePolicy,
        reporter: Reporter<(u32, Sent)>,
        attempt: u32,
    ) -> BoxFuture<'a, Result<HttpResponse, AttemptFailure>> {
        Box::pin(async move {
            use futures_util::StreamExt;
            let sent = |stage| (attempt, stage);
            let response = match policy {
                ResponsePolicy::Successful => {
                    self.send_method(method, request, &reporter, sent).await?
                }
                ResponsePolicy::Protocol => self.send_raw(method, request, &reporter, sent).await?,
            };
            let status = response.status();
            let headers = response.headers().clone();
            let body = Box::pin(response.bytes_stream().map(|chunk| {
                chunk.map_err(|error| AttemptFailure::Connection(super::error_chain(&error)))
            }));
            Ok(HttpResponse { status, headers, body })
        })
    }
}
