//! gRPC unary calls under the project token: the management `AgentDB` and the `AgentThreads`
//! services of cloud-protocol, in-cluster or beside a local agent-db. tonic's channel with a codec
//! over the workspace's prost, so the messages are `livekit-protocol`'s and nothing is generated
//! here; a service is its path and its message types.

use std::marker::PhantomData;
use std::sync::Arc;

use livekit_protocol::agent_db;
use lk_agents_host_derive::Host;
use prost::Message;
use serde::{Deserialize, Serialize};
use tonic::codec::{Codec, DecodeBuf, Decoder, EncodeBuf, Encoder};
use tonic::transport::{Channel, ClientTlsConfig, Endpoint};
use tonic::{Code, Request, Status};

use crate::StorageError;
use crate::threads::{duration_of, seconds_of};
use crate::token::TokenMinter;

/// One service's channel. Lazy: the channel is made on the first call, on the runtime that
/// call runs on (a `Storage` is built from Python with no runtime entered), and kept.
#[derive(Clone)]
pub(crate) struct Client {
    endpoint: Endpoint,
    channel: Arc<tokio::sync::OnceCell<Channel>>,
    minter: TokenMinter,
}

impl Client {
    pub fn new(url: &str, minter: TokenMinter) -> Result<Self, StorageError> {
        let mut endpoint = Endpoint::from_shared(url.to_owned())
            .map_err(|e| StorageError::Invalid(format!("grpc url {url}: {e}")))?;
        if url.starts_with("https://") {
            endpoint = endpoint
                .tls_config(ClientTlsConfig::new().with_webpki_roots())
                .map_err(|e| StorageError::Invalid(format!("grpc tls {url}: {e}")))?;
        }
        Ok(Self { endpoint, channel: Arc::new(tokio::sync::OnceCell::new()), minter })
    }

    pub async fn unary<Req: Message + 'static, Resp: Message + Default + 'static>(
        &self,
        path: &'static str,
        req: Req,
    ) -> Result<Resp, StorageError> {
        let mut request = Request::new(req);
        let bearer = format!("Bearer {}", self.minter.mint()?)
            .parse()
            .map_err(|_| StorageError::Invalid("token is not a header value".into()))?;
        request.metadata_mut().insert("authorization", bearer);

        let channel = self.channel.get_or_init(|| async { self.endpoint.connect_lazy() }).await;
        let mut grpc = tonic::client::Grpc::new(channel.clone());
        grpc.ready().await.map_err(|e| StorageError::Unavailable(e.to_string()))?;
        let path = http::uri::PathAndQuery::from_static(path);
        let response =
            grpc.unary(request, path, ProstCodec::default()).await.map_err(from_status)?;
        Ok(response.into_inner())
    }
}

/// A gRPC status as the storage error the Twirp client used to raise, so callers keep their
/// matches: the code's snake_case name for anything not named above.
fn from_status(status: Status) -> StorageError {
    let message = status.message().to_owned();
    match status.code() {
        Code::NotFound => StorageError::NotFound(message),
        Code::Unavailable | Code::DeadlineExceeded => StorageError::Unavailable(message),
        Code::Unauthenticated | Code::PermissionDenied => StorageError::Unauthorized(message),
        code => StorageError::Server { code: code_name(code).to_owned(), message },
    }
}

fn code_name(code: Code) -> &'static str {
    match code {
        Code::Ok => "ok",
        Code::Cancelled => "canceled",
        Code::Unknown => "unknown",
        Code::InvalidArgument => "invalid_argument",
        Code::DeadlineExceeded => "deadline_exceeded",
        Code::NotFound => "not_found",
        Code::AlreadyExists => "already_exists",
        Code::PermissionDenied => "permission_denied",
        Code::ResourceExhausted => "resource_exhausted",
        Code::FailedPrecondition => "failed_precondition",
        Code::Aborted => "aborted",
        Code::OutOfRange => "out_of_range",
        Code::Unimplemented => "unimplemented",
        Code::Internal => "internal",
        Code::Unavailable => "unavailable",
        Code::DataLoss => "data_loss",
        Code::Unauthenticated => "unauthenticated",
    }
}

/// prost over tonic's buffers: the two crates share `bytes`, so a message encodes straight
/// into the frame.
struct ProstCodec<Req, Resp>(PhantomData<(Req, Resp)>);

impl<Req, Resp> Default for ProstCodec<Req, Resp> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<Req: Message + 'static, Resp: Message + Default + 'static> Codec for ProstCodec<Req, Resp> {
    type Encode = Req;
    type Decode = Resp;
    type Encoder = ProstEncoder<Req>;
    type Decoder = ProstDecoder<Resp>;

    fn encoder(&mut self) -> Self::Encoder {
        ProstEncoder(PhantomData)
    }

    fn decoder(&mut self) -> Self::Decoder {
        ProstDecoder(PhantomData)
    }
}

struct ProstEncoder<T>(PhantomData<T>);

impl<T: Message> Encoder for ProstEncoder<T> {
    type Item = T;
    type Error = Status;

    fn encode(&mut self, item: T, dst: &mut EncodeBuf<'_>) -> Result<(), Status> {
        item.encode(dst).map_err(|e| Status::internal(e.to_string()))
    }
}

struct ProstDecoder<T>(PhantomData<T>);

impl<T: Message + Default> Decoder for ProstDecoder<T> {
    type Item = T;
    type Error = Status;

    fn decode(&mut self, src: &mut DecodeBuf<'_>) -> Result<Option<T>, Status> {
        T::decode(src).map(Some).map_err(|e| Status::internal(e.to_string()))
    }
}

/// `AgentDB`: create, get and delete a database (the lifecycle API; queries run over the data
/// plane). A thread's database is normally made by `AgentThreads.CreateThread`; this is
/// for local development and tests.
#[derive(Clone)]
pub struct AgentDbAdmin {
    client: Client,
}

/// A created database: `AgentDB.CreateDatabase`'s answer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Host)]
#[python(pydantic, module = "storage", config = "frozen=True")]
pub struct DatabaseInfo {
    /// `DB_..`.
    pub database_id: String,
    /// Unix seconds; `0` for never.
    pub expires_at: i64,
}

impl AgentDbAdmin {
    pub fn new(url: &str, minter: TokenMinter) -> Result<Self, StorageError> {
        Ok(Self { client: Client::new(url, minter)? })
    }

    /// A database in the project's data region, `ttl_seconds` 0 for never.
    pub async fn create_database(&self, ttl_seconds: i64) -> Result<DatabaseInfo, StorageError> {
        let r: agent_db::CreateResponse = self
            .client
            .unary(
                "/cloud_protocol.AgentDB/CreateDatabase",
                agent_db::CreateRequest { ttl: duration_of(ttl_seconds) },
            )
            .await?;
        Ok(DatabaseInfo { database_id: r.database_id, expires_at: seconds_of(r.expires_at) })
    }

    pub async fn get_database(
        &self,
        database_id: &str,
    ) -> Result<agent_db::AgentDatabase, StorageError> {
        self.client
            .unary(
                "/cloud_protocol.AgentDB/GetDatabase",
                agent_db::GetRequest { database_id: database_id.to_owned() },
            )
            .await
    }

    pub async fn delete_database(&self, database_id: &str) -> Result<(), StorageError> {
        self.client
            .unary::<_, agent_db::DeleteResponse>(
                "/cloud_protocol.AgentDB/DeleteDatabase",
                agent_db::DeleteRequest { database_id: database_id.to_owned() },
            )
            .await
            .map(|_| ())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_status_keeps_its_code_name_and_the_named_ones_map() {
        let e = from_status(Status::already_exists("key taken"));
        assert!(
            matches!(&e, StorageError::Server { code, message } if code == "already_exists" && message == "key taken")
        );
        assert!(matches!(from_status(Status::not_found("x")), StorageError::NotFound(_)));
        assert!(matches!(from_status(Status::unavailable("x")), StorageError::Unavailable(_)));
        assert!(matches!(
            from_status(Status::permission_denied("x")),
            StorageError::Unauthorized(_)
        ));
        assert!(Client::new("not a url", TokenMinter::new("k", "s").unwrap()).is_err());
    }
}
