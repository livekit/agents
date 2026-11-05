//! LiveKit's RoomService, for what the framework does in a job's room on its own behalf.

use livekit_api::services::room::RoomClient;
use livekit_api::services::{ServerError, ServerErrorCode, ServiceError};

/// A project's RoomService: its `LIVEKIT_URL` (`wss://` or `https://`) and key pair.
pub struct RoomService(RoomClient);

/// What `remove_participant` found.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Removal {
    Removed,
    /// The participant was not in the room (`not_found`).
    NotInRoom,
}

impl RoomService {
    pub fn new(url: &str, api_key: &str, api_secret: &str) -> Self {
        Self(RoomClient::with_api_key(url, api_key, api_secret))
    }

    /// `RemoveParticipant`: `identity` leaves `room`. A participant already gone is
    /// [`Removal::NotInRoom`], not an error.
    pub async fn remove_participant(
        &self,
        room: &str,
        identity: &str,
    ) -> Result<Removal, ServiceError> {
        match self.0.remove_participant(room, identity).await {
            Ok(()) => Ok(Removal::Removed),
            Err(ServiceError::Twirp(ServerError::Twirp(code)))
                if code.code == ServerErrorCode::NOT_FOUND =>
            {
                Ok(Removal::NotInRoom)
            }
            Err(error) => Err(error),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Removal, RoomService};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    /// A server answering one request with `status` and the JSON `body`; the request's path.
    async fn answer_once(
        status: &str,
        body: &'static str,
    ) -> (String, tokio::task::JoinHandle<String>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let status = status.to_owned();
        let served = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut request = Vec::new();
            let mut buf = [0u8; 4096];
            // The headers, then a body of at least one byte (the protobuf request).
            while !request.windows(4).any(|w| w == b"\r\n\r\n") {
                let n = socket.read(&mut buf).await.unwrap();
                request.extend_from_slice(&buf[..n]);
            }
            let response = format!(
                "HTTP/1.1 {status}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                body.len()
            );
            socket.write_all(response.as_bytes()).await.unwrap();
            let head = String::from_utf8_lossy(&request).into_owned();
            head.split_whitespace().nth(1).unwrap_or_default().to_owned()
        });
        (url, served)
    }

    #[tokio::test]
    async fn a_participant_in_the_room_is_removed() {
        let (url, _served) = answer_once("200 OK", "").await;
        let service = RoomService::new(&url, "key", "a-secret-long-enough-for-signing-000000");
        assert_eq!(service.remove_participant("room", "avatar").await.unwrap(), Removal::Removed);
    }

    #[tokio::test]
    async fn a_participant_already_gone_is_not_an_error() {
        let (url, served) =
            answer_once("404 Not Found", r#"{"code":"not_found","msg":"participant not found"}"#)
                .await;
        let service = RoomService::new(&url, "key", "a-secret-long-enough-for-signing-000000");
        let removal = service.remove_participant("room", "avatar").await.unwrap();
        assert_eq!(removal, Removal::NotInRoom);
        assert_eq!(served.await.unwrap(), "/twirp/livekit.RoomService/RemoveParticipant");
    }

    #[tokio::test]
    async fn any_other_failure_is_returned() {
        let (url, _served) =
            answer_once("403 Forbidden", r#"{"code":"permission_denied","msg":"no"}"#).await;
        let service = RoomService::new(&url, "key", "a-secret-long-enough-for-signing-000000");
        let error = service.remove_participant("room", "avatar").await.unwrap_err();
        assert!(error.to_string().contains("permission_denied"), "{error}");
    }
}
