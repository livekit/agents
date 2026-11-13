use std::collections::BTreeMap;
use std::time::Duration;

use livekit_api::services::connector::{ConnectTwilioCallOptions, ConnectorClient};
use livekit_api::services::room::RoomClient;
use livekit_api::services::sip::{CreateSIPParticipantOptions, SIPClient};
use livekit_api::services::{ServerError, ServerErrorCode, ServiceError};
use livekit_protocol as proto;
use lk_agents_abi::errors::ApiError;

/// A project's RoomService: its `LIVEKIT_URL` (`wss://` or `https://`) and key pair.
pub struct RoomService(RoomClient);

/// What `remove_participant` found.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Removal {
    Removed,
    /// The participant was not in the room (`not_found`).
    NotInRoom,
}

fn not_found(error: &ServiceError) -> bool {
    matches!(error, ServiceError::Twirp(ServerError::Twirp(code))
        if code.code == ServerErrorCode::NOT_FOUND)
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
            Err(error) if not_found(&error) => Ok(Removal::NotInRoom),
            Err(error) => Err(error),
        }
    }

    /// `MoveParticipant`: `identity` leaves `room` for `destination`.
    pub async fn move_participant(
        &self,
        room: &str,
        identity: &str,
        destination: &str,
    ) -> Result<(), ServiceError> {
        self.0.move_participant(room, identity, destination).await
    }

    /// `DeleteRoom`: everyone in `room` is disconnected. A room already gone is not an error.
    pub async fn delete_room(&self, room: &str) -> Result<(), ServiceError> {
        match self.0.delete_room(room).await {
            Err(error) if not_found(&error) => Ok(()),
            other => other,
        }
    }
}

/// One outbound SIP call into a room.
#[derive(Debug, Clone, Default)]
pub struct SipCall {
    pub sip_trunk_id: Option<String>,
    pub sip_call_to: String,
    pub room_name: String,
    pub participant_identity: String,
    pub wait_until_answered: bool,
    pub sip_number: Option<String>,
    pub headers: BTreeMap<String, String>,
    pub dtmf: String,
    pub ringing_timeout: Option<Duration>,
    /// The trunk to dial through when no `sip_trunk_id` names one.
    pub trunk: Option<proto::SipOutboundConfig>,
}

/// A project's telephony: its SIP service and its Twilio connector.
pub struct Telephony {
    sip: SIPClient,
    connector: ConnectorClient,
}

impl Telephony {
    pub fn new(url: &str, api_key: &str, api_secret: &str) -> Self {
        Self {
            sip: SIPClient::with_api_key(url, api_key, api_secret),
            connector: ConnectorClient::with_api_key(url, api_key, api_secret),
        }
    }

    /// `CreateSIPParticipant`: dials `call`, returning once it is answered when
    /// `wait_until_answered`.
    pub async fn create_sip_participant(&self, call: SipCall) -> Result<(), ServiceError> {
        let options = CreateSIPParticipantOptions {
            participant_identity: call.participant_identity,
            sip_number: call.sip_number,
            dtmf: Some(call.dtmf).filter(|dtmf| !dtmf.is_empty()),
            wait_until_answered: Some(call.wait_until_answered),
            ringing_timeout: call.ringing_timeout,
            headers: Some(call.headers.into_iter().collect()),
            ..Default::default()
        };
        let trunk = call.sip_trunk_id.unwrap_or_default();
        self.sip
            .create_sip_participant(trunk, call.sip_call_to, call.room_name, options, call.trunk)
            .await
            .map(drop)
    }

    /// `ConnectTwilioCall` (outbound): the WebSocket URL Twilio's media stream connects to, as
    /// `identity` in `room`.
    pub async fn connect_twilio_call(
        &self,
        room: &str,
        identity: &str,
    ) -> Result<String, ServiceError> {
        let options = ConnectTwilioCallOptions {
            participant_identity: Some(identity.to_owned()),
            ..Default::default()
        };
        let direction = proto::connect_twilio_call_request::TwilioCallDirection::Outbound;
        let response = self.connector.connect_twilio_call(direction, room, options).await?;
        Ok(response.connect_url)
    }
}

/// A project request's failure: an unsent request is a connection failure.
pub fn api_error(error: ServiceError) -> ApiError {
    match error {
        ServiceError::Twirp(ServerError::Request(error)) => {
            ApiError::connection(Some(crate::http::error_chain(&error)))
        }
        error => ApiError::new(error.to_string()),
    }
}

/// A form POST that does not follow redirects: the status and body of the first response.
pub async fn post_form(
    url: &str,
    headers: &BTreeMap<String, String>,
    form: &[(String, String)],
) -> Result<(u16, String), ApiError> {
    let unbuilt =
        |error: crate::tls::TlsError| ApiError::new(error.to_string()).with_retryable(false);
    let builder = crate::tls::http_client_builder().map_err(unbuilt)?;
    let client =
        crate::tls::build(builder.redirect(reqwest::redirect::Policy::none())).map_err(unbuilt)?;
    let body = url::form_urlencoded::Serializer::new(String::new()).extend_pairs(form).finish();
    let mut request = client
        .post(url)
        .header(reqwest::header::CONTENT_TYPE, "application/x-www-form-urlencoded")
        .body(body);
    for (name, value) in headers {
        request = request.header(name, value);
    }
    let failed =
        |error: reqwest::Error| ApiError::connection(Some(crate::http::error_chain(&error)));
    let response = request.send().await.map_err(failed)?;
    let status = response.status().as_u16();
    let text = response.text().await.map_err(failed)?;
    Ok((status, text))
}

#[cfg(test)]
mod tests {
    use super::{Removal, RoomService, post_form};
    use std::collections::BTreeMap;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    /// A server answering one request with `status` and the JSON `body`; the request's path.
    async fn answer_once(
        status: &str,
        body: &'static str,
    ) -> (String, impl std::future::Future<Output = Result<String, tokio::task::JoinError>>) {
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

    // Catches: a room already gone failing the caller's cleanup.
    #[tokio::test]
    async fn deleting_a_room_already_gone_is_not_an_error() {
        let (url, served) =
            answer_once("404 Not Found", r#"{"code":"not_found","msg":"room not found"}"#).await;
        let service = RoomService::new(&url, "key", "a-secret-long-enough-for-signing-000000");
        service.delete_room("room").await.unwrap();
        assert_eq!(served.await.unwrap(), "/twirp/livekit.RoomService/DeleteRoom");
    }

    #[tokio::test]
    async fn a_participant_is_moved() {
        let (url, served) = answer_once("200 OK", "").await;
        let service = RoomService::new(&url, "key", "a-secret-long-enough-for-signing-000000");
        service.move_participant("from", "agent", "to").await.unwrap();
        assert_eq!(served.await.unwrap(), "/twirp/livekit.RoomService/MoveParticipant");
    }

    // Catches: a redirect followed (Twilio's answer is the first response), or the form or its
    // headers not sent as given.
    #[tokio::test]
    async fn a_form_post_returns_the_first_response_without_following_a_redirect() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}/Calls.json", listener.local_addr().unwrap());
        let location = url.replace("Calls", "Elsewhere");
        let served = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut request = Vec::new();
            let mut buf = [0u8; 4096];
            while !String::from_utf8_lossy(&request).ends_with("To=%2B1555&Twiml=%3CResponse%2F%3E")
            {
                let n = socket.read(&mut buf).await.unwrap();
                request.extend_from_slice(&buf[..n]);
            }
            let body = r#"{"code":20003}"#;
            let response = format!(
                "HTTP/1.1 302 Found\r\nlocation: {location}\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                body.len()
            );
            socket.write_all(response.as_bytes()).await.unwrap();
            String::from_utf8_lossy(&request).to_lowercase()
        });
        let headers = BTreeMap::from([("Authorization".to_owned(), "Basic abc".to_owned())]);
        let form =
            [("To".to_owned(), "+1555".to_owned()), ("Twiml".to_owned(), "<Response/>".to_owned())];
        let (status, body) = post_form(&url, &headers, &form).await.unwrap();
        assert_eq!((status, body.as_str()), (302, r#"{"code":20003}"#));
        let request = served.await.unwrap();
        assert!(request.starts_with("post /calls.json"), "{request}");
        assert!(request.contains("authorization: basic abc"), "{request}");
        assert!(request.contains("content-type: application/x-www-form-urlencoded"), "{request}");
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
