use std::collections::BTreeMap;
use std::time::Duration;

use livekit_api::services::connector::ConnectTwilioCallOptions;
use livekit_api::services::sip::CreateSIPParticipantOptions;
use livekit_api::services::{LiveKitApi, ServerError, ServerErrorCode, ServiceError, TokenGrants};
use livekit_protocol as proto;
use lk_agents_abi::errors::{ApiError, ErrorKind};
use lk_agents_abi::telemetry::record::JobKey;

use crate::job_token;

/// A project's RoomService at the server's `LIVEKIT_URL` (`wss://` or `https://`).
pub struct RoomService(LiveKitApi);

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

/// The project's API at the server's `LIVEKIT_URL`. With the server's key pair, each call is
/// signed with it. Without one (token mode), each call uses a token of `job` with the grants of
/// that call, which the server issues.
fn server_api(env: &BTreeMap<String, String>, job: JobKey) -> LiveKitApi {
    let get = |name: &str| env.get(name).map_or("", String::as_str);
    let (url, key, secret) =
        (get("LIVEKIT_URL"), get("LIVEKIT_API_KEY"), get("LIVEKIT_API_SECRET"));
    if !key.is_empty() || !secret.is_empty() {
        return LiveKitApi::with_api_key(url, key, secret);
    }
    LiveKitApi::with_token_provider(url, move |grants: TokenGrants| {
        let job = job.clone();
        async move { Ok(job_token::token(&job, &grants.to_claims_json()?).await?) }
    })
}

impl RoomService {
    pub fn new(url: &str, api_key: &str, api_secret: &str) -> Self {
        Self(LiveKitApi::with_api_key(url, api_key, api_secret))
    }

    /// The server's RoomService for `job`: see [`server_api`].
    pub fn of_server(env: &BTreeMap<String, String>, job: JobKey) -> Self {
        Self(server_api(env, job))
    }

    /// `RemoveParticipant`: `identity` leaves `room`. A participant already gone is
    /// [`Removal::NotInRoom`], not an error.
    pub async fn remove_participant(
        &self,
        room: &str,
        identity: &str,
    ) -> Result<Removal, ServiceError> {
        match self.0.room().remove_participant(room, identity).await {
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
        self.0.room().move_participant(room, identity, destination).await
    }

    /// `DeleteRoom`: everyone in `room` is disconnected. A room already gone is not an error.
    pub async fn delete_room(&self, room: &str) -> Result<(), ServiceError> {
        match self.0.room().delete_room(room).await {
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
pub struct Telephony(LiveKitApi);

impl Telephony {
    pub fn new(url: &str, api_key: &str, api_secret: &str) -> Self {
        Self(LiveKitApi::with_api_key(url, api_key, api_secret))
    }

    /// The server's telephony for `job`: see [`server_api`].
    pub fn of_server(env: &BTreeMap<String, String>, job: JobKey) -> Self {
        Self(server_api(env, job))
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
        self.0
            .sip()
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
        let response = self.0.connector().connect_twilio_call(direction, room, options).await?;
        Ok(response.connect_url)
    }
}

/// A project request's failure: an unsent request is a connection failure.
pub fn api_error(error: ServiceError) -> ApiError {
    match error {
        ServiceError::Twirp(ServerError::Request(error)) => {
            ApiError::connection(Some(crate::http::error_chain(&error)))
        }
        error => ApiError::new(ErrorKind::Server, error.to_string()),
    }
}

/// A form POST that does not follow redirects: the status and body of the first response.
pub async fn post_form(
    url: &str,
    headers: &BTreeMap<String, String>,
    form: &[(String, String)],
) -> Result<(u16, String), ApiError> {
    let unbuilt =
        |error: crate::tls::TlsError| ApiError::new(ErrorKind::InvalidRequest, error.to_string());
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

    // Two jobs run in token mode, and the first calls the server API. Catches a call that signs
    // with no key instead of using a job token with the call's grants, a call that uses another
    // job's token, a call outside a running job, and a job token asked for while a key pair is
    // set.
    #[tokio::test]
    async fn without_a_key_pair_each_call_uses_a_token_of_its_own_job() {
        use crate::job_token::{self, Fetch, JobToken, JobTokens};
        use lk_agents_abi::telemetry::record::JobKey;
        use parking_lot::Mutex;
        use std::sync::Arc;
        let recorder = |asked: Arc<Mutex<Vec<String>>>| -> Fetch {
            Box::new(move |claims| {
                asked.lock().push(claims);
                let token = JobToken { token: "job-token".into(), expires_at: i64::MAX / 2 };
                Box::pin(std::future::ready(Ok(token)))
            })
        };
        let (asked, other) = (Arc::default(), Arc::default());
        let clock = Arc::new(crate::clock::RealClock::new());
        let (job, newer) = (JobKey("AJ_room_1".into()), JobKey("AJ_room_2".into()));
        job_token::open(
            job.clone(),
            Arc::new(JobTokens::new(recorder(Arc::clone(&asked)), clock.clone())),
        );
        job_token::open(
            newer.clone(),
            Arc::new(JobTokens::new(recorder(Arc::clone(&other)), clock)),
        );

        let (url, served) = answer_once("200 OK", "").await;
        let env = BTreeMap::from([("LIVEKIT_URL".to_owned(), url)]);
        RoomService::of_server(&env, job.clone()).delete_room("room").await.unwrap();
        assert_eq!(served.await.unwrap(), "/twirp/livekit.RoomService/DeleteRoom");
        let ended = RoomService::of_server(&env, JobKey("AJ_ended".into()));
        assert!(ended.delete_room("room").await.is_err(), "no running job, no token");

        let (url, served) = answer_once("200 OK", "").await;
        let keyed = BTreeMap::from([
            ("LIVEKIT_URL".to_owned(), url),
            ("LIVEKIT_API_KEY".to_owned(), "key".to_owned()),
            ("LIVEKIT_API_SECRET".to_owned(), "a-secret-long-enough-for-signing-000000".to_owned()),
        ]);
        RoomService::of_server(&keyed, job.clone()).delete_room("room").await.unwrap();
        served.await.unwrap();
        job_token::close(&job);
        job_token::close(&newer);
        assert_eq!(
            *asked.lock(),
            [r#"{"video":{"roomCreate":true}}"#],
            "one job token, its grants"
        );
        assert!(other.lock().is_empty(), "another job's token was asked for");
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
