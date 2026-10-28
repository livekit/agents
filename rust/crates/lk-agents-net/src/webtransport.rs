//! Raw WebTransport sessions, with shared TLS setup and HTTP connect headers.

use web_transport_quinn::ClientBuilder;
use web_transport_quinn::proto::ConnectRequest;

use crate::transport::TransportError;

pub use web_transport_quinn::{RecvStream, SendStream, Session};

/// Open a session; the caller owns the connection deadline, stream protocol and session lifetime.
/// `insecure` is reserved for local servers with a self-signed certificate.
pub async fn connect(
    url: &str,
    headers: Vec<(String, String)>,
    insecure: bool,
) -> Result<Session, TransportError> {
    let url = url
        .parse::<url::Url>()
        .map_err(|error| TransportError::Connect(format!("bad WebTransport url: {error}")))?;
    let mut request = ConnectRequest::new(url);
    for (name, value) in headers {
        let header_name: web_transport_quinn::http::HeaderName = name
            .parse()
            .map_err(|_| TransportError::Connect(format!("bad header name: {name}")))?;
        let header_value = value
            .parse()
            .map_err(|_| TransportError::Connect(format!("bad header value for {name}")))?;
        request = request.with_header(header_name, header_value);
    }
    let client = if insecure {
        ClientBuilder::new().dangerous().with_no_certificate_verification()
    } else {
        ClientBuilder::new().with_system_roots()
    }
    .map_err(|error| TransportError::Connect(error.to_string()))?;
    client.connect(request).await.map_err(|error| TransportError::Connect(error.to_string()))
}
