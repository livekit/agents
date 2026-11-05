//! Real TLS handshakes through the framework's HTTP, storage and worker transports.

use std::path::Path;
use std::process::Command;
use std::sync::Arc;
use std::time::Duration;

use base64::Engine as _;
use futures_util::{SinkExt, StreamExt};
use lk_agents_abi::storage::Storage as _;
use lk_agents_net::{env::ProcessEnv, tls, websocket};
use lk_agents_storage::namespace::Namespace;
use lk_agents_worker::transport::{WorkerTransport, WtTransport};
use rustls::pki_types::{CertificateDer, PrivateKeyDer};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio_rustls::TlsAcceptor;

#[test]
fn every_transport_uses_the_configured_ca_and_reuses_the_trust_configuration() {
    let ca_key = rcgen::KeyPair::generate().unwrap();
    let mut ca_params = rcgen::CertificateParams::default();
    ca_params.is_ca = rcgen::IsCa::Ca(rcgen::BasicConstraints::Unconstrained);
    let ca = ca_params.self_signed(&ca_key).unwrap();
    let issuer = rcgen::Issuer::from_params(&ca_params, &ca_key);
    let directory = std::env::temp_dir().join(lk_agents_net::strings::shortuuid("tls-"));
    let roots = directory.join("roots");
    std::fs::create_dir_all(&roots).unwrap();
    let der = base64::engine::general_purpose::STANDARD.encode(ca.der());
    let pem = format!("-----BEGIN CERTIFICATE-----\n{der}\n-----END CERTIFICATE-----\n");
    std::fs::write(roots.join("ca.pem"), pem).unwrap();
    for (name, hostname, expired) in [
        ("cert", "localhost", false),
        ("wrong-host", "other.example", false),
        ("expired", "localhost", true),
    ] {
        let mut params = rcgen::CertificateParams::new(vec![hostname.to_owned()]).unwrap();
        if expired {
            params.not_after = rcgen::date_time_ymd(2000, 1, 1);
        }
        let key = rcgen::KeyPair::generate().unwrap();
        let cert = params.signed_by(&key, &issuer).unwrap();
        std::fs::write(directory.join(format!("{name}.der")), cert.der()).unwrap();
        std::fs::write(directory.join(format!("{name}.key")), key.serialize_der()).unwrap();
    }
    std::fs::write(directory.join("empty.pem"), "").unwrap();
    std::fs::write(directory.join("malformed.pem"), "invalid certificate data").unwrap();
    for mode in ["file", "directory", "empty", "malformed", "missing"] {
        let mut command = Command::new(std::env::current_exe().unwrap());
        command
            .args(["--ignored", "--exact", &test_name("trust_probe"), "--nocapture"])
            .env("TLS_TEST_DIRECTORY", &directory)
            .env("TLS_TEST_MODE", mode)
            .env_remove("SSL_CERT_FILE")
            .env_remove("SSL_CERT_DIR");
        match mode {
            "file" => command.env("SSL_CERT_FILE", roots.join("ca.pem")),
            "directory" => command.env("SSL_CERT_DIR", &roots),
            _ => command.env("SSL_CERT_FILE", directory.join(format!("{mode}.pem"))),
        };
        let result = command.output().unwrap();
        assert!(
            result.status.success(),
            "{mode}: {}{}",
            String::from_utf8_lossy(&result.stdout),
            String::from_utf8_lossy(&result.stderr)
        );
    }
    std::fs::remove_dir_all(directory).unwrap();
}

#[tokio::test]
#[ignore = "subprocess probe with isolated CA environment"]
async fn trust_probe() {
    tokio::time::timeout(Duration::from_secs(15), probe()).await.unwrap();
}

async fn probe() {
    if matches!(std::env::var("TLS_TEST_MODE").unwrap().as_str(), "empty" | "malformed" | "missing")
    {
        assert!(tls::config().await.unwrap_err().contains("no usable TLS root certificates"));
        assert!(tls::http_client_builder().is_err(), "must not fall back to bundled roots");
        return;
    }
    let directory = std::env::var("TLS_TEST_DIRECTORY").unwrap();
    let directory = Path::new(&directory);
    let cert = CertificateDer::from(std::fs::read(directory.join("cert.der")).unwrap());
    let key = std::fs::read(directory.join("cert.key")).unwrap();
    let shared = tls::config().await.unwrap();

    let (url, server) = http_server(cert.clone(), &key).await;
    let client = tls::http_client_builder().unwrap().build().unwrap();
    assert!(client.get(&url).send().await.unwrap().status().is_success());
    server.await.unwrap();

    let (url, server) = http_server(cert.clone(), &key).await;
    let storage = Namespace::new(
        &ProcessEnv::current_runtime(),
        None,
        Some(url),
        Some("key".into()),
        Some("secret".into()),
        None,
        None,
    )
    .unwrap();
    assert!(storage.list_threads("scope".into(), None, None).await.unwrap().threads.is_empty());
    server.await.unwrap();
    storage.aclose().await;

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("wss://localhost:{}/", listener.local_addr().unwrap().port());
    let acceptor = tls_acceptor(cert.clone(), &key);
    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let stream = acceptor.accept(stream).await.unwrap();
        let mut socket = tokio_tungstenite::accept_async(stream).await.unwrap();
        socket.send(websocket::Message::Text("ready".into())).await.unwrap();
        socket.close(None).await.unwrap();
    });
    let mut socket = websocket::connect_with_limit(&url, vec![], None).await.unwrap();
    assert_eq!(socket.next().await.unwrap().unwrap(), websocket::Message::Text("ready".into()));
    drop(socket);
    server.await.unwrap();

    let (url, mut server) = quic_server(cert, &key);
    let peer = tokio::spawn(async move {
        let session = server.accept().await.unwrap().ok().await.unwrap();
        let (_send, mut recv) = session.accept_bi().await.unwrap();
        let mut header = [0; 4];
        recv.read(&mut header).await.unwrap();
    });
    let connected = WtTransport { insecure: false }.connect(&url, None, vec![]).await.unwrap();
    connected.sink.send(livekit_protocol::WorkerMessage::default()).await.unwrap();
    peer.await.unwrap();
    connected.sink.close().await;

    let untrusted = rcgen::generate_simple_self_signed(vec!["localhost".into()]).unwrap();
    reject_certificate(untrusted.cert.der().clone(), &untrusted.signing_key.serialize_der()).await;
    for name in ["wrong-host", "expired"] {
        let cert =
            CertificateDer::from(std::fs::read(directory.join(format!("{name}.der"))).unwrap());
        let key = std::fs::read(directory.join(format!("{name}.key"))).unwrap();
        reject_certificate(cert, &key).await;
    }
    assert!(Arc::ptr_eq(&shared, &tls::config().await.unwrap()));
}

async fn reject_certificate(cert: CertificateDer<'static>, key: &[u8]) {
    for transport in ["http", "storage", "websocket"] {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let scheme = if transport == "websocket" { "wss" } else { "https" };
        let url = format!("{scheme}://localhost:{}", listener.local_addr().unwrap().port());
        let acceptor = tls_acceptor(cert.clone(), key);
        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            assert!(acceptor.accept(stream).await.is_err());
        });
        match transport {
            "http" => {
                let client = tls::http_client_builder().unwrap().build().unwrap();
                assert!(client.get(url).send().await.is_err());
            }
            "storage" => {
                let storage = Namespace::new(
                    &ProcessEnv::current_runtime(),
                    None,
                    Some(url),
                    Some("key".into()),
                    Some("secret".into()),
                    None,
                    None,
                )
                .unwrap();
                assert!(storage.list_threads("scope".into(), None, None).await.is_err());
                storage.aclose().await;
            }
            _ => assert!(websocket::connect_with_limit(&url, vec![], None).await.is_err()),
        }
        server.await.unwrap();
    }
    let (url, mut server) = quic_server(cert, key);
    let peer = tokio::spawn(async move {
        // The server's accept loop silently discards failed TLS handshakes.
        if let Some(request) = server.accept().await {
            let _ = request.ok().await;
        }
    });
    let error = WtTransport { insecure: false }.connect(&url, None, vec![]).await.err().unwrap();
    assert!(error.to_string().contains("certificate"), "{error}");
    peer.abort();
    let _ = peer.await;
}

fn quic_server(cert: CertificateDer<'static>, key: &[u8]) -> (String, web_transport_quinn::Server) {
    let key = PrivateKeyDer::Pkcs8(key.to_vec().into());
    let mut config = rustls::ServerConfig::builder_with_provider(Arc::new(
        rustls::crypto::ring::default_provider(),
    ))
    .with_safe_default_protocol_versions()
    .unwrap()
    .with_no_client_auth()
    .with_single_cert(vec![cert], key)
    .unwrap();
    config.alpn_protocols = vec![web_transport_quinn::ALPN.as_bytes().to_vec()];
    let config =
        web_transport_quinn::quinn::crypto::rustls::QuicServerConfig::try_from(config).unwrap();
    let config = web_transport_quinn::quinn::ServerConfig::with_crypto(Arc::new(config));
    let endpoint =
        web_transport_quinn::quinn::Endpoint::server(config, "[::]:0".parse().unwrap()).unwrap();
    let url = format!("https://localhost:{}", endpoint.local_addr().unwrap().port());
    let server = web_transport_quinn::Server::new(endpoint);
    (url, server)
}

fn tls_acceptor(cert: CertificateDer<'static>, key: &[u8]) -> TlsAcceptor {
    let provider = Arc::new(rustls::crypto::ring::default_provider());
    let config = rustls::ServerConfig::builder_with_provider(provider)
        .with_safe_default_protocol_versions()
        .unwrap()
        .with_no_client_auth()
        .with_single_cert(vec![cert], PrivateKeyDer::Pkcs8(key.to_vec().into()))
        .unwrap();
    TlsAcceptor::from(Arc::new(config))
}

async fn http_server(
    cert: CertificateDer<'static>,
    key: &[u8],
) -> (String, tokio::task::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("https://localhost:{}", listener.local_addr().unwrap().port());
    let acceptor = tls_acceptor(cert, key);
    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut stream = acceptor.accept(stream).await.unwrap();
        let mut request = Vec::new();
        while !request.ends_with(b"\r\n\r\n") {
            request.push(stream.read_u8().await.unwrap());
            assert!(request.len() < 16 * 1024);
        }
        let body = r#"{"items":[],"pageInfo":{}}"#;
        stream.write_all(format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len(),
        ).as_bytes()).await.unwrap();
        stream.shutdown().await.unwrap();
    });
    (url, server)
}

/// `name`'s full path in this integration binary, for re-running it alone in a child process.
fn test_name(name: &str) -> String {
    let module = module_path!().split_once("::").map_or("", |(_, module)| module);
    if module.is_empty() { name.to_owned() } else { format!("{module}::{name}") }
}
