//! One trust store for every framework HTTP, WebSocket and WebTransport client.
//!
//! The first use snapshots the system roots, or only `SSL_CERT_FILE`/`SSL_CERT_DIR`
//! when configured. Immutable TLS configuration can be inherited by a forkserver's
//! children; sockets and connection pools remain process-local.

use std::sync::{Arc, OnceLock};

use rustls::{ClientConfig, RootCertStore};

static CONFIG: OnceLock<Result<Arc<ClientConfig>, String>> = OnceLock::new();

/// The shared trust configuration. The first call performs blocking certificate I/O.
pub fn client_config() -> Result<Arc<ClientConfig>, String> {
    CONFIG
        .get_or_init(|| {
            let native = rustls_native_certs::load_native_certs();
            for error in native.errors {
                log::warn!("failed to load TLS root certificate: {error}");
            }
            let mut roots = RootCertStore::empty();
            let (accepted, rejected) = roots.add_parsable_certificates(native.certs);
            if accepted == 0 {
                return Err("no usable TLS root certificates; check SSL_CERT_FILE/SSL_CERT_DIR or the system trust store".into());
            }
            if rejected > 0 {
                log::warn!("ignored {rejected} invalid TLS root certificates");
            }
            let provider = rustls::crypto::ring::default_provider();
            let _already_installed = provider.clone().install_default();
            let config = ClientConfig::builder_with_provider(Arc::new(provider))
                .with_safe_default_protocol_versions()
                .map_err(|error| error.to_string())?
                .with_root_certificates(roots)
                .with_no_client_auth();
            Ok(Arc::new(config))
        })
        .clone()
}

/// Load certificates off the async executor; later calls only clone the shared configuration.
pub async fn config() -> Result<Arc<ClientConfig>, String> {
    if let Some(config) = CONFIG.get() {
        return config.clone();
    }
    crate::blocking_io::blocking_io()
        .run((), client_config)
        .await
        .map_err(|error| error.to_string())?
}

/// An HTTP builder using the same trust configuration as the socket transports.
pub fn http_client_builder() -> Result<reqwest::ClientBuilder, String> {
    let mut tls = (*client_config()?).clone();
    tls.alpn_protocols = vec![b"h2".to_vec(), b"http/1.1".to_vec()];
    Ok(reqwest::Client::builder().tls_backend_preconfigured(tls).dns_resolver(crate::dns::Resolver))
}
