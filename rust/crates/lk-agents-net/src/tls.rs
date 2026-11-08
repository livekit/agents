use std::sync::{Arc, OnceLock};

use rustls::{ClientConfig, RootCertStore};

// Immutable TLS configuration may be inherited by forked children; sockets and pools may not.

/// The trust configuration could not be built: no usable root, or a client the TLS backend refused.
#[derive(Debug, Clone, thiserror::Error)]
#[error("{0}")]
pub struct TlsError(String);

impl From<crate::blocking_io::JobPanicked> for TlsError {
    fn from(error: crate::blocking_io::JobPanicked) -> Self {
        TlsError(error.to_string())
    }
}

static CONFIG: OnceLock<Result<Arc<ClientConfig>, TlsError>> = OnceLock::new();

/// The shared trust configuration. The first call performs blocking certificate I/O.
pub fn client_config() -> Result<Arc<ClientConfig>, TlsError> {
    CONFIG
        .get_or_init(|| {
            let native = rustls_native_certs::load_native_certs();
            for error in native.errors {
                log::warn!(target: lk_agents_abi::LOG_TARGET, "failed to load TLS root certificate: {error}");
            }
            let mut roots = RootCertStore::empty();
            let (accepted, rejected) = roots.add_parsable_certificates(native.certs);
            if accepted == 0 {
                return Err(TlsError("no usable TLS root certificates; check SSL_CERT_FILE/SSL_CERT_DIR or the system trust store".into()));
            }
            if rejected > 0 {
                log::warn!(target: lk_agents_abi::LOG_TARGET, "ignored {rejected} invalid TLS root certificates");
            }
            let provider = rustls::crypto::ring::default_provider();
            let _already_installed = provider.clone().install_default();
            let config = ClientConfig::builder_with_provider(Arc::new(provider))
                .with_safe_default_protocol_versions()
                .map_err(|error| TlsError(error.to_string()))?
                .with_root_certificates(roots)
                .with_no_client_auth();
            Ok(Arc::new(config))
        })
        .clone()
}

/// Load certificates off the async executor; later calls only clone the shared configuration.
pub async fn config() -> Result<Arc<ClientConfig>, TlsError> {
    if let Some(config) = CONFIG.get() {
        return config.clone();
    }
    crate::blocking_io::blocking_io().try_run((), client_config).await
}

/// An HTTP builder using the same trust configuration as the socket transports.
pub fn http_client_builder() -> Result<reqwest::ClientBuilder, TlsError> {
    let mut tls = (*client_config()?).clone();
    tls.alpn_protocols = vec![b"h2".to_vec(), b"http/1.1".to_vec()];
    Ok(reqwest::Client::builder().tls_backend_preconfigured(tls).dns_resolver(crate::dns::Resolver))
}

/// A client built from [`http_client_builder`]; its build failure is the TLS backend's refusal.
pub fn build(builder: reqwest::ClientBuilder) -> Result<reqwest::Client, TlsError> {
    builder.build().map_err(|error| TlsError(crate::http::error_chain(&error)))
}
