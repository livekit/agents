//! Upstream reset codes, written into the host's `worker_reset_codes` module. Each value is checked
//! against the protocol's own definition at compile time.

use livekit_protocol::agent_http::HttpStreamResetCode as Code;

#[crate::host(file(module = "worker_reset_codes"))]
const _: () = ();

#[crate::host(constant)]
pub const HSR_ABORT: i32 = 0;
#[crate::host(constant)]
pub const HSR_REFUSED: i32 = 1;
#[crate::host(constant)]
pub const HSR_INTERNAL: i32 = 2;
#[crate::host(constant)]
pub const HSR_TIMEOUT: i32 = 3;
#[crate::host(constant)]
pub const HSR_PROTOCOL: i32 = 4;

const _: () = assert!(
    HSR_ABORT == Code::HsrAbort as i32
        && HSR_REFUSED == Code::HsrRefused as i32
        && HSR_INTERNAL == Code::HsrInternal as i32
        && HSR_TIMEOUT == Code::HsrTimeout as i32
        && HSR_PROTOCOL == Code::HsrProtocol as i32
);
