//! The crate's one integration binary: the A2A door, end of turn, MCP, media, TLS and speech
//! through the public API.

#[path = "a2a/attachments_tests.rs"]
mod a2a_attachments;
#[path = "a2a/door_tests.rs"]
mod a2a_door;
#[path = "a2a/turns_tests.rs"]
mod a2a_turns;
#[path = "eot_tests.rs"]
mod eot;
#[path = "images_tests.rs"]
mod images;
#[path = "mcp/lifecycle_tests.rs"]
mod mcp_lifecycle;
#[path = "media_tests.rs"]
mod media;
#[path = "tls_tests.rs"]
mod tls;
#[path = "tts_tests.rs"]
mod tts;
#[path = "tts_live_tests.rs"]
mod tts_live;
