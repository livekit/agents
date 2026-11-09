//! Integration binary for tests that need process isolation or other crates' public APIs.

#[path = "eot_tests.rs"]
mod eot;
#[path = "images_tests.rs"]
mod images;
#[path = "media_tests.rs"]
mod media;
#[path = "tls_tests.rs"]
mod tls;
#[path = "tts_tests.rs"]
mod tts;
#[path = "tts_live_tests.rs"]
mod tts_live;
