//! Deliberately failing uses for the I/O policy (`io/clippy.toml`); test_clippy.py compiles them.
#![allow(unused)]
pub fn raw_locks() {
    let _ = std::sync::Mutex::new(());
    let _ = std::sync::RwLock::new(());
    let _ = parking_lot::Mutex::new(());
    let _ = parking_lot::RwLock::new(());
    let _ = parking_lot::ReentrantMutex::new(());
    let _ = parking_lot::FairMutex::new(());
    let _: Option<lock_api::Mutex<parking_lot::RawMutex, ()>> = None;
}

pub async fn tokio_blocking_pool(handle: tokio::runtime::Handle) {
    let _ = tokio::task::spawn_blocking(|| ());
    let _ = handle.spawn_blocking(|| ());
    tokio::task::block_in_place(|| ());
    let _ = tokio::net::lookup_host("localhost:80").await;
    let _ = tokio::fs::read_to_string("fixture").await;
    let _: Option<tokio::fs::File> = None;
    let _ = tokio::io::stdin();
    let _ = tokio_tungstenite::connect_async("ws://localhost").await;
}
fn main() {}
