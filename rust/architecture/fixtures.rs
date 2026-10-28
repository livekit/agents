#![allow(unused, deprecated)]
#![allow(clippy::disallowed_macros, reason = "legacy macro sites")]
use rand::{Rng, RngCore};
pub fn ambient(rng: &mut impl RngCore) {
    let _ = std::thread::spawn(|| {});
    let _ = std::time::Instant::now();
    let _ = std::time::SystemTime::now();
    let _ = std::collections::hash_map::RandomState::new();
    let _ = std::process::id();
    let _ = std::env::var("FIXTURE");
    let _ = rand::random::<u32>();
    let _ = rng.next_u64();
    let _ = rng.random::<u32>();
    let _ = uuid::Uuid::new_v4();
    let _ = std::sync::mpsc::channel::<()>();
    let _ = tokio::sync::mpsc::unbounded_channel::<()>();
    let _ = futures::channel::mpsc::unbounded::<()>();
    let _ = crossbeam_channel::unbounded::<()>();
    let _: std::collections::HashMap<u8, u8> = Default::default();
    let _: std::collections::HashSet<u8> = Default::default();
    let _ = std::sync::Mutex::new(());
    let _ = std::sync::RwLock::new(());
    let _ = tokio::sync::Mutex::new(());
    let _ = tokio::sync::RwLock::new(());
    let _ = parking_lot::Mutex::new(());
    let _ = parking_lot::RwLock::new(());
    let _ = std::sync::atomic::AtomicU64::new(0);
}
pub fn default_only() {
    let _ = std::env::var("DEFAULT_ONLY");
}
pub fn optional_native() {
    let _ = std::thread::Builder::new().spawn(|| {});
}
pub fn through_handle(handle: &tokio::runtime::Handle) {
    let _ = handle.spawn(async {});
}
pub async fn tasks() {
    tokio::spawn(async {});
    tokio::select! { _ = async {} => {} }
    tokio::join!(async {}, async {});
    tokio::time::sleep(std::time::Duration::ZERO).await;
}
#[allow(clippy::disallowed_methods, reason = "legacy")]
pub fn old() {
    std::thread::spawn(|| {});
    std::thread::spawn(|| {});
}

pub fn missing_exemption() {
    std::thread::spawn(|| {});
}
pub async fn macro_growth() {
    tokio::select! { _ = async {} => {} }
    tokio::select! { _ = async {} => {} }
}
fn main() {}

pub fn argument_exemption(
    #[allow(clippy::disallowed_types, reason = "ineffective argument permission")]
    input: tokio::sync::mpsc::UnboundedReceiver<()>,
) {
    drop(input);
}

pub async fn async_argument_exemption(
    #[allow(clippy::disallowed_types, reason = "ineffective async argument permission")]
    input: tokio::sync::mpsc::UnboundedReceiver<()>,
) {
    drop(input);
}

#[allow(clippy::disallowed_types, reason = "function permission covers the signature")]
pub async fn function_exemption(input: tokio::sync::mpsc::UnboundedReceiver<()>) {
    drop(input);
}
