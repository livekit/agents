//! Deliberately failing uses for the I/O policy (`io/clippy.toml`); test_clippy.py compiles them.
#![allow(unused)]
pub fn raw_locks() {
    let _ = std::sync::Mutex::new(());
    let _ = std::sync::RwLock::new(());
    let _ = parking_lot::Mutex::new(());
    let _ = parking_lot::RwLock::new(());
}
fn main() {}
