//! The native artifact cache publishes only complete builds under an exclusive OS lock.

#[path = "cache.rs"]
mod cache;

use std::fs::{self, OpenOptions};
use std::io;
use std::path::Path;

fn valid(entry: &Path) -> bool {
    entry.join("library.a").is_file()
}

#[test]
fn keys_follow_contents_not_checkout_paths_or_file_times() {
    let first = tempfile::tempdir().unwrap();
    let second = tempfile::tempdir().unwrap();
    for root in [first.path(), second.path()] {
        fs::write(root.join("source.c"), b"source").unwrap();
    }
    let key = |path: &Path| {
        let mut key = cache::Key::new();
        key.input("compiler", b"compiler-v1");
        key.tree(path).unwrap();
        key.finish()
    };
    assert_eq!(key(first.path()), key(second.path()));
    fs::write(second.path().join("source.c"), b"changed").unwrap();
    assert_ne!(key(first.path()), key(second.path()));
}

#[test]
fn complete_artifacts_are_reused_and_missing_artifacts_are_rebuilt() {
    let root = tempfile::tempdir().unwrap();
    let build = |entry: &Path, _: &Path| fs::write(entry.join("library.a"), b"archive");
    let entry = cache::get_or_build(root.path(), "test", "key", valid, build).unwrap();
    cache::get_or_build(root.path(), "test", "key", valid, |_, _| panic!("cache miss")).unwrap();
    fs::remove_file(entry.join("library.a")).unwrap();
    cache::get_or_build(root.path(), "test", "key", valid, build).unwrap();
    assert!(valid(&entry));
}

#[test]
fn failed_builds_are_retried_without_reusing_partial_output() {
    let root = tempfile::tempdir().unwrap();
    let result = cache::get_or_build(root.path(), "test", "key", valid, |entry, _| {
        fs::write(entry.join("partial"), b"incomplete")?;
        Err(io::Error::other("interrupted build"))
    });
    assert!(result.is_err());
    cache::get_or_build(root.path(), "test", "key", valid, |entry, _| {
        assert!(!entry.join("partial").exists());
        fs::write(entry.join("library.a"), b"complete")
    })
    .unwrap();
}

#[test]
fn the_lock_is_held_until_the_artifacts_are_validated_and_published() {
    let root = tempfile::tempdir().unwrap();
    cache::get_or_build(root.path(), "test", "key", valid, |entry, _| {
        let competing =
            OpenOptions::new().read(true).write(true).open(root.path().join("test-key.lock"))?;
        assert!(matches!(competing.try_lock(), Err(fs::TryLockError::WouldBlock)));
        fs::write(entry.join("library.a"), b"complete")
    })
    .unwrap();
}

#[test]
fn successful_builds_without_expected_outputs_are_not_published() {
    let root = tempfile::tempdir().unwrap();
    let result = cache::get_or_build(root.path(), "test", "key", valid, |_, _| Ok(()));
    assert!(result.is_err());
    assert!(!root.path().join("test-key/.complete").exists());
}

#[test]
fn build_is_invisible_until_validated_then_published_at_the_final_prefix() {
    let root = tempfile::tempdir().unwrap();
    let published =
        cache::get_or_build(root.path(), "test", "key", valid, |staging, final_path| {
            assert_ne!(staging, final_path);
            assert!(!final_path.exists());
            fs::write(staging.join("library.a"), b"complete")?;
            fs::write(staging.join("prefix"), final_path.to_string_lossy().as_bytes())
        })
        .unwrap();
    assert!(valid(&published));
    assert_eq!(fs::read_to_string(published.join("prefix")).unwrap(), published.to_string_lossy());
}

#[path = "libraries.rs"]
mod libraries;

#[test]
fn cache_locations_follow_the_build_machine_and_ignore_relative_defaults() {
    let roots = tempfile::tempdir().unwrap();
    let home = roots.path().join("home with spaces");
    let xdg = roots.path().join("xdg");
    let local = roots.path().join("local app data");
    let suffix = Path::new("livekit-agents").join("native");
    assert_eq!(
        cache::default_directory("macos", Some(home.clone()), None, None),
        Some(home.join("Library").join("Caches").join(&suffix))
    );
    assert_eq!(
        cache::default_directory("windows", None, None, Some(local.clone())),
        Some(local.join(&suffix))
    );
    assert_eq!(
        cache::default_directory("linux", Some(home.clone()), Some(xdg.clone()), None),
        Some(xdg.join(&suffix))
    );
    assert_eq!(
        cache::default_directory("linux", Some(home.clone()), Some("relative".into()), None),
        Some(home.join(".cache").join(suffix))
    );
    assert_eq!(cache::default_directory("linux", None, None, None), None);
}

#[test]
fn unavailable_user_cache_falls_back_but_explicit_errors_are_reported() {
    let root = tempfile::tempdir().unwrap();
    let unavailable = root.path().join("a-file");
    fs::write(&unavailable, b"not a directory").unwrap();
    let fallback = root.path().join("target cache");
    assert_eq!(cache::directory(None, Some(&unavailable), &fallback).unwrap(), fallback);
    assert!(cache::directory(Some(&unavailable), None, &fallback).is_err());
    assert!(cache::directory(Some(Path::new("relative")), None, &fallback).is_err());
}

#[test]
fn static_archives_are_found_in_unix_and_msvc_layouts_with_spaces() {
    let root = tempfile::tempdir().unwrap();
    let lib = root.path().join("native libraries");
    fs::create_dir(&lib).unwrap();
    assert_eq!(libraries::archive(&lib, "avcodec"), None);
    let unix = lib.join("libavcodec.a");
    fs::write(&unix, b"archive").unwrap();
    assert_eq!(libraries::archive(&lib, "avcodec"), Some(unix.clone()));
    fs::remove_file(unix).unwrap();
    let msvc = lib.join("avcodec.lib");
    fs::write(&msvc, b"archive").unwrap();
    assert_eq!(libraries::archive(&lib, "avcodec"), Some(msvc));
}
