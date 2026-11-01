//! Native deadline enforcement does not depend on a responsive Python loop or stderr reader.

use std::process::{Command, Output, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use lk_agents_abi::cli::CliShutdown as _;
use lk_agents_cli::shutdown::Shutdown;

fn watchdog_process(mode: &str, stderr: Stdio) -> Output {
    let mut child = Command::new(std::env::current_exe().expect("test executable"))
        .args(["--exact", "watchdog_child", "--ignored", "--nocapture"])
        .env("LK_TEST_CLI_CLOSE", mode)
        .stdout(Stdio::piped())
        .stderr(stderr)
        .spawn()
        .expect("launch watchdog child");
    let deadline = Instant::now() + Duration::from_secs(12);
    loop {
        if child.try_wait().expect("observe watchdog child").is_some() {
            return child.wait_with_output().expect("collect reaped watchdog child");
        }
        if Instant::now() >= deadline {
            child.kill().expect("kill timed-out watchdog child");
            let output = child.wait_with_output().expect("reap timed-out watchdog child");
            panic!("watchdog failed to terminate its process: {output:?}");
        }
        thread::sleep(Duration::from_millis(10));
    }
}

#[test]
fn incomplete_loop_cleanup_terminates_nonzero() {
    let output = watchdog_process("pending", Stdio::piped());
    assert_eq!(output.status.code(), Some(1));
    #[cfg(unix)]
    assert_eq!(
        String::from_utf8_lossy(&output.stderr).matches("shutdown deadline exceeded").count(),
        1
    );
}

#[test]
fn completed_loop_cleanup_disarms_the_native_deadline() {
    let output = watchdog_process("closed", Stdio::piped());
    assert!(output.status.success(), "{output:?}");
    assert!(output.stderr.is_empty());
}

#[cfg(unix)]
#[test]
fn saturated_stderr_cannot_hold_the_cleanup_owner_alive() {
    use std::io::{Read, Write};
    use std::os::fd::OwnedFd;
    use std::os::unix::net::UnixStream;

    let (mut reader, mut writer) = UnixStream::pair().expect("stderr pair");
    writer.set_nonblocking(true).expect("nonblocking stderr");
    loop {
        match writer.write(&[b'x'; 4096]) {
            Ok(written) => assert!(written > 0),
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => break,
            Err(error) => panic!("fill stderr: {error}"),
        }
    }
    writer.set_nonblocking(false).expect("blocking stderr");
    let output = watchdog_process("pending", Stdio::from(OwnedFd::from(writer)));
    assert_eq!(output.status.code(), Some(1));
    let mut retained = Vec::new();
    reader.read_to_end(&mut retained).expect("closed stderr");
    assert!(!retained.is_empty());
}

#[test]
#[ignore = "subprocess entrypoint for native watchdog tests"]
fn watchdog_child() {
    let mode = std::env::var("LK_TEST_CLI_CLOSE").expect("watchdog scenario");
    let signals = Shutdown::new(0.2, 0.0).unwrap();
    signals.begin();
    if mode == "closed" {
        signals.closed();
        thread::sleep(Duration::from_millis(400));
    } else {
        thread::sleep(Duration::from_secs(30));
    }
}
