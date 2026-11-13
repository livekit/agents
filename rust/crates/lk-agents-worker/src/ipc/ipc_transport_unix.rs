//! The job channel's socket on Unix: a connected socket pair, adopted by the runtime.

use std::io;
use std::os::fd::{AsRawFd, FromRawFd, IntoRawFd, OwnedFd};
use std::os::unix::net::UnixStream;

pub fn pair() -> io::Result<(i64, i64)> {
    // std creates both ends close-on-exec.
    let (parent, child) = UnixStream::pair()?;
    for end in [&parent, &child] {
        no_sigpipe(end)?;
    }
    Ok((parent.into_raw_fd().into(), child.into_raw_fd().into()))
}

pub fn adopt(endpoint: i64) -> io::Result<tokio::net::UnixStream> {
    let fd = i32::try_from(endpoint)
        .ok()
        .filter(|fd| *fd >= 0)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "bad IPC endpoint"))?;
    // SAFETY: the caller hands the descriptor's ownership over.
    let fd = unsafe { OwnedFd::from_raw_fd(fd) };
    // The job process received its end inheritable (multiprocessing's pass_fds or
    // SCM_RIGHTS): a process it spawns must not keep the connection open.
    // SAFETY: plain flag update on an owned descriptor.
    if unsafe { libc::fcntl(fd.as_raw_fd(), libc::F_SETFD, libc::FD_CLOEXEC) } < 0 {
        return Err(io::Error::last_os_error());
    }
    let stream = UnixStream::from(fd);
    stream.set_nonblocking(true)?;
    tokio::net::UnixStream::from_std(stream)
}

/// macOS has no `MSG_NOSIGNAL`: without this, a write to a dead peer raises SIGPIPE, which
/// kills a process that does not ignore it. Linux std writes already pass `MSG_NOSIGNAL`.
fn no_sigpipe(_end: &UnixStream) -> io::Result<()> {
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        let on: libc::c_int = 1;
        // SAFETY: a valid socket and an int option of the declared size.
        let rc = unsafe {
            libc::setsockopt(
                _end.as_raw_fd(),
                libc::SOL_SOCKET,
                libc::SO_NOSIGPIPE,
                (&on as *const libc::c_int).cast(),
                std::mem::size_of::<libc::c_int>() as libc::socklen_t,
            )
        };
        if rc != 0 {
            return Err(io::Error::last_os_error());
        }
    }
    Ok(())
}
