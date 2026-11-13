//! The job channel's named pipe on Windows.
//!
//! Unverified locally (no Windows host); CI runs the channel tests on Windows.

use std::io;
use std::os::windows::io::{AsRawHandle, FromRawHandle, IntoRawHandle, OwnedHandle};

use tokio::net::windows::named_pipe::NamedPipeClient;
use windows_sys::Win32::Foundation::{
    GENERIC_READ, GENERIC_WRITE, HANDLE, INVALID_HANDLE_VALUE, LocalFree,
};
use windows_sys::Win32::Security::Authorization::{
    ConvertSidToStringSidW, ConvertStringSecurityDescriptorToSecurityDescriptorW, SDDL_REVISION_1,
};
use windows_sys::Win32::Security::{
    GetTokenInformation, PSECURITY_DESCRIPTOR, SECURITY_ATTRIBUTES, TOKEN_QUERY, TOKEN_USER,
    TokenUser,
};
use windows_sys::Win32::Storage::FileSystem::{
    CreateFileW, FILE_FLAG_FIRST_PIPE_INSTANCE, FILE_FLAG_OVERLAPPED, OPEN_EXISTING,
    PIPE_ACCESS_DUPLEX,
};
use windows_sys::Win32::System::Pipes::{
    CreateNamedPipeW, GetNamedPipeClientProcessId, PIPE_READMODE_BYTE, PIPE_REJECT_REMOTE_CLIENTS,
    PIPE_TYPE_BYTE, PIPE_WAIT,
};
use windows_sys::Win32::System::Threading::{GetCurrentProcess, OpenProcessToken};

fn check(ok: i32) -> io::Result<()> {
    if ok == 0 { Err(io::Error::last_os_error()) } else { Ok(()) }
}

/// A security descriptor that grants the current user alone access, freed on drop.
struct UserOnly(PSECURITY_DESCRIPTOR);

impl UserOnly {
    fn new() -> io::Result<Self> {
        let mut token: HANDLE = std::ptr::null_mut();
        // SAFETY: the current process pseudo-handle and a writable token slot.
        check(unsafe { OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &mut token) })?;
        // SAFETY: OpenProcessToken returned a new handle we own.
        let token = unsafe { OwnedHandle::from_raw_handle(token) };
        let mut len = 0;
        // SAFETY: a size query: no buffer, a writable length.
        unsafe {
            GetTokenInformation(token.as_raw_handle(), TokenUser, std::ptr::null_mut(), 0, &mut len)
        };
        let mut user = vec![0u64; (len as usize).div_ceil(8)];
        // SAFETY: an 8-byte aligned buffer of at least `len` bytes.
        check(unsafe {
            GetTokenInformation(
                token.as_raw_handle(),
                TokenUser,
                user.as_mut_ptr().cast(),
                len,
                &mut len,
            )
        })?;
        // SAFETY: the buffer now holds a TOKEN_USER whose SID points into it.
        let sid = unsafe { (*user.as_ptr().cast::<TOKEN_USER>()).User.Sid };
        let mut text = std::ptr::null_mut();
        // SAFETY: a valid SID; the string is LocalAlloc'd and freed below.
        check(unsafe { ConvertSidToStringSidW(sid, &mut text) })?;
        // SAFETY: a NUL-terminated UTF-16 string from ConvertSidToStringSidW.
        let sid = String::from_utf16_lossy(unsafe {
            std::slice::from_raw_parts(text, (0..).take_while(|&i| *text.add(i) != 0).count())
        });
        // SAFETY: allocated by ConvertSidToStringSidW.
        unsafe { LocalFree(text.cast()) };
        // Protected DACL: generic all to this user, nobody else.
        let sddl: Vec<u16> = format!("D:P(A;;GA;;;{sid})").encode_utf16().chain([0]).collect();
        let mut descriptor = std::ptr::null_mut();
        // SAFETY: a NUL-terminated SDDL string; the descriptor is LocalAlloc'd, freed on drop.
        check(unsafe {
            ConvertStringSecurityDescriptorToSecurityDescriptorW(
                sddl.as_ptr(),
                SDDL_REVISION_1,
                &mut descriptor,
                std::ptr::null_mut(),
            )
        })?;
        Ok(Self(descriptor))
    }
}

impl Drop for UserOnly {
    fn drop(&mut self) {
        // SAFETY: allocated by ConvertStringSecurityDescriptorToSecurityDescriptorW.
        unsafe { LocalFree(self.0) };
    }
}

/// A single-instance pipe with an unguessable name, only the current user may open, that
/// this process connects to itself at once: the connected client must be this process, or
/// the pair is refused (a process that raced the connect took the one instance).
pub fn pair() -> io::Result<(i64, i64)> {
    let mut random = [0u8; 16];
    getrandom::fill(&mut random).map_err(|error| io::Error::other(error.to_string()))?;
    let random: String = random.iter().map(|byte| format!("{byte:02x}")).collect();
    let name = format!(r"\\.\pipe\livekit-agents-{}-{random}", std::process::id());
    let name: Vec<u16> = name.encode_utf16().chain([0]).collect();
    let user_only = UserOnly::new()?;
    let security = SECURITY_ATTRIBUTES {
        nLength: std::mem::size_of::<SECURITY_ATTRIBUTES>() as u32,
        lpSecurityDescriptor: user_only.0,
        bInheritHandle: 0,
    };
    // SAFETY: a NUL-terminated name, a valid security descriptor; one instance.
    let server = unsafe {
        CreateNamedPipeW(
            name.as_ptr(),
            PIPE_ACCESS_DUPLEX | FILE_FLAG_FIRST_PIPE_INSTANCE | FILE_FLAG_OVERLAPPED,
            PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT | PIPE_REJECT_REMOTE_CLIENTS,
            1,
            64 << 10,
            64 << 10,
            0,
            &security,
        )
    };
    if server == INVALID_HANDLE_VALUE {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: CreateNamedPipeW returned a new handle we own.
    let server = unsafe { OwnedHandle::from_raw_handle(server) };
    // SAFETY: a NUL-terminated name; no sharing; a handle not inherited by children.
    let client = unsafe {
        CreateFileW(
            name.as_ptr(),
            GENERIC_READ | GENERIC_WRITE,
            0,
            std::ptr::null(),
            OPEN_EXISTING,
            FILE_FLAG_OVERLAPPED,
            std::ptr::null_mut(),
        )
    };
    if client == INVALID_HANDLE_VALUE {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: CreateFileW returned a new handle we own.
    let client = unsafe { OwnedHandle::from_raw_handle(client) };
    let mut pid = 0;
    // SAFETY: a valid server handle and a writable u32.
    check(unsafe { GetNamedPipeClientProcessId(server.as_raw_handle(), &mut pid) })?;
    if pid != std::process::id() {
        return Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            "another process connected to the IPC pipe",
        ));
    }
    Ok((server.into_raw_handle() as i64, client.into_raw_handle() as i64))
}

/// Either end: both are overlapped pipe handles, read and written the same way.
pub fn adopt(endpoint: i64) -> io::Result<NamedPipeClient> {
    if endpoint <= 0 {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "bad IPC endpoint"));
    }
    // SAFETY: the caller hands the handle's ownership over.
    unsafe { NamedPipeClient::from_raw_handle(endpoint as _) }
}
