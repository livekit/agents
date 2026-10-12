//! Workspace source checks. Invoked by scripts/check_layout.py, which also owns the dependency
//! and host-declaration rules (only `lk-agents-abi` declares `#[derive(Host)]` records).
mod architecture;

fn repo_root() -> Result<std::path::PathBuf, String> {
    Ok(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.."))
}

fn main() -> std::process::ExitCode {
    let root = std::env::args_os()
        .nth(1)
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| repo_root().unwrap());
    match architecture::check(&root) {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            std::process::ExitCode::FAILURE
        }
    }
}
