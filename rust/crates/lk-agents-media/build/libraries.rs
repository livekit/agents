//! Static archive discovery accepts both Unix and MSVC FFmpeg install layouts.

use std::path::{Path, PathBuf};

pub fn archive(directory: &Path, name: &str) -> Option<PathBuf> {
    [format!("lib{name}.a"), format!("{name}.lib")]
        .into_iter()
        .map(|filename| directory.join(filename))
        .find(|path| path.is_file())
}
