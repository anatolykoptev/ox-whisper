//! Scratch files that clean up after themselves.
//!
//! Uploads and ffmpeg conversions live in a tmpfs that is charged to the
//! container's memory limit, so a file nobody removes is a slow outage:
//! `/health` stays green while every request fails once the tmpfs is full.
//! [`TempFile`] makes the leak impossible at the source: whoever owns the
//! value owns the file, and dropping it — normal return, early `?`, or an axum
//! handler future cancelled by a client that went away — removes the file.
//! (A panic aborts the process in release builds, `panic = "abort"`, so the
//! tmpfs dies with the container; no unwinding is relied on.)

use std::io;
use std::path::{Path, PathBuf};

/// A file removed when the value is dropped.
#[derive(Debug)]
pub struct TempFile {
    path: PathBuf,
}

impl TempFile {
    /// Takes ownership of `path`: it is removed on drop (a missing file is
    /// fine). Taken *before* the file is written, so a failed or partial write
    /// is removed too.
    pub fn own(path: PathBuf) -> Self {
        Self { path }
    }

    /// Creates `<dir>/<uuid>.<ext>` holding `data`. `ext` is client input
    /// (an upload's file name), so it is reduced to a short ASCII-alphanumeric
    /// suffix; anything else becomes `bin`, never `wav`: `ensure_wav` trusts a
    /// `.wav` suffix and skips ffmpeg, so a mangled name (`voice.og_g`) must
    /// not turn Ogg bytes into a "WAV".
    pub fn create(dir: &Path, ext: &str, data: &[u8]) -> io::Result<Self> {
        let file = Self::own(dir.join(format!("{}.{}", uuid::Uuid::new_v4(), safe_ext(ext))));
        std::fs::write(&file.path, data)?;
        Ok(file)
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        match std::fs::remove_file(&self.path) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::NotFound => {}
            Err(e) => tracing::warn!("temp file {} not removed: {e}", self.path.display()),
        }
    }
}

fn safe_ext(ext: &str) -> &str {
    if !ext.is_empty() && ext.len() <= 10 && ext.bytes().all(|b| b.is_ascii_alphanumeric()) {
        ext
    } else {
        "bin"
    }
}

#[cfg(test)]
pub(crate) fn scratch_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "oxw-{name}-{}-{}",
        std::process::id(),
        uuid::Uuid::new_v4()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Names of the files in `dir`.
#[cfg(test)]
pub(crate) fn listing(dir: &Path) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    names
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_file_exists_while_owned_and_is_gone_after_drop() {
        let dir = scratch_dir("tmpfile");
        let file = TempFile::create(&dir, "ogg", b"audio").unwrap();
        let path = file.path().to_path_buf();
        assert_eq!(std::fs::read(&path).unwrap(), b"audio");
        assert!(path.to_string_lossy().ends_with(".ogg"));
        drop(file);
        assert!(listing(&dir).is_empty(), "file survived its owner");
        std::fs::remove_dir(&dir).unwrap();
    }

    #[test]
    fn dropping_an_already_removed_file_is_not_an_error() {
        let dir = scratch_dir("tmpfile-gone");
        let file = TempFile::create(&dir, "wav", b"x").unwrap();
        std::fs::remove_file(file.path()).unwrap();
        drop(file);
        std::fs::remove_dir(&dir).unwrap();
    }

    #[test]
    fn a_failed_write_leaves_nothing_behind() {
        // The directory does not exist, so the write fails; nothing to remove,
        // and the owner must cope.
        let dir = std::env::temp_dir().join(format!("oxw-nodir-{}", uuid::Uuid::new_v4()));
        assert!(TempFile::create(&dir, "wav", b"x").is_err());
    }

    #[test]
    fn client_supplied_extensions_cannot_escape_the_directory() {
        for evil in [
            "../../etc/passwd",
            "a/b",
            "wav\0",
            "",
            "waaaaaaaaaaaaaaaav",
            "é",
        ] {
            assert_eq!(safe_ext(evil), "bin", "{evil:?}");
        }
        assert_eq!(safe_ext("mp3"), "mp3");
        assert_eq!(safe_ext("M4A"), "M4A");
    }
}
